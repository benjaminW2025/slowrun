import ast
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path
import unittest

import torch

from diagnostics import probe_episode, preview_muon
from test_upstream_audit import load, LocalOnly
from test_virtual_rank import FixedScaler
from virtual_rank import collect_virtual_gradients, select_trial_parameters, validate_gradients


def assert_tree(test, actual, expected):
    if isinstance(expected, torch.Tensor): torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        test.assertEqual(actual.keys(), expected.keys())
        for k in expected: assert_tree(test, actual[k], expected[k])
    elif isinstance(expected, (list, tuple)):
        test.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected): assert_tree(test, a, b)
    else: test.assertEqual(actual, expected)


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads(); torch.set_num_threads(2)
        self.ns = load(ast.parse(Path(__file__).with_name('train.py').read_text()), LocalOnly())
        torch.manual_seed(42)
        self.model = self.ns['GPT'](self.ns['GPTConfig'](vocab_size=64, dropout=.2, xsa_mode='off'))
        self.model.init_weights(); self.optimizer = self.model.setup_optimizer()
        self.params = list(self.model.parameters())
        self.batches = [(torch.randint(64, (1, 16)), torch.randint(64, (1, 16))) for _ in range(4)]
        self.forward = lambda x, y: self.model(x, y, num_iterations=2)
        # Populate real momentum/variance buffers before probing; initialization alone is insufficient.
        for _ in range(2):
            self.forward(*self.batches[0]).backward(); self.optimizer.step(); self.model.zero_grad(set_to_none=True)
        iterator = iter(self.batches[1:] + [self.batches[0]])
        self.capture = ([], [])
        collect_virtual_gradients(self.model, self.params, [], *self.batches[0],
                                  lambda: (*next(iterator), 1), self.forward, FixedScaler(128),
                                  2, 1, .5, capture_episode=self.capture)

    def tearDown(self): torch.set_num_threads(self.threads)

    def snapshot(self):
        return ([p.detach().clone() for p in self.params],
                [None if p.grad is None else p.grad.clone() for p in self.params],
                copy.deepcopy(self.optimizer.state_dict()), torch.get_rng_state().clone(),
                [m.training for m in self.model.modules()])

    def check_snapshot(self, before):
        after = self.snapshot()
        for a, b in zip(after, before): assert_tree(self, a, b)

    def test_recurrent_autocast_keeps_all_selected_matrix_gradients(self):
        # Use the cache setting from the production context, with CPU BF16 for
        # this portable regression. CUDA FP16 still requires a GPU smoke test.
        tree = ast.parse(Path(__file__).with_name('train.py').read_text())
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'autocast_ctx' for t in n.targets))
        call = assignment.value.body
        cache = next(k.value.value for k in call.keywords if k.arg == 'cache_enabled')
        self.assertFalse(cache)
        for mode in ('qk', 'v', 'mlp'):
            def forward(x, y):
                with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=cache):
                    return self.model(x, y, num_iterations=2)
            records = probe_episode(self.model, self.optimizer, self.model.named_parameters(),
                                    *self.capture, forward, self.ns['muon_step_fused'], 128,
                                    modes=(mode,), radii=(.1,), alignment_only=True)
            self.assertGreater(records[0]['gradient_a_norm'], 0)
            self.assertGreater(records[0]['gradient_b_norm'], 0)
            self.assertGreater(records[0]['trials'][0]['b_prime_norm'], 0)
            # The same production recurrence silently lost ALL these gradients
            # with the old cached context; now diagnostics must reject it.
            def broken_forward(x, y):
                with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=True):
                    return self.model(x, y, num_iterations=2)
            with self.assertRaisesRegex(RuntimeError, 'disconnected'):
                probe_episode(self.model, self.optimizer, self.model.named_parameters(),
                              *self.capture, broken_forward, self.ns['muon_step_fused'], 128,
                              modes=(mode,), radii=(.1,), alignment_only=True)

    def test_recurrent_autocast_training_backpropagates_to_matrices(self):
        self.model.zero_grad(set_to_none=True)
        with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
            loss = self.model(*self.batches[0], num_iterations=2)
        (loss * 128).backward()
        ids = {id(p) for g in self.optimizer.param_groups if g['kind'] == 'muon' for p in g['params']}
        for mode in ('qk', 'v', 'mlp'):
            selected = select_trial_parameters(self.model.named_parameters(), ids, mode)
            self.assertTrue(all(p.grad is not None for p in selected))
            self.assertGreater(sum(float(p.grad.norm()) for p in selected), 0)

    def test_cpu_fp16_scaler_all_arms_one_and_two_pass_updates(self):
        # Actual FP16 casts, half rotary buffers and GradScaler, on CPU. This
        # exercises precision/scale semantics but does not validate CUDA kernels.
        for mode in ('off', 'qk', 'v', 'mlp'):
            with self.subTest(mode=mode):
                ns = load(ast.parse(Path(__file__).with_name('train.py').read_text()), LocalOnly())
                ns['ACTIVATION_DTYPE'] = torch.float16
                torch.manual_seed(42)
                model = ns['GPT'](ns['GPTConfig'](vocab_size=64, dropout=0., xsa_mode='off'))
                model.init_weights()
                optimizer = model.setup_optimizer()
                params = list(model.parameters())
                ids = {id(p) for group in optimizer.param_groups for p in group['params']}
                muon_ids = {id(p) for group in optimizer.param_groups if group['kind'] == 'muon' for p in group['params']}
                trial = select_trial_parameters(model.named_parameters(), muon_ids, mode)
                scaler = torch.amp.GradScaler('cpu', init_scale=128)
                for iterations in (1, 2, 2):
                    def forward(x, y):
                        with torch.autocast('cpu', dtype=torch.float16, cache_enabled=False):
                            return model(x, y, num_iterations=iterations)
                    batch_iter = iter(self.batches[1:] + [self.batches[0]])
                    loss, *_ = collect_virtual_gradients(
                        model, params, trial, *self.batches[0], lambda: (*next(batch_iter), 1),
                        forward, scaler, 2, 1, .5)
                    scaler.unscale_(optimizer)
                    self.assertTrue(bool(torch.isfinite(loss)))
                    validate_gradients(model.named_parameters(), ids)
                    if iterations == 2:
                        for selected_mode in ('qk', 'v', 'mlp'):
                            selected = select_trial_parameters(model.named_parameters(), muon_ids, selected_mode)
                            self.assertGreater(sum(float(p.grad.norm()) for p in selected), 0)
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer); scaler.update()
                    self.assertEqual(previous_scale, scaler.get_scale())
                    self.assertTrue(all(bool(torch.isfinite(p).all()) for p in params))
                    model.zero_grad(set_to_none=True)

    def test_probe_preserves_training_state_and_next_update(self):
        before = self.snapshot()
        records = probe_episode(self.model, self.optimizer, self.model.named_parameters(),
                                *self.capture, self.forward, self.ns['muon_step_fused'], 128,
                                modes=('mlp', 'qk', 'v'), radii=(.01, .1))
        self.check_snapshot(before)
        self.assertEqual([r['mode'] for r in records], ['mlp', 'qk', 'v'])
        for r in records:
            self.assertEqual(len(r['trials']), 10)
            for t in r['trials']:
                self.assertAlmostEqual(t['b_loss'] - r['b_loss'], t['b_loss_change'])
                if t['radius'] is not None: self.assertAlmostEqual(t['step_norm'], t['radius'], places=5)
        # Apply a true update after probing and compare to the same update from the saved pre-probe state.
        for p in self.params:
            if p.grad is not None: p.grad.div_(128)
        self.optimizer.step()
        after = [p.detach().clone() for p in self.params]
        with torch.no_grad():
            for p, value in zip(self.params, before[0]): p.copy_(value)
        self.optimizer.load_state_dict(before[2])
        for p, g in zip(self.params, before[1]): p.grad = None if g is None else g / 128
        self.optimizer.step()
        for p, value in zip(self.params, after): torch.testing.assert_close(p, value, rtol=0, atol=0)

    def test_preview_matches_actual_muon_step_with_live_state_and_decay(self):
        muon_ids = {id(p) for g in self.optimizer.param_groups if g['kind'] == 'muon' for p in g['params']}
        selected = select_trial_parameters(self.model.named_parameters(), muon_ids, 'qk')
        gradients = [torch.randn_like(p) for p in selected]
        before = self.snapshot()
        directions, decay, masks = preview_muon(self.optimizer, selected, gradients, self.ns['muon_step_fused'])
        self.check_snapshot(before)
        initial = [p.detach().clone() for p in selected]
        self.model.zero_grad(set_to_none=True)
        for p, g in zip(selected, gradients): p.grad = g.clone()
        self.optimizer.step()
        for p, original, d, w in zip(selected, initial, directions, decay):
            torch.testing.assert_close(p, original + d + w, rtol=1e-5, atol=2e-7)

    def test_probe_exception_restores_weights_gradients_modes_rng(self):
        before = self.snapshot()
        calls = 0
        def fail_during_trial(x, y):
            nonlocal calls
            calls += 1
            # One A and B backward precede the first perturbed B evaluation.
            if calls == 3: raise RuntimeError('deliberate probe failure')
            return self.forward(x, y)
        with self.assertRaisesRegex(RuntimeError, 'deliberate probe failure'):
            probe_episode(self.model, self.optimizer, self.model.named_parameters(), *self.capture,
                          fail_during_trial, self.ns['muon_step_fused'], 128)
        self.check_snapshot(before)

    def test_alignment_and_transfer_values_against_analytic_quadratic(self):
        class Quadratic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block = torch.nn.Module()
                self.block.mlp = torch.nn.Linear(2, 1, bias=False)
                with torch.no_grad(): self.block.mlp.weight.copy_(torch.tensor([[.2, .3]]))
            def forward(self, x, y): return (self.block.mlp(x) - y).square().mean()
        model = Quadratic(); p = model.block.mlp.weight
        optimizer = self.ns['DistMuonAdamW']([dict(kind='muon', params=[p], lr=.01,
                                                  momentum=.9, beta2=.95, weight_decay=.2, ns_steps=5)])
        p.grad = torch.tensor([[-5.2, 0.]]) * 128
        x = torch.tensor([[1., 0.]])
        records = probe_episode(model, optimizer, model.named_parameters(),
                                [(x, torch.tensor([[1.]]))], [(x, torch.tensor([[2.]]))],
                                model, self.ns['muon_step_fused'], 128, radii=(.1, .5))
        record = records[0]
        self.assertAlmostEqual(record['cosine_a_b'], 1, places=5)
        self.assertAlmostEqual(record['gradient_a_norm'], 1.6, places=5)
        self.assertAlmostEqual(record['gradient_b_norm'], 3.6, places=5)
        trial = next(t for t in record['trials'] if t['kind'] == 'raw_a' and t['radius'] == .1)
        self.assertAlmostEqual(trial['b_loss_change'], -.35, places=5)
        self.assertAlmostEqual(trial['first_order_b_loss_change'], -.36, places=5)
        self.assertAlmostEqual(trial['nonlinear_residual'], .01, places=5)
        self.assertAlmostEqual(trial['cosine_a_b_prime'], 1, places=5)
        self.assertAlmostEqual(trial['relative_b_gradient_change'], .2 / 3.6, places=5)
        self.assertAlmostEqual(trial['cosine_a_b_correction'], -1, places=5)
        self.assertAlmostEqual(trial['cosine_b_b_prime'], 1, places=5)
        self.assertAlmostEqual(trial['b_prime_norm'], 3.4, places=5)
        self.assertAlmostEqual(trial['b_prime_norm_ratio'], 3.4 / 3.6, places=5)

    def test_alignment_only_restores_state_and_handles_zero_gradient(self):
        muon_ids = {id(p) for g in self.optimizer.param_groups if g['kind'] == 'muon' for p in g['params']}
        selected = select_trial_parameters(self.model.named_parameters(), muon_ids, 'qk')
        # Zero attention output projection blocks QK gradients at both theta and theta'.
        with torch.no_grad():
            for block in self.model.transformer.h: block.attn.c_proj.weight.zero_()
        before = self.snapshot()
        records = probe_episode(self.model, self.optimizer, self.model.named_parameters(),
                                *self.capture, self.forward, None, 128,
                                modes=('qk',), radii=(.1,), alignment_only=True)
        self.check_snapshot(before)
        m, t = records[0], records[0]['trials'][0]
        self.assertIsNone(m['cosine_a_b'])
        self.assertEqual(t['step_norm'], 0)
        self.assertEqual(t['b_loss_change'], 0)
        self.assertIsNone(t['cosine_b_b_prime'])
        self.assertIsNone(t['relative_b_gradient_change'])
        self.assertEqual(t['b_correction_norm'], 0)

    def test_actual_profile_only_block_writes_all_groups_without_update(self):
        tree = ast.parse(Path(__file__).with_name('train.py').read_text())
        loop = next(n for n in tree.body if isinstance(n, ast.While))
        gate = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'capture_episodes is not None')
        self.model.zero_grad(set_to_none=True)
        before = self.snapshot()
        with tempfile.TemporaryDirectory() as run_dir:
            ns = dict(torch=torch, probe_episode=probe_episode, orig_model=self.model,
                      optimizer=self.optimizer, capture_episodes=[self.capture],
                      probe_modes=('qk', 'v', 'mlp'), probe_radii=(.1, .5),
                      step=2, current_epoch=1, active_num_iterations=2, TRAIN_BACKPROP_ITERATIONS=1,
                      muon_step_fused=self.ns['muon_step_fused'], autocast_ctx=nullcontext(),
                      scaler=SimpleNamespace(get_scale=lambda: 128), MATRIX_LR=.032, lrm=1., wd=.8,
                      synchronize=lambda: None, print0=lambda *a, **k: None,
                      args=SimpleNamespace(probe_pairs=1, probe_pair_offset=0, probe_alignment_only=True, probe_only=True,
                                           mg_mode='off', precision='fp32', init_model='checkpoint.pt'),
                      time=time, os=os, json=json, sys=sys, master_process=True, run_dir=run_dir,
                      dist=SimpleNamespace(is_initialized=lambda: False),
                      wandb_run=SimpleNamespace(finish=lambda: None))
            with self.assertRaises(SystemExit) as exit:
                exec(compile(ast.Module([gate], type_ignores=[]), 'profile_only_block', 'exec'), ns)
            self.assertEqual(exit.exception.code, 0)
            self.check_snapshot(before)
            path = Path(run_dir) / 'probes.jsonl'
            record = json.loads(path.read_text())
            self.assertEqual([m['mode'] for m in record['measurements']], ['qk', 'v', 'mlp'])
            self.assertEqual(record['episodes'][0]['sequences_a'], 1)
            summary = subprocess.check_output([sys.executable, str(Path(__file__).with_name('summarize_probes.py')),
                                               str(path), '--alignment', '--aggregate'], text=True)
            self.assertIn('cos(B,Bprime)', summary)
            self.assertIn('B-hurt=', summary)

    def test_standalone_pair_offset_selects_new_batches_without_backward(self):
        tree = ast.parse(Path(__file__).with_name('train.py').read_text())
        loop = next(n for n in tree.body if isinstance(n, ast.While))
        gate = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'args.probe_only and args.probe_alignment_only')
        captures = [([], []), ([], [])]
        batches = [(torch.tensor([[i]]), torch.tensor([[i + 1]]), 1) for i in range(12)]
        ns = dict(args=SimpleNamespace(probe_only=True, probe_alignment_only=True, probe_pair_offset=3),
                  torch=torch, microsteps_per_half=1, capture_episodes=captures,
                  x=batches[0][0], y=batches[0][1], train_loader=iter(batches[1:]))
        exec(compile(ast.Module([gate], type_ignores=[]), 'pair_offset_capture', 'exec'), ns)
        actual = [[int(half[0][0].item()) for half in episode] for episode in captures]
        self.assertEqual(actual, [[6, 7], [8, 9]])
        self.assertEqual(ns['x'].item(), 10)

    def test_multiple_episode_capture_matches_actual_batches(self):
        captures = [([], []), ([], [])]
        iterator = iter(self.batches[1:] + [self.batches[0]])
        collect_virtual_gradients(self.model, self.params, [], *self.batches[0],
                                  lambda: (*next(iterator), 1), self.forward, FixedScaler(128),
                                  2, 1, .5, capture_episodes=captures)
        for episode, originals in zip(captures, (self.batches[:2], self.batches[2:])):
            for half, original in zip(episode, originals):
                self.assertEqual(len(half), 1)
                for a, b in zip(half[0], original): torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_capture_does_not_change_training_or_consume_extra_batches(self):
        self.assertEqual(len(self.capture[0]), 1); self.assertEqual(len(self.capture[1]), 1)
        for captured, original in zip((self.capture[0][0], self.capture[1][0]), self.batches[:2]):
            for a, b in zip(captured, original): torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.model.zero_grad(set_to_none=True)
        # Replay the collector with the same RNG and inputs, without capture.
        rng = torch.get_rng_state().clone()
        iterator = iter(self.batches[1:] + [self.batches[0]])
        collect_virtual_gradients(self.model, self.params, [], *self.batches[0],
                                  lambda: (*next(iterator), 1), self.forward, FixedScaler(128), 2, 1, .5)
        expected = [None if p.grad is None else p.grad.clone() for p in self.params]
        end_rng = torch.get_rng_state().clone(); torch.set_rng_state(rng)
        iterator = iter(self.batches[1:] + [self.batches[0]])
        captured = ([], [])
        collect_virtual_gradients(self.model, self.params, [], *self.batches[0],
                                  lambda: (*next(iterator), 1), self.forward, FixedScaler(128), 2, 1, .5,
                                  capture_episode=captured)
        for p, g in zip(self.params, expected):
            if g is not None: torch.testing.assert_close(p.grad, g, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), end_rng, rtol=0, atol=0)

if __name__ == '__main__': unittest.main()
