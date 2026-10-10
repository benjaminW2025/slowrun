import ast
import copy
from pathlib import Path
import unittest

import torch

from diagnostics import probe_episode, preview_muon
from test_upstream_audit import load, LocalOnly
from test_virtual_rank import FixedScaler
from virtual_rank import collect_virtual_gradients, select_trial_parameters


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
