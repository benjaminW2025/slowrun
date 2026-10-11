import ast
import copy
import math
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

from diagnostics import preview_muon
from transfer import agreement_component, probe_transfer_episode, _evaluate_c
from test_upstream_audit import load, LocalOnly
from test_diagnostics import assert_tree
from summarize_probes import summarize_alignment


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads(); torch.set_num_threads(2)
        self.ns = load(ast.parse(Path(__file__).with_name('train.py').read_text()), LocalOnly())
        self.model = torch.nn.Module()
        self.model.block = torch.nn.Module()
        self.model.block.mlp = torch.nn.Linear(2, 1, bias=False)
        self.p = self.model.block.mlp.weight
        with torch.no_grad(): self.p.copy_(torch.tensor([[.2, .3]]))
        self.optimizer = self.ns['DistMuonAdamW']([dict(kind='muon', params=[self.p], lr=.01,
                                                     momentum=.9, beta2=.95, weight_decay=.2, ns_steps=5)])
        self.x = torch.eye(2)
        self.a = [(self.x, torch.tensor([[-1.8], [-5.7]]))]  # grad A=(2,6)
        self.b = [(self.x, torch.tensor([[-3.8], [2.3]]))]   # grad B=(4,-2)
        self.c = [(self.x, torch.tensor([[1.2], [-.7]]))]   # grad C=(-1,1)
        self.forward = lambda x, y: (self.model.block.mlp(x) - y).square().mean()

    def tearDown(self): torch.set_num_threads(self.threads)

    def snapshot(self):
        return (self.p.detach().clone(), None if self.p.grad is None else self.p.grad.clone(),
                copy.deepcopy(self.optimizer.state_dict()), torch.get_rng_state().clone(),
                [m.training for m in self.model.modules()])

    def run_probe(self, c=None, forward=None):
        return probe_transfer_episode(self.model, self.optimizer, self.model.named_parameters(),
                                      self.a, self.b, forward or self.forward, self.ns['muon_step_fused'],
                                      128, modes=('mlp',), radii=(.1, .5),
                                      c_batches=self.c if c is None else c, transfer_radius=.1)[0]

    def test_summary_does_not_mix_appended_probe_events(self):
        measurement = self.run_probe()
        records = [dict(update=1, measurements=[measurement]), dict(update=1, measurements=[measurement])]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'probes.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in records))
            output = io.StringIO()
            with redirect_stdout(output): summarize_alignment([path], aggregate=True)
            lines = [line for line in output.getvalue().splitlines() if 'B-hurt=' in line]
            self.assertEqual(len(lines), 4)  # Two radii per distinct event.
            self.assertTrue(all('1 pairs' in line for line in lines))

    def test_agreement_exact_decomposition_and_coordinate_signs(self):
        a, b = torch.tensor([[2., 6., -5., 0.]]), torch.tensor([[4., -2., -3., 1.]])
        shared, = agreement_component([a], [b])
        torch.testing.assert_close(shared, torch.tensor([[2., 0., -3., 0.]]))
        paired = (a + b) / 2
        torch.testing.assert_close(shared + (paired - shared), paired)

    def test_analytic_c_transfer_and_matched_norms(self):
        before = self.snapshot(); record = self.run_probe()
        assert_tree(self, self.snapshot(), before)
        self.assertAlmostEqual(record['c_loss'], 1., places=6)
        rows = {(r['source'], r['transform']): r for r in record['transfer_baselines']}
        self.assertAlmostEqual(rows['agreement', 'raw']['c_loss_change'], .105, places=5)
        self.assertAlmostEqual(rows['remainder', 'raw']['c_loss_change'], -.1/math.sqrt(5)+.005, places=5)
        for trial in record['trials']:
            # Hessian of each quadratic is identity, so adapted B = B - r*A/||A||.
            ga, gb = torch.tensor([[2., 6.]]), torch.tensor([[4., -2.]])
            bp = gb - trial['radius'] * ga / ga.norm()
            self.assertAlmostEqual(trial['b_prime_norm'], float(bp.norm()), places=5)
            self.assertEqual(len(trial['transfer']), 6)
            rows.update({(f"{trial['radius']}:{r['source']}", r['transform']): r for r in trial['transfer']})
        for row in rows.values():
            if row['no_op']:
                self.assertEqual(row['applied_step_norm'], 0)
                self.assertEqual(row['c_loss_change'], 0)
            else:
                self.assertAlmostEqual(row['applied_step_norm'], .1, places=6)
                self.assertAlmostEqual(row['c_loss_change'] - row['linearized_c_loss_change'], .005, places=5)

    def test_muon_metrics_match_production_preview_with_live_state(self):
        self.p.grad = torch.tensor([[.3, -.1]]); self.optimizer.step()
        self.p.grad = torch.tensor([[7., 8.]])  # Must be preserved, not used by new probes.
        before = self.snapshot(); record = self.run_probe()
        assert_tree(self, self.snapshot(), before)
        ga, = torch.autograd.grad(self.forward(*self.a[0]), [self.p])
        gb, = torch.autograd.grad(self.forward(*self.b[0]), [self.p])
        def direction(g):
            d, *_ = preview_muon(self.optimizer, [self.p], [g], self.ns['muon_step_fused'])
            return d[0]
        da, db = direction(ga), direction(gb)
        for trial in record['trials']:
            original = self.p.detach().clone()
            with torch.no_grad(): self.p.copy_(original - trial['radius'] * ga / ga.norm())
            bp, = torch.autograd.grad(self.forward(*self.b[0]), [self.p])
            with torch.no_grad(): self.p.copy_(original)
            dp, dm, dab = direction(bp), direction((ga+bp)/2), direction((ga+gb)/2)
            def cosine(a, b): return float((a*b).sum()/(a.norm()*b.norm()))
            metrics = trial['muon_b_prime']
            self.assertAlmostEqual(metrics['cosine_muon_b_b_prime'], cosine(db, dp), places=5)
            self.assertAlmostEqual(metrics['cosine_muon_a_b_prime'], cosine(da, dp), places=5)
            self.assertAlmostEqual(metrics['cosine_muon_paired_meta'], cosine(dab, dm), places=5)
        assert_tree(self, self.snapshot(), before)

    def test_c_does_not_change_candidate_gradients(self):
        captured = []
        def capture(*args, **kwargs):
            captured.append((args[0], [g.clone() for g in args[1]]))
            return _evaluate_c(*args, **kwargs)
        with patch('transfer._evaluate_c', side_effect=capture):
            first = self.run_probe()
            cutoff = len(captured)
            second = self.run_probe(c=[(self.x, torch.tensor([[-100.], [100.]]))])
        self.assertEqual(len(captured), 2 * cutoff)
        for (source_a, ga), (source_b, gb) in zip(captured[:cutoff], captured[cutoff:]):
            self.assertEqual(source_a, source_b)
            assert_tree(self, ga, gb)
        self.assertNotEqual(first['c_loss'], second['c_loss'])

    def test_exception_restores_state_and_weights(self):
        before = self.snapshot(); calls = 0
        def fail(x, y):
            nonlocal calls
            calls += 1
            if calls == 4: raise RuntimeError('C trial failed')
            return self.forward(x, y)
        with self.assertRaisesRegex(RuntimeError, 'C trial failed'): self.run_probe(forward=fail)
        assert_tree(self, self.snapshot(), before)

    def test_fp16_transformer_transfer_is_finite_and_preserves_state(self):
        ns = self.ns
        ns['ACTIVATION_DTYPE'] = torch.float16
        torch.manual_seed(42)
        model = ns['GPT'](ns['GPTConfig'](vocab_size=64, dropout=.2, xsa_mode='off'))
        model.init_weights(); optimizer = model.setup_optimizer()
        batches = [(torch.randint(64, (1, 16)), torch.randint(64, (1, 16))) for _ in range(3)]
        def forward(x, y):
            with torch.autocast('cpu', dtype=torch.float16, cache_enabled=False):
                return model(x, y, num_iterations=2)
        # Populate real momentum and open initially zero attention/MLP outputs.
        for _ in range(2):
            (forward(*batches[0]) * 128).backward()
            for p in model.parameters():
                if p.grad is not None: p.grad.div_(128)
            optimizer.step(); model.zero_grad(set_to_none=True)
        before = ([p.detach().clone() for p in model.parameters()],
                  copy.deepcopy(optimizer.state_dict()), torch.get_rng_state().clone(),
                  [m.training for m in model.modules()])
        records = probe_transfer_episode(model, optimizer, model.named_parameters(),
                                         [batches[0]], [batches[1]], forward, ns['muon_step_fused'],
                                         128, modes=('qk', 'v', 'mlp'), radii=(.1, .5),
                                         c_batches=[batches[2]], transfer_radius=.01)
        after = ([p.detach().clone() for p in model.parameters()],
                 copy.deepcopy(optimizer.state_dict()), torch.get_rng_state().clone(),
                 [m.training for m in model.modules()])
        assert_tree(self, after, before)
        for record in records:
            self.assertGreater(record['gradient_c_norm'], 0)
            for trial in record['trials']:
                self.assertIsNotNone(trial['muon_b_prime']['cosine_muon_b_b_prime'])
                for row in trial['transfer']:
                    self.assertTrue(math.isfinite(row['c_loss_change']))

    def test_muon_only_needs_no_c_or_training_gradients(self):
        before = self.snapshot()
        record, = probe_transfer_episode(self.model, self.optimizer, self.model.named_parameters(),
                                        self.a, self.b, self.forward, self.ns['muon_step_fused'],
                                        128, modes=('mlp',), radii=(.1,))
        self.assertIsNone(record['c_loss']); self.assertEqual(record['transfer_baselines'], [])
        self.assertIn('muon_b_prime', record['trials'][0])
        assert_tree(self, self.snapshot(), before)

if __name__ == '__main__': unittest.main()
