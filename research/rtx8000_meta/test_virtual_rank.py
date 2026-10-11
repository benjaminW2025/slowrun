import unittest

import torch
from torch import nn
import torch.nn.functional as F

from virtual_rank import collect_virtual_gradients, select_trial_parameters, _temporary_step, validate_gradients


class FixedScaler:
    def __init__(self, scale=1.0):
        self.factor = scale

    def scale(self, loss):
        return loss * self.factor


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trial = nn.Linear(2, 2, bias=False)
        self.other = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.trial.weight.copy_(torch.tensor([[0.2, -0.1], [0.3, 0.4]]))
            self.other.weight.copy_(torch.tensor([[0.5, -0.2]]))

    def forward(self, x, y):
        return (self.other(torch.tanh(self.trial(x))) - y).square().mean()


def reference_grads(model, batches, use_trial, radius, microsteps_per_half):
    params = list(model.parameters())
    expected = [torch.zeros_like(p) for p in params]
    original = [p.detach().clone() for p in params]
    total = len(batches)
    for i in range(0, total, 2 * microsteps_per_half):
        a = [torch.zeros_like(p) for p in params]
        for j in range(i, i + microsteps_per_half):
            grads = torch.autograd.grad(model(*batches[j]), params)
            for dst, src in zip(a, grads):
                dst.add_(src / total)
        if use_trial:
            direction = a[0] / a[0].norm()
            with torch.no_grad():
                params[0].sub_(radius * direction)
        b = [torch.zeros_like(p) for p in params]
        for j in range(i + microsteps_per_half, i + 2 * microsteps_per_half):
            grads = torch.autograd.grad(model(*batches[j]), params)
            for dst, src in zip(b, grads):
                dst.add_(src / total)
        with torch.no_grad():
            if use_trial:
                params[0].copy_(original[0])
        for j in range(len(params)):
            expected[j].add_(a[j] + b[j])
    return expected


class VirtualRankTests(unittest.TestCase):
    def setUp(self):
        self.batches = [
            (torch.tensor([[1.0, 0.5]]), torch.tensor([[0.3]])),
            (torch.tensor([[0.2, -0.8]]), torch.tensor([[-0.1]])),
            (torch.tensor([[-0.7, 0.4]]), torch.tensor([[0.6]])),
            (torch.tensor([[0.9, -0.3]]), torch.tensor([[0.2]])),
        ]

    def run_case(self, use_trial, scale, microsteps_per_half=1):
        model = TinyModel()
        params = list(model.parameters())
        original = [p.detach().clone() for p in params]
        batches = self.batches * microsteps_per_half
        expected = reference_grads(model, batches, use_trial, radius=0.5,
                                   microsteps_per_half=microsteps_per_half)
        batch_iter = iter(batches[1:] + [batches[0]])
        trial_params = [params[0]] if use_trial else []
        collect_virtual_gradients(
            model, params, trial_params, *self.batches[0],
            lambda: (*next(batch_iter), 1), model, FixedScaler(scale),
            virtual_ranks=2, microsteps_per_half=microsteps_per_half, step_norm=0.5,
        )
        for p, before, want in zip(params, original, expected):
            torch.testing.assert_close(p, before, rtol=0, atol=0)
            torch.testing.assert_close(p.grad / scale, want, rtol=1e-5, atol=1e-6)

    def test_meta_gradient_matches_independent_rank_reference(self):
        self.run_case(use_trial=True, scale=1.0)

    def test_baseline_matches_ordinary_batch_gradient(self):
        self.run_case(use_trial=False, scale=1.0)

    def test_scaled_gradient_keeps_trial_radius(self):
        self.run_case(use_trial=True, scale=128.0)

    def test_multiple_microbatches_per_half(self):
        self.run_case(use_trial=True, scale=128.0, microsteps_per_half=2)

    def test_zero_initialized_output_blocks_trial_gradient(self):
        model = TinyModel()
        with torch.no_grad():
            model.other.weight.zero_()
        params = list(model.parameters())
        original = [p.detach().clone() for p in params]
        expected = reference_grads(model, self.batches, False, 0.5, 1)
        batch_iter = iter(self.batches[1:] + [self.batches[0]])
        collect_virtual_gradients(
            model, params, [model.trial.weight], *self.batches[0],
            lambda: (*next(batch_iter), 1), model, FixedScaler(128.0),
            virtual_ranks=2, microsteps_per_half=1, step_norm=0.5,
        )
        for p, before, want in zip(params, original, expected):
            torch.testing.assert_close(p, before, rtol=0, atol=0)
            torch.testing.assert_close(p.grad / 128.0, want)
        self.assertEqual(params[0].grad.norm().item(), 0.0)
        self.assertGreater(params[1].grad.norm().item(), 0.0)
        with torch.no_grad():
            for p in params:
                p.add_(p.grad / 128.0, alpha=-0.1)
        model.zero_grad(set_to_none=True)
        model(*self.batches[0]).backward()
        self.assertGreater(params[0].grad.norm().item(), 0.0)
        before = params[0].detach().clone()
        saved = _temporary_step([params[0]], 0.5)
        self.assertIsNotNone(saved)
        torch.testing.assert_close((params[0] - before).norm(), torch.tensor(0.5))

    def test_nonfinite_trial_gradient_still_raises(self):
        p = nn.Parameter(torch.ones(2, 2))
        for value in (float('nan'), float('inf')):
            p.grad = torch.full_like(p, value)
            with self.assertRaises(FloatingPointError):
                _temporary_step([p], 0.5)
            torch.testing.assert_close(p, torch.ones_like(p), rtol=0, atol=0)

    def test_fp16_large_vocabulary_retains_non_target_gradients(self):
        # A token contribution to a 2048-token mean with 256 accumulated
        # microbatches. Pre-dividing the loss erased every non-target gradient.
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            model = nn.Module()
            model.register_parameter('logits', nn.Parameter(torch.zeros(1, 50257)))
            target = torch.tensor([0])
            x = torch.zeros(1, dtype=torch.long)
            reference_loss = F.cross_entropy(model.logits, target) / 2048
            expected, = torch.autograd.grad(reference_loss, model.logits)
            def fp16_forward(batch_x, batch_y):
                return F.cross_entropy(model.logits.half().float(), batch_y) / 2048
            collect_virtual_gradients(
                model, [model.logits], [], x, target, lambda: (x, target, 1),
                fp16_forward, FixedScaler(128.0), virtual_ranks=8,
                microsteps_per_half=16, step_norm=0.5,
            )
            actual = model.logits.grad / 128.0
            self.assertTrue(bool((actual[0, 1:] > 0).all()))
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0)
        finally:
            torch.set_num_threads(old_threads)

    def test_gradient_guard_distinguishes_zero_missing_and_nonfinite(self):
        model = TinyModel()
        named = list(model.named_parameters())
        ids = {id(p) for name, p in named}
        with self.assertRaisesRegex(RuntimeError, 'Missing training gradients'):
            validate_gradients(named, ids)
        with self.assertRaisesRegex(RuntimeError, 'Missing temporary A gradients'):
            _temporary_step([model.trial.weight], .5)
        for name, p in named: p.grad = torch.zeros_like(p)
        self.assertEqual(validate_gradients(named, ids), 2)
        for value in (float('nan'), float('inf')):
            model.trial.weight.grad.fill_(value)
            with self.assertRaisesRegex(FloatingPointError, 'trial.weight'):
                validate_gradients(named, ids)
        # An intentionally unused parameter outside the optimizer is permitted.
        model.register_parameter('disabled', nn.Parameter(torch.ones(1)))
        model.trial.weight.grad.zero_()
        self.assertEqual(validate_gradients(model.named_parameters(), ids), 2)

    def test_b_failure_restores_temporary_weights(self):
        model = TinyModel()
        original = model.trial.weight.detach().clone()
        calls = 0
        def fail_on_b(x, y):
            nonlocal calls
            calls += 1
            if calls == 2: raise RuntimeError('failed B forward')
            return model(x, y)
        with self.assertRaisesRegex(RuntimeError, 'failed B forward'):
            collect_virtual_gradients(model, list(model.parameters()), [model.trial.weight],
                                      *self.batches[0], lambda: (*self.batches[1], 1),
                                      fail_on_b, FixedScaler(128), 1, 1, .5)
        torch.testing.assert_close(model.trial.weight, original, rtol=0, atol=0)

    def test_matrix_selection(self):
        weights = {
            name: nn.Parameter(torch.ones(2, 2))
            for name in (
                "h.0.attn.c_q.weight", "h.0.attn.c_k.weight",
                "h.0.attn.c_v.weight", "h.0.attn.c_proj.weight",
                "h.0.mlp.c_fc.weight", "h.0.mlp.c_proj.weight",
            )
        }
        named = list(weights.items())
        muon_ids = {id(p) for p in weights.values()}
        for mode, expected in {
            "off": [],
            "qk": ["h.0.attn.c_q.weight", "h.0.attn.c_k.weight"],
            "v": ["h.0.attn.c_v.weight"],
            "qkv": ["h.0.attn.c_q.weight", "h.0.attn.c_k.weight", "h.0.attn.c_v.weight"],
            "mlp": ["h.0.mlp.c_fc.weight", "h.0.mlp.c_proj.weight"],
        }.items():
            selected = select_trial_parameters(named, muon_ids, mode)
            self.assertEqual([name for name, p in named if any(p is q for q in selected)], expected)


if __name__ == "__main__":
    unittest.main()
