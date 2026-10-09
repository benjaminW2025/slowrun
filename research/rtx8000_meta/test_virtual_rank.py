import unittest

import torch
from torch import nn

from virtual_rank import collect_virtual_gradients, select_trial_parameters


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
            "mlp": ["h.0.mlp.c_fc.weight", "h.0.mlp.c_proj.weight"],
        }.items():
            selected = select_trial_parameters(named, muon_ids, mode)
            self.assertEqual([name for name, p in named if any(p is q for q in selected)], expected)


if __name__ == "__main__":
    unittest.main()
