"""Exercise the copied optimizer without executing the training CLI."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist


def load_optimizer():
    tree = ast.parse(Path(__file__).with_name('train.py').read_text())
    names = {'adamw_step_fused', 'muon_step_fused', 'DistMuonAdamW'}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == 'polar_express_coeffs' for t in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []  # Exercise eager math; GPU compilation is a separate check.
            nodes.append(node)
    namespace = {'torch': torch, 'dist': dist}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'optimizer_under_test', 'exec'), namespace)
    return namespace['DistMuonAdamW']


class SingleProcessOptimizerTests(unittest.TestCase):
    def test_hybrid_update_without_process_group(self):
        self.assertFalse(dist.is_initialized())
        optimizer_type = load_optimizer()
        adam = torch.nn.Parameter(torch.tensor([0.4, -0.3]))
        reference = torch.nn.Parameter(adam.detach().clone())
        matrix = torch.nn.Parameter(torch.tensor([[0.2, -0.1], [0.3, 0.4]]))
        inactive = torch.nn.Parameter(torch.tensor([[0.1, 0.2], [-0.2, 0.3]]))
        before_matrix, before_inactive = matrix.detach().clone(), inactive.detach().clone()
        optimizer = optimizer_type([
            dict(kind='adamw', params=[adam], lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1),
            dict(kind='muon', params=[matrix, inactive], lr=0.02, momentum=0.9,
                 beta2=0.95, weight_decay=0.1, ns_steps=5),
        ])
        reference_optimizer = torch.optim.AdamW([reference], lr=0.01, betas=(0.9, 0.95),
                                                eps=1e-8, weight_decay=0.1)
        with patch.object(dist, 'get_rank', side_effect=AssertionError('unexpected rank lookup')), \
             patch.object(dist, 'get_world_size', side_effect=AssertionError('unexpected world lookup')), \
             patch.object(dist, 'all_reduce', side_effect=AssertionError('unexpected collective')), \
             patch.object(dist, 'reduce_scatter_tensor', side_effect=AssertionError('unexpected collective')), \
             patch.object(dist, 'all_gather_into_tensor', side_effect=AssertionError('unexpected collective')):
            for _ in range(2):
                adam.grad = torch.tensor([0.2, -0.4])
                reference.grad = adam.grad.clone()
                matrix.grad = torch.tensor([[0.1, 0.3], [-0.2, 0.4]])
                optimizer.step()
                reference_optimizer.step()
                torch.testing.assert_close(adam, reference)
                self.assertTrue(torch.isfinite(matrix).all())
                torch.testing.assert_close(inactive, before_inactive, rtol=0, atol=0)
        self.assertFalse(torch.equal(matrix, before_matrix))
        self.assertEqual(optimizer.state[adam]['step'], 2)


if __name__ == '__main__':
    unittest.main()
