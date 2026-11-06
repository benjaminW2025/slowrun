"""Check production mixed-precision configuration without requiring a CUDA device."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


class PrecisionTests(unittest.TestCase):
    def configure(self, precision):
        tree = ast.parse(Path(__file__).with_name('train.py').read_text())
        names = {'ACTIVATION_DTYPE', 'autocast_ctx', 'scaler'}
        nodes = [node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
        self.assertEqual(len(nodes), 3)
        calls = {}
        def autocast(**kwargs):
            calls['autocast'] = kwargs
            return nullcontext()
        def scaler(*args, **kwargs):
            calls['scaler'] = kwargs
            return None
        fake_torch = SimpleNamespace(float16=torch.float16, bfloat16=torch.bfloat16,
                                    float32=torch.float32,
                                    amp=SimpleNamespace(autocast=autocast, GradScaler=scaler))
        namespace = dict(torch=fake_torch, args=SimpleNamespace(precision=precision, loss_scale=128),
                         device_type='cuda', nullcontext=nullcontext)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'precision_config', 'exec'), namespace)
        return calls

    def test_bf16_autocast_disables_scaling_and_detached_weight_cache(self):
        calls = self.configure('bf16')
        self.assertEqual(calls['autocast']['dtype'], torch.bfloat16)
        self.assertFalse(calls['autocast']['cache_enabled'])
        self.assertFalse(calls['scaler']['enabled'])

    def test_fp16_keeps_scaling_and_fp32_disables_autocast(self):
        calls = self.configure('fp16')
        self.assertEqual(calls['autocast']['dtype'], torch.float16)
        self.assertTrue(calls['scaler']['enabled'])
        calls = self.configure('fp32')
        self.assertNotIn('autocast', calls)
        self.assertFalse(calls['scaler']['enabled'])
