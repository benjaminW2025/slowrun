"""Exercise the actual final-evaluation blocks without launching training."""
import ast
import os
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

TREE = ast.parse(Path(__file__).with_name('train.py').read_text())

def execute(node, ns):
    exec(compile(ast.Module([node], type_ignores=[]), 'training_reporting_under_test', 'exec'), ns)

class ReportingTests(unittest.TestCase):
    def model_namespace(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad(): model.weight.fill_(2)
        return dict(torch=torch, os=os, model=model, orig_model=model,
                    print0=lambda *a, **k: None, wandb_run=SimpleNamespace(log=lambda *a, **k: None),
                    evaluate_bpb=lambda m, *a, **k: (float(m.weight.item()), float(m.weight.item())),
                    build_val_loader=lambda: [], eval_steps=1, token_bytes=None, NUM_ITERATIONS=2,
                    autocast_ctx=nullcontext(), min_val_bpb=float('inf'), min_val_loss=float('inf'),
                    val_bpb=0, val_loss=0, model_variant='raw', ddp=False, master_process=True)

    def test_ema_bias_correction_counts_actual_updates(self):
        loop = next(n for n in TREE.body if isinstance(n, ast.While))
        update = next(n for n in loop.body if isinstance(n, ast.If) and 'step % args.update_ema_every' in ast.unparse(n.test))
        final = next(n for n in TREE.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'ema_params is not None')
        for steps in (1, 5, 10, 11, 20, 21, 500):
            ns = self.model_namespace()
            ns.update(args=SimpleNamespace(update_ema_every=10), param_ema_beta=.9,
                      ema_params=[torch.zeros_like(ns['model'].weight)], ema_updates=0)
            for step in range(steps):
                ns['step'] = step; execute(update, ns)
            self.assertEqual(ns['ema_updates'], (steps + 9) // 10)
            execute(final, ns)
            self.assertAlmostEqual(ns['model'].weight.item(), 2, places=5)
            self.assertEqual(ns['model_variant'], 'ema')
            self.assertAlmostEqual(ns['val_loss'], 2, places=5)

    def test_checkpoint_average_metric_matches_saved_weights(self):
        final = next(n for n in TREE.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'len(late_ckpt_paths) >= 2')
        with tempfile.TemporaryDirectory() as temp:
            paths = [str(Path(temp) / f'{i}.pt') for i in range(2)]
            for p, value in zip(paths, (1., 4.)):
                torch.save({'weight': torch.tensor([[value]])}, p)
            ns = self.model_namespace(); ns['late_ckpt_paths'] = paths
            execute(final, ns)
            self.assertEqual(ns['model_variant'], 'checkpoint_average')
            self.assertAlmostEqual(ns['model'].weight.item(), 3)
            self.assertAlmostEqual(ns['val_loss'], 3)
            self.assertAlmostEqual(ns['val_bpb'], 3)

    def test_short_pilot_excludes_checkpoints_before_swa_window(self):
        gate = next(n for n in ast.walk(TREE) if isinstance(n, ast.If) and ast.unparse(n.test) == '_swa_start_step >= 0 and step >= _swa_start_step' and any(isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == 'save' for x in ast.walk(n)))
        ns = self.model_namespace()
        ns.update(_swa_start_step=500 - 4 * (500 / 15), late_ckpt_paths=[],
                  checkpoints_dir='/unused', current_epoch=1, args=SimpleNamespace(swa_last_epochs=4))
        with patch.object(torch, 'save') as save:
            ns['step'] = 190; execute(gate, ns)
            save.assert_not_called(); self.assertEqual(ns['late_ckpt_paths'], [])
            ns['step'] = 380; execute(gate, ns)
            save.assert_called_once(); self.assertEqual(len(ns['late_ckpt_paths']), 1)

if __name__ == '__main__': unittest.main()
