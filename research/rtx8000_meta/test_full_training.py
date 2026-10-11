"""Execute the complete training CLI on a tiny CPU fixture.

Only sequence length/tokenizer size and compilation are scaled/stubbed; the
production training loop, loader, optimizer, recurrence and reporting execute.
This is not a CUDA FP16 or compiled-kernel validation.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

TRAIN = Path(__file__).with_name('train.py').resolve()
HARNESS = r'''
import ast, sys, types
import torch
from pathlib import Path
source = Path(sys.argv.pop(1))
sys.path.insert(0, str(source.parent))
torch.set_num_threads(2)
torch.compile = lambda fn=None, **kw: fn if fn is not None else lambda f: f
class Tokenizer:
    n_vocab = 64
    _special_tokens = {'<|endoftext|>': 63}
    def decode_single_token_bytes(self, i): return b'x'
sys.modules['tiktoken'] = types.SimpleNamespace(get_encoding=lambda name: Tokenizer())
class Run:
    def __init__(self): self.summary = {}
    def log(self, *a, **kw): pass
    def log_code(self, *a, **kw): pass
    def finish(self): pass
sys.modules['wandb'] = types.SimpleNamespace(init=lambda **kw: Run())
tree = ast.parse(source.read_text())
for n in tree.body:
    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MAX_SEQ_LEN' for t in n.targets):
        n.value = ast.Constant(16)
exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'),
     {'__name__': '__main__', '__file__': str(source)})
'''

class FullTrainingTests(unittest.TestCase):
    def run_training(self, temp, mode, extra=()):
        dataset = Path(temp) / 'data.pt'
        if not dataset.exists():
            generator = torch.Generator().manual_seed(7)
            torch.save(dict(tokens=torch.randint(0, 63, (17 * 128,), generator=generator),
                            doc_starts=torch.tensor([0]), bos_id=50256, seq_shuffle_seed=42), dataset)
        command = [sys.executable, '-c', HARNESS, str(TRAIN),
                   '--precision', 'fp32', '--no-compile', '--n_layer', '2',
                   '--n_head', '4', '--n_embd', '32', '--device-batch-size', '1',
                   '--virtual-ranks', '2', '--total-batch-size', '64',
                   '--schedule-steps', '3', '--max-train-steps', '3', '--eval-tokens', '64',
                   '--iteration-transition-ratio', '0.67', '--dropout', '0',
                   '--input_bin', str(dataset), '--input_val_bin', str(dataset),
                   '--mg-mode', mode, '--run-name', mode, *extra]
        environment = {k: v for k, v in os.environ.items() if k not in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')}
        environment['CUDA_VISIBLE_DEVICES'] = ''
        result = subprocess.run(command, cwd=temp, env=environment, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout, Path(temp) / 'runs' / mode

    def test_all_arms_cross_recurrence_transition_and_save_finite_models(self):
        with tempfile.TemporaryDirectory() as temp:
            for mode in ('off', 'qk', 'v', 'mlp'):
                with self.subTest(mode=mode):
                    exported = Path(temp) / 'exports' / f'{mode}.json'
                    output, run = self.run_training(temp, mode, ['--output_json', str(exported)])
                    self.assertIn('Recurrent iterations -> 2 at step 1', output)
                    self.assertIn('Gradient check:', output)
                    self.assertIn('Reached max_train_steps=3', output)
                    report = json.loads((run / 'result.json').read_text())
                    self.assertEqual(report, json.loads(exported.read_text()))
                    self.assertEqual(report['steps'], 3)
                    self.assertEqual(report['effective_train_tokens'], 3 * 64)
                    self.assertEqual(report['ema_updates'], 1)
                    self.assertEqual(report['checkpoint_average_count'], 0)
                    self.assertTrue(torch.isfinite(torch.tensor(report['raw_final_val_loss'])))
                    weights = torch.load(run / 'model.pt', weights_only=True)
                    self.assertTrue(all(bool(torch.isfinite(w).all()) for w in weights.values()))
                    # First EMA update occurs after the first actual optimizer
                    # step: both zero-initialized output matrices must have moved.
                    for name in ('transformer.h.0.attn.c_proj.weight', 'transformer.h.0.mlp.c_proj.weight'):
                        self.assertGreater(float(weights[name].norm()), 0)

    def test_full_probe_path_exits_without_updating_or_saving_model(self):
        with tempfile.TemporaryDirectory() as temp:
            output, run = self.run_training(temp, 'off', [
                '--iteration-schedule', 'constant', '--probe-every', '1',
                '--probe-pairs', '2', '--probe-modes', 'qk,v,mlp', '--probe-radii', '.1',
                '--probe-alignment-only', '--probe-only'])
            self.assertIn('Profiling complete; no optimizer update applied', output)
            self.assertFalse((run / 'model.pt').exists())
            self.assertFalse((run / 'result.json').exists())
            events = [json.loads(line) for line in (run / 'probes.jsonl').read_text().splitlines()]
            self.assertEqual(len(events), 1)
            self.assertEqual(len(events[0]['measurements']), 6)
            # QKV can legitimately be zero at initialization; MLP output cannot.
            self.assertTrue(all(m['gradient_a_norm'] > 0 for m in events[0]['measurements'] if m['mode'] == 'mlp'))

if __name__ == '__main__': unittest.main()
