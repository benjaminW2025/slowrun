"""Compare the experiment to immutable upstream Git sources, without running the CLI."""
import ast
import copy
from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from virtual_rank import collect_virtual_gradients
from test_virtual_rank import FixedScaler, TinyModel

HERE = Path(__file__).resolve().parent
BASE = 'a2c443e18f99d0f6e35c6013c752e636fe402213'
META = '45bf3fd'

def upstream(ref):
    return ast.parse(subprocess.check_output(['git', 'show', f'{ref}:tiny/train.py'], cwd=HERE, text=True))

def definitions(tree):
    return {n.name: n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}

class Work:
    def get_future(self): return self
    def wait(self): pass

class IdentityCollectives:
    # World-size-one identity operations exercise the original distributed path.
    ReduceOp = SimpleNamespace(AVG='avg')
    def is_initialized(self): return True
    def get_rank(self): return 0
    def get_world_size(self): return 1
    def all_reduce(self, tensor, **kwargs): return Work()
    def reduce_scatter_tensor(self, out, source, **kwargs):
        out.copy_(source); return Work()
    def all_gather_into_tensor(self, out, source, **kwargs):
        out.copy_(source); return Work()

class LocalOnly(IdentityCollectives):
    def is_initialized(self): return False
    def all_reduce(self, *a, **k): raise AssertionError('unexpected collective')
    reduce_scatter_tensor = all_reduce
    all_gather_into_tensor = all_reduce

def load(tree, distributed, common_precision=False):
    tree = copy.deepcopy(tree)
    nodes = []
    for n in tree.body:
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)):
            if isinstance(n, ast.FunctionDef):
                n.decorator_list = [d for d in n.decorator_list if not (
                    isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == 'compile')]
            nodes.append(n)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'polar_express_coeffs' for t in n.targets):
            nodes.append(n)
    if common_precision:
        # Isolate structural/math equivalence from the documented BF16->FP32 adaptation.
        class FloatReference(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'bfloat16':
                    node.func.attr = 'float'
                return node
        nodes = [FloatReference().visit(n) for n in nodes]
    ns = dict(torch=torch, nn=nn, F=F, dist=distributed, dataclass=dataclass, math=math,
              MAX_SEQ_LEN=16, DEPTH=2, N_HEAD=4, N_EMBD=32, WINDOW_PATTERN='SSSL', NUM_ITERATIONS=2,
              ACTIVATION_DTYPE=torch.float32, TRAIN_BACKPROP_ITERATIONS=1,
              MATRIX_LR=.032, UNEMBEDDING_LR=.0008, EMBEDDING_LR=.12, SCALAR_LR=.2,
              WEIGHT_DECAY=.8, ADAM_BETAS=(.8, .95), _fa3=None)
    exec(compile(ast.fix_missing_locations(ast.Module(nodes, type_ignores=[])), 'upstream_audit', 'exec'), ns)
    ns['print0'] = lambda *a, **k: None
    ns['get_dist_info'] = lambda: (False, 0, 0, 1)
    ns['flash_attn'] = SimpleNamespace(flash_attn_func=ns['flash_attn_func'])
    return ns

class UpstreamAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.base = upstream(BASE)
        cls.meta = upstream(META)
        cls.experiment = ast.parse((HERE / 'train.py').read_text())

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.before_threads)

    def test_only_documented_definition_changes(self):
        base, experiment = definitions(self.base), definitions(self.experiment)
        changed = {name for name in base if ast.dump(base[name]) != ast.dump(experiment[name])}
        self.assertEqual(changed, {'GPT', 'muon_step_fused', 'DistMuonAdamW'})
        # All GPT methods except the two documented dtype adaptations are exact copies.
        bm, em = definitions(base['GPT']), definitions(experiment['GPT'])
        for name in bm:
            if name not in {'init_weights', '_precompute_rotary'}:
                self.assertEqual(ast.dump(bm[name]), ast.dump(em[name]), name)
        expected_muon = copy.deepcopy(base['muon_step_fused'])
        for n in ast.walk(expected_muon):
            if isinstance(n, ast.Attribute) and n.attr == 'bfloat16': n.attr = 'float'
        # The new opt-in decay branch must preserve the original default math.
        actual_muon = copy.deepcopy(experiment['muon_step_fused'])
        self.assertEqual(actual_muon.args.args.pop().arg, 'cautious_decay')
        self.assertTrue(actual_muon.args.defaults.pop().value)
        for n in ast.walk(actual_muon):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.IfExp):
                if any(isinstance(t, ast.Name) and t.id == 'mask' for t in n.targets):
                    n.value = n.value.body
        self.assertEqual(ast.dump(expected_muon), ast.dump(actual_muon))
        # Original PR uses the same optimizer setup as the older baseline.
        pm = definitions(definitions(self.meta)['GPT'])
        self.assertEqual(ast.dump(bm['setup_optimizer']), ast.dump(pm['setup_optimizer']))

    def test_baseline_cli_defaults_and_polar_coefficients(self):
        def defaults(tree):
            result = {}
            for node in tree.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    call = node.value
                    if isinstance(call.func, ast.Attribute) and call.func.attr == 'add_argument':
                        result[ast.literal_eval(call.args[0])] = {
                            k.arg: ast.dump(k.value) for k in call.keywords if k.arg in {'default', 'action', 'choices'}
                        }
            return result
        before, after = defaults(self.base), defaults(self.experiment)
        for key in before:
            if key != '--device-batch-size': self.assertEqual(before[key], after[key], key)
        def coefficients(tree):
            return next(ast.dump(n) for n in tree.body if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'polar_express_coeffs' for t in n.targets))
        self.assertEqual(coefficients(self.base), coefficients(self.experiment))
        self.assertEqual(coefficients(self.meta), coefficients(self.experiment))

    def test_original_distributed_optimizer_matches_local_path(self):
        b = load(self.base, IdentityCollectives(), common_precision=True)
        e = load(self.experiment, LocalOnly())
        for rows, cols in [(3, 5), (5, 3), (4, 4)]:
            torch.manual_seed(123)
            original = [nn.Parameter(torch.randn(rows, cols)) for _ in range(3)]
            original += [nn.Parameter(torch.randn(1024, 2)), nn.Parameter(torch.randn(3))]
            adapted = [nn.Parameter(p.detach().clone()) for p in original]
            def groups(p):
                return [dict(kind='muon', params=p[:3], lr=.032, momentum=.9, beta2=.95,
                             weight_decay=.8, ns_steps=5),
                        dict(kind='adamw', params=p[3:], lr=.01, betas=(.8, .95), eps=1e-10, weight_decay=.8)]
            bo, eo = b['DistMuonAdamW'](groups(original)), e['DistMuonAdamW'](groups(adapted))
            for step in range(4):
                for i, (p, q) in enumerate(zip(original, adapted)):
                    p.grad = None if i == 2 and step % 2 == 0 else torch.randn_like(p)
                    q.grad = None if p.grad is None else p.grad.clone()
                bo.step(); eo.step()
                for p, q in zip(original, adapted): torch.testing.assert_close(p, q, rtol=0, atol=0)
                for p, q in [(original[0], adapted[0]), *zip(original[3:], adapted[3:])]:
                    for key, value in bo.state[p].items():
                        other = eo.state[q][key]
                        if isinstance(value, torch.Tensor): torch.testing.assert_close(value, other, rtol=0, atol=0)
                        else: self.assertEqual(value, other)

    def test_transformer_forward_backward_matches_original(self):
        b = load(self.base, IdentityCollectives(), common_precision=True)
        e = load(self.experiment, LocalOnly())
        torch.manual_seed(42)
        bm = b['GPT'](b['GPTConfig'](vocab_size=64, dropout=0, xsa_mode='off')); bm.init_weights()
        em = e['GPT'](e['GPTConfig'](vocab_size=64, dropout=0, xsa_mode='off')); em.init_weights()
        em.load_state_dict(bm.state_dict())
        bo, eo = bm.setup_optimizer(), em.setup_optimizer()
        x = torch.arange(32).view(2, 16) % 64; y = (x + 1) % 64
        for iterations in (1, 2, 2):
            bl, el = bm(x, y, num_iterations=iterations), em(x, y, num_iterations=iterations)
            torch.testing.assert_close(bl, el, rtol=0, atol=0)
            bl.backward(); el.backward()
            for p, q in zip(bm.parameters(), em.parameters()):
                self.assertEqual(p.grad is None, q.grad is None)
                if p.grad is not None: torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
            bo.step(); eo.step()
            for p, q in zip(bm.parameters(), em.parameters()): torch.testing.assert_close(p, q, rtol=0, atol=0)
            bm.zero_grad(set_to_none=True); em.zero_grad(set_to_none=True)

    def test_virtual_episodes_match_actual_pr_temporary_step(self):
        node = definitions(self.meta)['mg_temporary_step']
        torch.manual_seed(42)
        batches = [(torch.randn(1, 2), torch.randn(1, 1)) for _ in range(32)]
        for selected in ([], [0], [1], [0, 1]):
            for microsteps in (1, 2):
                original, adapted = TinyModel(), TinyModel()
                op, ap = list(original.parameters()), list(adapted.parameters())
                ns = dict(torch=torch, mg_params=[op[i] for i in selected], mg_owned=list(range(len(selected))),
                          args=SimpleNamespace(mg_step_norm=.5, mg_step_schedule='const'))
                exec(compile(ast.Module([copy.deepcopy(node)], type_ignores=[]), 'actual_pr99', 'exec'), ns)
                expected = [torch.zeros_like(p) for p in op]
                used = batches[:2 * 8 * microsteps]
                for start in range(0, len(used), 2 * microsteps):
                    original.zero_grad(set_to_none=True)
                    def batch_loss(half):
                        chunk = used[start + half * microsteps:start + (half + 1) * microsteps]
                        return original(torch.cat([x for x, y in chunk]), torch.cat([y for x, y in chunk]))
                    (batch_loss(0) * .5).backward()
                    if selected: kept, alpha = ns['mg_temporary_step'](0)
                    (batch_loss(1) * .5).backward()
                    if selected:
                        with torch.no_grad(): torch._foreach_add_(ns['mg_params'], kept, alpha=-alpha)
                    for dst, p in zip(expected, op): dst.add_(p.grad / 8)
                iterator = iter(used[1:] + [used[0]])
                collect_virtual_gradients(adapted, ap, [ap[i] for i in selected], *used[0],
                                          lambda: (*next(iterator), 1), adapted, FixedScaler(128), 8, microsteps, .5)
                for p, want in zip(ap, expected): torch.testing.assert_close(p.grad / 128, want, rtol=2e-5, atol=2e-6)

if __name__ == '__main__': unittest.main()
