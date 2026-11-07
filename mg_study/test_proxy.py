"""
Correctness tests for mg_study/proxy.py. Uses synthetic data in a temp dir, so it needs no dataset.

    python mg_study/test_proxy.py               # cuda if available, else cpu
    python mg_study/test_proxy.py --device cpu

Checks (fp32, no autocast, against hand-written references):
  1. --mg off: the grouped gradient equals the ordinary full-batch mean gradient
  2. every MG mode leaves the parameters exactly as they were (the temporary step is undone)
  3. descent / ascent: the gradient equals a from-scratch reference implementation of the record's step
  4. random: deterministic given the seed, uses both signs, and each group matches the reference for its sign
  5. the logged cosine equals cos(g_A(theta), g_B(theta')) computed by hand
  6. end to end: main() runs for every arm (eager and torch.compile), writes result.json, and the loss drops
"""
import argparse
import copy
import json
import os
import random
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proxy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="auto")
ap.add_argument("--skip-compile", action="store_true")
ap.add_argument("--only-compile", action="store_true", help="run only test 7 (torch.compile)")
targs = ap.parse_args()
DEV = torch.device("cuda" if (targs.device == "auto" and torch.cuda.is_available()) else
                   ("cpu" if targs.device == "auto" else targs.device))
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print(f"testing on {DEV}")
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def make_cfg(**kw):
    cfg = proxy.get_args([])
    cfg.mg_groups, cfg.mg_step_norm, cfg.mg_plastic = 2, 0.5, "mlp"
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def small_model(seed=0):
    torch.manual_seed(seed)
    return proxy.GPT(50257, 64, 2, 4, 32, 0.0).to(DEV).double()   # float64: references must match to ~1e-10


def batch(seed=1, B=8, T=32):
    g = torch.Generator().manual_seed(seed)
    s = torch.randint(0, 50257, (B, T + 1), generator=g).to(DEV)
    return s[:, :-1], s[:, 1:]


def grads(model):
    return [p.grad.detach().clone() for p in model.parameters()]


def maxdiff(a, b):
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


def scale(a):
    return max(float(x.abs().max()) for x in a)


def reference_group_grad(model, xb, yb, sign, eta, plastic_names):
    """One record-style rank, written from scratch: g_A at theta, step theta - sign*eta*g_A/|g_A| on the
    plastic matrices, g_B at the moved point (on a deep copy), return (g_A + g_B(theta'))/2 and cos."""
    h = xb.size(0) // 2
    m = copy.deepcopy(model)
    m.zero_grad(set_to_none=True)
    m(xb[:h], yb[:h]).backward()
    gA = {n: p.grad.detach().clone() for n, p in m.named_parameters()}
    norm = torch.sqrt(sum(gA[n].pow(2).sum() for n in plastic_names))
    m2 = copy.deepcopy(model)
    with torch.no_grad():
        for n, p in m2.named_parameters():
            if n in plastic_names:
                p.sub_(sign * eta * gA[n] / norm)
    m2.zero_grad(set_to_none=True)
    m2(xb[h:], yb[h:]).backward()
    gB = {n: p.grad.detach().clone() for n, p in m2.named_parameters()}
    names = [n for n, _ in model.named_parameters()]
    dot = sum((gA[n] * gB[n]).sum() for n in plastic_names)
    cos = float(dot / (torch.sqrt(sum(gA[n].pow(2).sum() for n in plastic_names)) *
                       torch.sqrt(sum(gB[n].pow(2).sum() for n in plastic_names))))
    return [(gA[n] + gB[n]) / 2 for n in names], cos


def run_compute(model, cfg, xb, yb, rng=None, log_cos=False):
    params = list(model.parameters())
    plastic = proxy.plastic_params(model, cfg.mg_plastic)
    loss, cos = proxy.compute_grad(model, params, plastic, xb, yb, cfg, rng or random.Random(0), log_cos)
    return grads(model), loss, cos


def compile_test():
    """7. torch.compile must use the temporarily moved weights. Compiled and eager kernels differ by
    float rounding (reduction order, fused kernels), so the reference is that noise floor measured on
    --mg off, where MG plays no part. Checks: (a) compiled vs eager differ no more for descent than for
    off, i.e. MG adds no compile-specific error; (b) the MG effect (descent - off) is the same compiled
    and eager; (c) compiled descent is far from compiled off, i.e. the step is not ignored."""
    print("7. torch.compile sees the temporary step (params are graph inputs, not baked-in constants)")
    torch.manual_seed(0)
    m32 = proxy.GPT(50257, 64, 2, 4, 32, 0.0).to(DEV)
    xb32, yb32 = batch(seed=7)
    params, plastic = list(m32.parameters()), proxy.plastic_params(m32, "mlp")
    mc = torch.compile(m32, dynamic=False)

    def g_of(model, mode):
        proxy.compute_grad(model, params, plastic, xb32, yb32, make_cfg(mg=mode), random.Random(0))
        return grads(m32)

    e_off, e_desc = g_of(m32, "off"), g_of(m32, "descent")
    s = scale(e_desc)
    for rep in range(2):   # twice: the second call runs the cached graph
        c_off, c_desc = g_of(mc, "off"), g_of(mc, "descent")
        floor = maxdiff(c_off, e_off) / s
        err = maxdiff(c_desc, e_desc) / s
        check(f"(a) call {rep + 1}: compiled-vs-eager error for descent within 10x the off noise floor",
              err <= max(10 * floor, 1e-5), f"(descent {err:.2e}, off floor {floor:.2e})")
        eff_e = [d - o for d, o in zip(e_desc, e_off)]
        eff_c = [d - o for d, o in zip(c_desc, c_off)]
        rel_eff = maxdiff(eff_c, eff_e) / scale(eff_e)
        check(f"(b) call {rep + 1}: MG effect (descent - off) same compiled and eager", rel_eff < 0.02,
              f"(rel diff {rel_eff:.2e} of the effect size {scale(eff_e) / s:.2e})")
    step = maxdiff(c_desc, c_off) / s
    check("(c) compiled descent != compiled off (the step is not ignored)", step > 100 * max(floor, 1e-7),
          f"(descent-off {step:.2e} vs noise floor {floor:.2e})")


if targs.only_compile:
    compile_test()
    print("\n" + (f"{len(FAILS)} FAILED: {FAILS}" if FAILS else "ALL TESTS PASSED"))
    sys.exit(1 if FAILS else 0)


xb, yb = batch()
model = small_model()
plastic_names = [n for n, p in model.named_parameters() if n.startswith("blocks.") and p.ndim == 2 and ".mlp_" in n]
check("plastic set = the 4 MLP matrices of the 2-layer model", len(plastic_names) == 4, str(plastic_names))

print("1. --mg off equals the full-batch mean gradient")
cfg = make_cfg(mg="off")
g_off, _, _ = run_compute(model, cfg, xb, yb)
model.zero_grad(set_to_none=True)
model(xb, yb).backward()
g_full = grads(model)
check("off == full batch", maxdiff(g_off, g_full) < 1e-10 * max(1.0, scale(g_full)), f"(max diff {maxdiff(g_off, g_full):.2e})")

print("2. parameters are restored exactly")
for mode in ["descent", "ascent", "random"]:
    before = [p.detach().clone() for p in model.parameters()]
    run_compute(model, make_cfg(mg=mode), xb, yb, random.Random(3), log_cos=True)
    check(f"{mode}: params bitwise unchanged", all(torch.equal(a, p) for a, p in zip(before, model.parameters())))

print("3. descent / ascent match the from-scratch reference")
G, gs = 2, xb.size(0) // 2
for mode, sign in [("descent", 1.0), ("ascent", -1.0)]:
    g_mg, _, cos = run_compute(model, make_cfg(mg=mode), xb, yb, log_cos=True)
    refs = [reference_group_grad(model, xb[i * gs:(i + 1) * gs], yb[i * gs:(i + 1) * gs], sign, 0.5, plastic_names) for i in range(G)]
    g_ref = [sum(r[0][j] for r in refs) / G for j in range(len(g_mg))]
    check(f"{mode} == reference", maxdiff(g_mg, g_ref) < 1e-9 * max(1.0, scale(g_ref)), f"(max diff {maxdiff(g_mg, g_ref):.2e})")
    check(f"{mode} differs from plain gradient", maxdiff(g_mg, g_full) > 1e-6, f"(diff vs off {maxdiff(g_mg, g_full):.2e})")
    cref = [r[1] for r in refs]
    check(f"{mode} logged cos == reference", len(cos) == G and all(abs(c - r) < 1e-9 and s == sign for (s, c), r in zip(cos, cref)),
          f"(logged {[round(c, 6) for _, c in cos]} ref {[round(c, 6) for c in cref]})")

print("4. random sign")
signs = [proxy.step_sign("random", r) for r in [random.Random(5)] for _ in range(400)]
check("random uses both signs ~50/50", 150 < sum(s > 0 for s in signs) < 250, f"({sum(s > 0 for s in signs)}/400 descent)")
s1 = [proxy.step_sign("random", random.Random(9)) for _ in range(1)]
r_a, r_b = random.Random(11), random.Random(11)
check("random deterministic given seed", [proxy.step_sign("random", r_a) for _ in range(50)] == [proxy.step_sign("random", r_b) for _ in range(50)])
for trial in range(4):
    rng_seed = 100 + trial
    g_mg, _, cos = run_compute(model, make_cfg(mg="random"), xb, yb, random.Random(rng_seed), log_cos=True)
    r = random.Random(rng_seed)
    gsigns = [proxy.step_sign("random", r) for _ in range(G)]
    refs = [reference_group_grad(model, xb[i * gs:(i + 1) * gs], yb[i * gs:(i + 1) * gs], gsigns[i], 0.5, plastic_names) for i in range(G)]
    g_ref = [sum(rr[0][j] for rr in refs) / G for j in range(len(g_mg))]
    check(f"random trial {trial} (signs {gsigns}) == reference", maxdiff(g_mg, g_ref) < 1e-9 * max(1.0, scale(g_ref)) and
          [s for s, _ in cos] == gsigns, f"(max diff {maxdiff(g_mg, g_ref):.2e})")

print("5. matrix plastic set + more groups")
cfg = make_cfg(mg="descent", mg_plastic="matrix", mg_groups=4)
g_mg, _, _ = run_compute(model, cfg, xb, yb)
pn = [n for n, p in model.named_parameters() if n.startswith("blocks.") and p.ndim == 2]
gs4 = xb.size(0) // 4
refs = [reference_group_grad(model, xb[i * gs4:(i + 1) * gs4], yb[i * gs4:(i + 1) * gs4], 1.0, 0.5, pn) for i in range(4)]
g_ref = [sum(r[0][j] for r in refs) / 4 for j in range(len(g_mg))]
check("matrix / 4 groups == reference", maxdiff(g_mg, g_ref) < 1e-9 * max(1.0, scale(g_ref)), f"(max diff {maxdiff(g_mg, g_ref):.2e})")

print("6. end-to-end runs on synthetic data")
tmp = tempfile.mkdtemp()
data_dir = os.path.join(tmp, "data")
os.makedirs(data_dir)
g = torch.Generator().manual_seed(0)
# learnable stream: one random 512-state Markov chain (4 successors per state) shared by train and val
states = torch.randint(0, 50257, (512,), generator=g)
nxt = torch.randint(0, 512, (512, 4), generator=g)
for name, n in [("fineweb_train.pt", 300_000), ("fineweb_val.pt", 60_000)]:
    idx = torch.empty(n, dtype=torch.long)
    cur = 0
    choice = torch.randint(0, 4, (n,), generator=g)
    for t in range(n):
        idx[t] = cur
        cur = int(nxt[cur, choice[t]])
    toks = states[idx].to(torch.uint16) if hasattr(torch, "uint16") else states[idx].to(torch.int32)
    torch.save({"tokens": toks, "doc_starts": torch.tensor([0]), "bos_id": 50256, "seq_shuffle_seed": 0, "seq_size": 129}, os.path.join(data_dir, name))
common = ["--data-dir", data_dir, "--train-tokens", "260000", "--val-tokens", "50000", "--train-eval-tokens", "20000",
          "--seq-len", "64", "--n-layer", "2", "--n-embd", "64", "--n-head", "4", "--batch-seqs", "32", "--mg-groups", "4",
          "--epochs", "1", "--eval-every", "40", "--log-every", "40", "--mg-log-every", "10", "--device", str(DEV)]
compiles = ["0"] if targs.skip_compile else ["0", "1"]
for comp in compiles:
    for mode in ["off", "descent", "random", "ascent"]:
        out = os.path.join(tmp, f"{mode}_c{comp}")
        res = proxy.main(common + ["--mg", mode, "--compile", comp, "--out", out])
        with open(os.path.join(out, "result.json")) as f:
            res2 = json.load(f)
        h = res2["history"]["val"]
        ok = all(k in res2 for k in ["final_val", "best_val", "final_train_eval", "mg_cos", "history"]) and \
            all(map(lambda v: v == v and v < 1e9, h)) and h[-1] < 6.0   # ln(50257) = 10.83; the chain's entropy is ln 4 = 1.39: must learn a lot
        signs = {s for _, s, _ in res2["mg_cos"]}
        if mode == "off":
            ok = ok and len(res2["mg_cos"]) == 0
        elif mode == "random":
            ok = ok and signs == {1.0, -1.0}
        else:
            ok = ok and signs == ({1.0} if mode == "descent" else {-1.0})
        check(f"main --mg {mode} --compile {comp}", ok, f"(val {[round(v, 3) for v in h]}, cos signs {sorted(signs)})")

if not targs.skip_compile:
    compile_test()

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("ALL TESTS PASSED")
