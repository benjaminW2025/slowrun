"""
How accurately can the knob terms be measured in bf16? Run before choosing --mg-eta for --mg knobs.

With f(t) = grad L_B(theta - t v_A), the knobs use O = (f(e) - f(-e))/2 (Hessian term, ~ -e H v) and
E = (f(e) + f(-e))/2 - f(0) (third-derivative term, ~ e^2/2 T[v, v]). E is a small difference of large
gradients, so bf16 rounding can swamp it at small e. For each probe size e this script computes O and E
under bf16 autocast (what training uses) and in fp32 (TF32 off) on the same batch and checkpoint, and
prints the relative error of the bf16 version. Usage, from the repo root:

    # 1. a realistic checkpoint: ~600 steps of off (about a quarter of a run)
    python3 mg_study/proxy.py --mg off --max-steps 600 --eval-every 100000 --out runs_probe/ckpt --save-model runs_probe/ckpt.pt
    # 2. the measurement
    python3 mg_study/probe_precision.py --ckpt runs_probe/ckpt.pt

Rule of thumb: use an e whose bf16 relative error on E is well below 1 (say < 0.3); at that point the
beta-weighted term is mostly signal rather than rounding noise.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proxy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--eps", default="0.03125,0.0625,0.125,0.25,0.5")
ap.add_argument("--batches", type=int, default=3, help="independent A/B pairs to average over")
ap.add_argument("--data-dir", default="fineweb_data")
ap.add_argument("--train-tokens", type=int, default=20_000_000)
ap.add_argument("--micro", type=int, default=8, help="sequences per half-batch (proxy default: 128 / (2 * 8) = 8)")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = torch.device("cuda")
cfg = proxy.get_args([])   # proxy default model shape
model = proxy.GPT(50257, cfg.n_embd, cfg.n_layer, cfg.n_head, cfg.seq_len, 0.0).to(dev)
model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=True))
model.train()
params = list(model.parameters())
plastic = proxy.plastic_params(model, cfg.mg_plastic)
pidx = {id(p) for p in plastic}
seqs = proxy.to_seqs(proxy.load_tokens(os.path.join(a.data_dir, "fineweb_train.pt"), a.train_tokens), cfg.seq_len)
g = torch.Generator().manual_seed(a.seed)


def grads_at(x, y, shift, bf16):
    """Gradient of L_B at theta + shift (shift applied to the plastic matrices), flattened over all params."""
    saved = [p.detach().clone() for p in plastic]
    with torch.no_grad():
        for p, d in zip(plastic, shift):
            p.add_(d)
    for p in params:
        p.grad = None
    if bf16:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)
    else:
        loss = model(x, y)
    loss.backward()
    out = torch.cat([p.grad.detach().double().flatten() for p in params])
    with torch.no_grad():
        for p, s in zip(plastic, saved):
            p.copy_(s)
    return out


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
eps_list = [float(e) for e in a.eps.split(",")]
rows = {e: {"O": [], "E": [], "nO": [], "nE": [], "nf0": []} for e in eps_list}
for bi in range(a.batches):
    idx = torch.randint(0, seqs.size(0), (2 * a.micro,), generator=g)
    s = seqs[idx].to(dev).long()
    xa, ya, xb, yb = s[:a.micro, :-1], s[:a.micro, 1:], s[a.micro:, :-1], s[a.micro:, 1:]
    for p in params:
        p.grad = None
    model(xa, ya).backward()   # fp32: the direction itself, identical for both precisions
    gA = [p.grad.detach().clone() for p in plastic]
    v = [x / torch.sqrt(sum(y.pow(2).sum() for y in gA)) for x in gA]
    zero = [torch.zeros_like(x) for x in v]
    f0 = {bf: grads_at(xb, yb, zero, bf) for bf in (True, False)}
    for e in eps_list:
        res = {}
        for bf in (True, False):
            fp = grads_at(xb, yb, [-e * x for x in v], bf)   # theta - e v  (f(e), descent side)
            fm = grads_at(xb, yb, [e * x for x in v], bf)    # theta + e v  (f(-e))
            res[bf] = ((fp - fm) / 2, (fp + fm) / 2 - f0[bf])
        (O16, E16), (O32, E32) = res[True], res[False]
        r = rows[e]
        r["O"].append(float((O16 - O32).norm() / O32.norm()))
        r["E"].append(float((E16 - E32).norm() / E32.norm()))
        r["nO"].append(float(O32.norm())); r["nE"].append(float(E32.norm())); r["nf0"].append(float(f0[False].norm()))
    print(f"batch {bi + 1}/{a.batches} done", flush=True)

mean = lambda xs: sum(xs) / len(xs)
print(f"\n{'eps':>8} {'|f(0)|':>10} {'|O| fp32':>10} {'|E| fp32':>10} {'bf16 rel err O':>15} {'bf16 rel err E':>15}  "
      f"{'alpha, beta to match eta=0.125':>30}")
for e in eps_list:
    r = rows[e]
    al, be = 0.125 / e, (0.125 / e) ** 2
    print(f"{e:>8.5f} {mean(r['nf0']):>10.4e} {mean(r['nO']):>10.4e} {mean(r['nE']):>10.4e} "
          f"{mean(r['O']):>15.3f} {mean(r['E']):>15.3f}  {f'alpha={al:g}, beta={be:g}':>30}")
print("\nE should scale ~ eps^2 and O ~ eps while the probe stays below B's 1-D minimum; a bf16 rel err on E near or"
      "\nabove 1 means the bf16 E at that eps is mostly rounding noise.")
