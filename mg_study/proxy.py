"""
Single-GPU proxy for the split-batch first-order meta-gradient (MG) study.

The tiny-track record script needs an 8xH100 node (FA3, fp8 kernels, 8 ranks). This file is a plain
PyTorch re-implementation of the parts that matter for the mechanism question, sized for one GPU
(written for a 48GB L40):

  * small GPT (RoPE, QK-norm, relu^2 MLP, untied head), torch SDPA attention, bf16 autocast
  * Muon (Newton-Schulz) on the transformer matrices, AdamW on embedding / lm_head
  * data-limited regime: the first --train-tokens of fineweb_train.pt, repeated for --epochs
  * the record's MG step, emulating its 8 ranks: the batch is split into --mg-groups groups and each
    group does its own split pairing (first half adapts, second half is evaluated at the adapted
    point, the step is undone), exactly like one rank of the record
  * --mg off | descent (the record) | ascent | random (fair coin per (step, group))
    | even (antithetic: B evaluated at both theta -/+ eta*v, averaged; ~1.5x compute)

Outputs <out>/result.json and <out>/log.txt. See mg_study/README_proxy.md.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_args(argv=None):
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--data-dir", default="fineweb_data")
    p.add_argument("--train-tokens", type=int, default=20_000_000, help="training set = first N tokens of fineweb_train.pt")
    p.add_argument("--val-tokens", type=int, default=2_000_000, help="val set = first N tokens of fineweb_val.pt")
    p.add_argument("--train-eval-tokens", type=int, default=2_000_000, help="clean (no dropout, unperturbed) train-loss eval on the first N training tokens")
    p.add_argument("--seq-len", type=int, default=1024)
    # model
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    # optimisation
    p.add_argument("--batch-seqs", type=int, default=128, help="sequences per optimizer step")
    p.add_argument("--epochs", type=float, default=15)
    p.add_argument("--max-steps", type=int, default=-1, help="stop after this many steps (smoke tests); -1 = full run")
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--adam-lr", type=float, default=0.003)
    p.add_argument("--weight-decay", type=float, default=0.1, help="decoupled WD on the Muon matrices")
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--warmdown-frac", type=float, default=0.4)
    # meta-gradient
    p.add_argument("--mg", default="descent", choices=["off", "descent", "ascent", "random", "even"],
                   help="even = antithetic: B at theta -/+ eta*v averaged (exact even part, ~1.5x compute)")
    p.add_argument("--mg-step-norm", type=float, default=0.5, help="L2 norm of the temporary step on the plastic matrices")
    p.add_argument("--mg-groups", type=int, default=8, help="emulated ranks: each group pairs its own halves")
    p.add_argument("--mg-plastic", default="mlp", choices=["mlp", "matrix"])
    p.add_argument("--mg-log-every", type=int, default=25, help="log cos(g_A, g_B(theta')) every N steps (0 = off)")
    p.add_argument("--mg-eta-schedule", default="const", choices=["const", "lr", "adaptive"],
                   help="const: eta = --mg-step-norm; lr: --mg-step-norm x LR multiplier; adaptive: start at "
                        "--mg-step-norm and shrink eta whenever the measured cos(g_A, g_B(theta')) falls below --mg-cos-target")
    p.add_argument("--mg-cos-target", type=float, default=0.02, help="adaptive: target group-mean cosine")
    p.add_argument("--mg-adapt-rate", type=float, default=0.5, help="adaptive: log(eta) += rate * (cos - target) per step")
    p.add_argument("--mg-eta-min", type=float, default=0.005, help="adaptive: lower bound on eta (upper bound = --mg-step-norm)")
    # run
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="output dir (default runs_proxy/<mg>_s<seed>)")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------- model

def up(t):
    """At least fp32 (fp64 stays fp64, so the float64 reference tests are exact)."""
    return t if t.dtype in (torch.float32, torch.float64) else t.float()


def rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(up(x).pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


class Rotary(nn.Module):
    def __init__(self, head_dim, max_len):
        super().__init__()
        inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        f = torch.outer(t, inv)
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)

    def forward(self, x):  # x: (B, H, T, D)
        T = x.size(2)
        cos, sin = self.cos[:T].to(x.dtype), self.sin[:T].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Block(nn.Module):
    def __init__(self, d, n_head, max_len, dropout, n_layer):
        super().__init__()
        self.n_head = n_head
        self.attn_qkv = nn.Linear(d, 3 * d, bias=False)
        self.attn_proj = nn.Linear(d, d, bias=False)
        self.mlp_fc = nn.Linear(d, 4 * d, bias=False)
        self.mlp_proj = nn.Linear(4 * d, d, bias=False)
        self.rotary = Rotary(d // n_head, max_len)
        self.dropout = dropout
        for lin in (self.attn_qkv, self.mlp_fc):
            nn.init.normal_(lin.weight, std=0.02)
        for lin in (self.attn_proj, self.mlp_proj):
            nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * n_layer))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.attn_qkv(rms_norm(x)).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k = self.rotary(rms_norm(q)), self.rotary(rms_norm(k))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = self.attn_proj(y.transpose(1, 2).reshape(B, T, C))
        x = x + F.dropout(y, self.dropout, self.training)
        y = self.mlp_proj(F.relu(self.mlp_fc(rms_norm(x))).square())
        return x + F.dropout(y, self.dropout, self.training)


class GPT(nn.Module):
    def __init__(self, vocab, d, n_layer, n_head, max_len, dropout):
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        nn.init.normal_(self.wte.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(d, n_head, max_len, dropout, n_layer) for _ in range(n_layer)])
        self.lm_head = nn.Linear(d, vocab, bias=False)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, idx, targets):
        x = self.wte(idx)
        for b in self.blocks:
            x = b(x)
        logits = up(self.lm_head(rms_norm(x)))
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))


# ----------------------------------------------------------------------------- optimiser

def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.is_cuda else G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.mT
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon:
    def __init__(self, params, lr, momentum, weight_decay):
        self.params, self.lr, self.momentum, self.wd = list(params), lr, momentum, weight_decay
        self.bufs = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self, lr_mult):
        lr = self.lr * lr_mult
        for p, buf in zip(self.params, self.bufs):
            g = p.grad
            buf.mul_(self.momentum).add_(g)
            u = newton_schulz(g.add(buf, alpha=self.momentum))   # Nesterov
            u = u * max(1.0, p.size(0) / p.size(1)) ** 0.5
            p.mul_(1.0 - lr * self.wd)
            p.add_(u, alpha=-lr)


# ----------------------------------------------------------------------------- data

def load_tokens(path, n_tokens):
    if not os.path.exists(path):
        sys.exit(f"[proxy] missing {path}: run `python prepare_data.py` from the repo root first")
    d = torch.load(path, weights_only=True)
    toks = d["tokens"]
    if n_tokens > toks.numel():
        sys.exit(f"[proxy] asked for {n_tokens:,} tokens but {path} has {toks.numel():,}")
    return toks[:n_tokens].to(torch.int32)


def to_seqs(toks, seq_len):
    n = toks.numel() // (seq_len + 1)
    return toks[: n * (seq_len + 1)].view(n, seq_len + 1)


# ----------------------------------------------------------------------------- MG gradient

def plastic_params(model, which):
    out = []
    for name, p in model.named_parameters():
        if name.startswith("blocks.") and p.ndim == 2 and (which == "matrix" or ".mlp_" in name):
            out.append(p)
    return out


def step_sign(mode, rng):
    """+1 = descent (theta - eta*v, the record), -1 = ascent."""
    if mode == "descent":
        return 1.0
    if mode == "ascent":
        return -1.0
    return 1.0 if rng.random() < 0.5 else -1.0


def compute_grad(model, params, plastic, xb, yb, cfg, rng, log_cos=False, eta=None):
    """Fill p.grad (for every p in params) with the gradient the optimizer receives.

    The batch is split into cfg.mg_groups groups; each group's first half A gives a temporary step
    of L2 norm cfg.mg_step_norm along -sign * g_A/||g_A|| on the plastic matrices, its second half B
    is evaluated at that point, the step is undone (exact copy-back). The result is the mean over all
    2 * groups half-batch gradients, i.e. for --mg off exactly the ordinary full-batch mean gradient.
    Returns (mean unperturbed train loss over the A halves, list of (sign, cos) per group)."""
    eta = cfg.mg_step_norm if eta is None else eta
    G = cfg.mg_groups
    B = xb.size(0)
    assert B % (2 * G) == 0, f"batch {B} must be divisible by 2*mg_groups={2 * G}"
    gs, half = B // G, B // (2 * G)
    acc = [torch.zeros_like(p) for p in params]
    lossA_sum, cos_log = 0.0, []
    for i in range(G):
        xa, ya = xb[i * gs: i * gs + half], yb[i * gs: i * gs + half]
        xq, yq = xb[i * gs + half: (i + 1) * gs], yb[i * gs + half: (i + 1) * gs]
        for p in params:
            p.grad = None
        lossA = model(xa, ya)
        lossA.backward()
        lossA_sum += lossA.detach()
        if cfg.mg == "off":
            model(xq, yq).backward()
        else:
            # descent / ascent / random: one evaluation of B at theta - sign*eta*v.
            # even: B at both theta - eta*v and theta + eta*v, each weighted 1/2 (antithetic): the
            # update is then exactly the even-in-eta part on every step (odd orders cancel within the step).
            signs = [1.0, -1.0] if cfg.mg == "even" else [step_sign(cfg.mg, rng)]
            w = 1.0 / len(signs)
            with torch.no_grad():
                gA = [p.grad.detach().clone() for p in plastic]   # clone: autograd accumulates into p.grad in place
                norm = torch.sqrt(sum(up(g).pow(2).sum() for g in gA)).clamp_min(1e-12)
                saved = [p.detach().clone() for p in plastic]
            for k, sign in enumerate(signs):
                with torch.no_grad():
                    if k > 0:
                        for p, sv in zip(plastic, saved):
                            p.copy_(sv)
                    alpha = (-sign * eta / norm).to(gA[0].dtype)
                    for p, g in zip(plastic, gA):
                        p.add_(g * alpha)
                    pre = [up(p.grad).clone() for p in plastic] if log_cos else None
                (model(xq, yq) * w).backward()
                if log_cos:
                    with torch.no_grad():
                        dot = nA = nB = 0.0
                        for p, ga, pr in zip(plastic, gA, pre):
                            gb = (up(p.grad) - pr) / w
                            dot = dot + (up(ga) * gb).sum()
                            nA = nA + up(ga).pow(2).sum()
                            nB = nB + gb.pow(2).sum()
                        cos_log.append((sign, float(dot / (nA.sqrt() * nB.sqrt()).clamp_min(1e-30))))
            with torch.no_grad():
                for p, sv in zip(plastic, saved):
                    p.copy_(sv)
        with torch.no_grad():
            for a, p in zip(acc, params):
                if p.grad is not None:
                    a.add_(p.grad)
    for a, p in zip(acc, params):
        p.grad = a.div_(2 * G)
    return float(lossA_sum) / G, cos_log


def lr_multiplier(s, total_steps, cfg):
    w, d = max(1, int(cfg.warmup_frac * total_steps)), int(cfg.warmdown_frac * total_steps)
    if s < w:
        return (s + 1) / w
    if s >= total_steps - d:
        return max(0.0, (total_steps - s) / max(d, 1))
    return 1.0


def adapt_eta(eta, mean_cos, cfg):
    """Multiplicative controller: eta grows when the cosine is above target, shrinks when below,
    clipped to [--mg-eta-min, --mg-step-norm]."""
    eta = eta * math.exp(cfg.mg_adapt_rate * (mean_cos - cfg.mg_cos_target))
    return min(cfg.mg_step_norm, max(cfg.mg_eta_min, eta))


# ----------------------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(model, seqs, micro, device, autocast):
    was_training = model.training
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, seqs.size(0) - micro + 1, micro):
        s = seqs[i: i + micro].to(device, non_blocking=True).long()
        with autocast():
            tot += float(model(s[:, :-1], s[:, 1:])) * micro
        n += micro
    model.train(was_training)
    return tot / max(n, 1)


# ----------------------------------------------------------------------------- main

def main(argv=None):
    cfg = get_args(argv)
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg.device)
    out = cfg.out or os.path.join("runs_proxy", f"{cfg.mg}_s{cfg.seed}")
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "w", buffering=1)

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    sign_rng = random.Random(cfg.seed * 7919 + 1)   # separate stream: the sign coin never shifts data order or init
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        import contextlib
        autocast = contextlib.nullcontext

    train_seqs = to_seqs(load_tokens(os.path.join(cfg.data_dir, "fineweb_train.pt"), cfg.train_tokens), cfg.seq_len)
    val_seqs = to_seqs(load_tokens(os.path.join(cfg.data_dir, "fineweb_val.pt"), cfg.val_tokens), cfg.seq_len)
    train_eval_seqs = train_seqs[: max(1, cfg.train_eval_tokens // (cfg.seq_len + 1))]
    n_train = train_seqs.size(0)
    steps_per_epoch = n_train // cfg.batch_seqs
    assert steps_per_epoch > 0, f"train set has {n_train} sequences < batch {cfg.batch_seqs}"
    total_steps = int(round(steps_per_epoch * cfg.epochs))
    run_steps = total_steps if cfg.max_steps < 0 else min(cfg.max_steps, total_steps)
    micro = cfg.batch_seqs // (2 * cfg.mg_groups)

    model = GPT(50257, cfg.n_embd, cfg.n_layer, cfg.n_head, cfg.seq_len, cfg.dropout).to(device)
    params = [p for p in model.parameters()]
    matrices = [p for n, p in model.named_parameters() if n.startswith("blocks.") and p.ndim == 2]
    others = [p for n, p in model.named_parameters() if not (n.startswith("blocks.") and p.ndim == 2)]
    assert len(matrices) + len(others) == len(params)
    plastic = plastic_params(model, cfg.mg_plastic)
    muon = Muon(matrices, cfg.muon_lr, cfg.muon_momentum, cfg.weight_decay)
    adam = torch.optim.AdamW(others, lr=cfg.adam_lr, betas=(0.9, 0.95), weight_decay=0.0,
                             fused=(device.type == "cuda"))
    fwd = torch.compile(model, dynamic=False) if cfg.compile else model

    n_params = sum(p.numel() for p in params)
    plastic_norm = float(torch.sqrt(sum(p.detach().float().pow(2).sum() for p in plastic)))
    log(f"[proxy] device={device} params={n_params:,} plastic={sum(p.numel() for p in plastic):,} "
        f"(|theta_P|={plastic_norm:.2f}, step/|theta_P|={cfg.mg_step_norm / plastic_norm:.4f})")
    log(f"[proxy] train {n_train:,} seqs x {cfg.seq_len} ({n_train * cfg.seq_len:,} tok), {steps_per_epoch} steps/epoch, "
        f"{total_steps} total steps ({cfg.epochs} epochs), running {run_steps}; micro-batch {micro} seqs; "
        f"mg={cfg.mg} groups={cfg.mg_groups} step_norm={cfg.mg_step_norm} eta_schedule={cfg.mg_eta_schedule} plastic={cfg.mg_plastic}")
    log(f"[proxy] config {json.dumps(vars(cfg))}")

    def lr_mult(s):
        return lr_multiplier(s, total_steps, cfg)

    adaptive = cfg.mg != "off" and cfg.mg_eta_schedule == "adaptive"
    eta_state, eta_hist = cfg.mg_step_norm, []

    hist = {"step": [], "val": [], "train_eval": []}
    cos_hist = []   # (step, sign, cos)
    best_val, t0, tokens_done = float("inf"), time.time(), 0
    order, ptr, epoch = None, 0, -1
    for step in range(run_steps):
        if order is None or ptr + cfg.batch_seqs > n_train:
            epoch += 1
            g = torch.Generator().manual_seed(cfg.seed * 100_003 + epoch)
            order, ptr = torch.randperm(n_train, generator=g), 0
        batch = train_seqs[order[ptr: ptr + cfg.batch_seqs]].to(device, non_blocking=True).long()
        ptr += cfg.batch_seqs
        record_cos = cfg.mg != "off" and cfg.mg_log_every > 0 and step % cfg.mg_log_every == 0
        m = lr_mult(step)
        if cfg.mg_eta_schedule == "lr":
            eta_now = cfg.mg_step_norm * m
        elif adaptive:
            eta_now = eta_state
        else:
            eta_now = cfg.mg_step_norm
        eta_hist.append(eta_now)
        with autocast():
            train_loss, cos_all = compute_grad(fwd, params, plastic, batch[:, :-1], batch[:, 1:], cfg, sign_rng,
                                               record_cos or adaptive, eta=eta_now)
        if adaptive and cos_all:
            eta_state = adapt_eta(eta_state, sum(c for _, c in cos_all) / len(cos_all), cfg)
        cos_log = cos_all if record_cos else []
        for gr in adam.param_groups:
            gr["lr"] = cfg.adam_lr * m
        muon.step(m)
        adam.step()
        for p in params:
            p.grad = None
        tokens_done += cfg.batch_seqs * cfg.seq_len
        for sgn, c in cos_log:
            cos_hist.append((step, sgn, c))
        if cos_log:
            for sgn in (1.0, -1.0):
                cs = [c for s_, c in cos_log if s_ == sgn]
                if cs:
                    log(f"[mg-snr] step {step} | cos(gA,gB) {sum(cs) / len(cs):+.5f} | n {len(cs)} | sign {'+desc' if sgn > 0 else '-asc'}")
        if step % cfg.log_every == 0 or step == run_steps - 1:
            el = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
            eta = el / (step + 1) * (run_steps - step - 1) / 60
            log(f"step {step:05d}/{run_steps} ep {epoch} | train {train_loss:.4f} | lr x{m:.3f} | mg eta {eta_now:.4f} | "
                f"{tokens_done / el:,.0f} tok/s | mem {mem:.1f}GB | eta {eta:.1f}m")
        if (step + 1) % cfg.eval_every == 0 or step == run_steps - 1:
            v = evaluate(fwd, val_seqs, micro, device, autocast)
            te = evaluate(fwd, train_eval_seqs, micro, device, autocast)
            best_val = min(best_val, v)
            hist["step"].append(step + 1); hist["val"].append(v); hist["train_eval"].append(te)
            log(f"eval step {step + 1:05d} | val {v:.4f} | train(clean) {te:.4f} | gap {v - te:+.4f} | best val {best_val:.4f}")
            if not math.isfinite(v):
                log("[proxy] non-finite val loss, stopping")
                break

    result = {
        "config": vars(cfg), "steps_run": len(hist["step"]) and hist["step"][-1], "total_steps": total_steps,
        "final_val": hist["val"][-1], "best_val": best_val, "final_train_eval": hist["train_eval"][-1],
        "final_gap": hist["val"][-1] - hist["train_eval"][-1], "history": hist, "mg_cos": cos_hist,
        "eta_hist": eta_hist, "eta_mean": sum(eta_hist) / max(len(eta_hist), 1),
        "minutes": (time.time() - t0) / 60, "tok_per_s": tokens_done / (time.time() - t0),
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
        "plastic_norm_init": plastic_norm,
    }
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    log(f"[proxy] done: final val {result['final_val']:.4f} | best val {best_val:.4f} | "
        f"{result['minutes']:.1f} min | wrote {out}/result.json")
    return result


if __name__ == "__main__":
    main()
