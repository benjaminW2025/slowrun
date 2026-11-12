"""Train a small causal LM and measure within-run attention drift."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, x, query_positions=None):
        batch, length, width = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(batch, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind()
        )
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        probabilities = None
        if query_positions is not None:
            queries = q.index_select(-2, query_positions).float()
            logits = (queries @ k.float().transpose(-2, -1)) / math.sqrt(q.size(-1))
            masked = torch.arange(length, device=x.device) > query_positions[:, None]
            probabilities = logits.masked_fill(masked, -torch.inf).softmax(dim=-1)
        return self.output(output.transpose(1, 2).reshape(batch, length, width)), probabilities


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    @torch.no_grad()
    def split_mlp(self):
        # Start both architectures with the same linear weights and biases.
        # The two halves now act sequentially, each with its own residual.
        up, _, down = self.mlp
        width = up.in_features
        hidden = up.out_features // 2
        halves = []
        for rows in (slice(0, hidden), slice(hidden, None)):
            narrow_up = nn.Linear(width, hidden)
            narrow_down = nn.Linear(hidden, width)
            narrow_up.weight.copy_(up.weight[rows])
            narrow_up.bias.copy_(up.bias[rows])
            narrow_down.weight.copy_(down.weight[:, rows])
            narrow_down.bias.copy_(down.bias / 2)
            halves.append(nn.Sequential(narrow_up, nn.GELU(), narrow_down))
        self.mlp, self.second_mlp = halves
        self.second_mlp_norm = nn.LayerNorm(width)

    def forward(self, x, query_positions):
        attention, probabilities = self.attention(self.attention_norm(x), query_positions)
        x = x + attention
        x = x + self.mlp(self.mlp_norm(x))
        if hasattr(self, "second_mlp"):
            x = x + self.second_mlp(self.second_mlp_norm(x))
        return x, probabilities


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["width"]
        self.tokens = nn.Embedding(config["vocab_size"], width)
        self.positions = nn.Embedding(config["context"], width)
        self.blocks = nn.ModuleList(
            Block(width, config["heads"]) for _ in range(config["layers"])
        )
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, config["vocab_size"], bias=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.output.weight = self.tokens.weight
        if config.get("mlp_depth", 1) == 2:
            for block in self.blocks:
                block.split_mlp()

    def forward(self, tokens, query_positions=None):
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.tokens(tokens) + self.positions(positions)
        maps = []
        for block in self.blocks:
            x, probabilities = block(x, query_positions)
            if probabilities is not None:
                maps.append(probabilities.cpu())
        return self.output(self.norm(x)), maps


def load_tokens(path, vocab_size):
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        tokens = payload.get("tokens") if isinstance(payload, dict) else None
        if not isinstance(tokens, torch.Tensor):
            raise ValueError(f"{path} must be a Slowrun data file with a tokens tensor")
        data = tokens.numpy()
    elif path.suffix == ".npy":
        data = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        data = np.fromfile(path, dtype=np.uint8)
    if data.ndim != 1 or not np.issubdtype(data.dtype, np.integer):
        raise ValueError(f"{path} must contain a 1-D array of integer token IDs")
    if not data.size or data.min() < 0 or data.max() >= vocab_size:
        raise ValueError(f"{path} must contain token IDs in [0, {vocab_size})")
    return data


def read_windows(tokens, starts, context, device):
    windows = np.stack([tokens[start : start + context + 1] for start in starts])
    batch = torch.from_numpy(windows.astype(np.int64, copy=False)).to(device)
    return batch[:, :-1], batch[:, 1:]


def validation_starts(tokens, context, count, seed):
    available = (len(tokens) - 1) // context
    if count > available:
        raise ValueError(f"Validation needs {count * context + 1} tokens for {count} windows")
    return np.random.default_rng(seed).choice(available, count, replace=False) * context


def token_loss(logits, targets):
    return F.cross_entropy(logits.flatten(0, 1), targets.reshape(-1))


@torch.inference_mode()
def evaluate(model, tokens, starts, context, batch_size, device):
    losses = [
        token_loss(*eval_batch(model, tokens, batch, context, device)).item() * len(batch)
        for batch in np.array_split(starts, math.ceil(len(starts) / batch_size))
    ]
    return sum(losses) / len(starts)


def eval_batch(model, tokens, starts, context, device):
    inputs, targets = read_windows(tokens, starts, context, device)
    return model(inputs)[0], targets


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def train(args):
    config = vars(args).copy()
    config.pop("command")
    config["train"] = str(args.train.resolve())
    config["val"] = str(args.val.resolve())
    config["out"] = str(args.out.resolve())
    config["resume"] = str(args.resume.resolve()) if args.resume else None
    if args.width % args.heads:
        raise ValueError("--width must be divisible by --heads")
    if args.train.resolve() == args.val.resolve():
        raise ValueError("Use separate training and held-out validation files")
    config["val_sha256"] = fingerprint(args.val)
    config["train_sha256"] = fingerprint(args.train)
    if config["val_sha256"] == config["train_sha256"]:
        raise ValueError("Training and validation files have identical contents")
    train_tokens = load_tokens(args.train, args.vocab_size)
    val_tokens = load_tokens(args.val, args.vocab_size)
    if len(train_tokens) <= args.context:
        raise ValueError("Training data needs at least context + 1 tokens")

    device = torch.device(args.device)
    starts = validation_starts(val_tokens, args.context, args.val_windows, args.data_seed)
    torch.manual_seed(args.seed)
    model = Transformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sampler = np.random.default_rng(args.data_seed)
    start_step = 0
    start_lr = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if "training_state" not in checkpoint:
            raise ValueError("Resume requires a checkpoint created with --fork-at")
        state = checkpoint["training_state"]
        if state["config"].get("mlp_depth", 1) != args.mlp_depth:
            raise ValueError("Resume configuration differs for mlp_depth")
        for key in (
            "width", "heads", "layers", "context", "vocab_size", "batch_size",
            "seed", "data_seed", "lr", "warmup", "val_windows", "train_sha256", "val_sha256",
        ):
            if state["config"][key] != config[key]:
                raise ValueError(f"Resume configuration differs for {key}")
        if checkpoint["metrics"]["step"] >= args.steps:
            raise ValueError("--steps must be greater than the checkpoint step")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(state["optimizer"])
        sampler.bit_generator.state = json.loads(state["sampler"])
        start_step = checkpoint["metrics"]["step"]
        start_lr = checkpoint["metrics"]["lr"]

    early_steps = {2**power for power in range(8)}
    qkv = model.blocks[0].attention.qkv.weight
    qk = qkv[: 2 * args.width]
    frozen_qk = qk.detach().clone() if args.freeze_layer0_qk else None
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def save(step, lr):
        model.eval()
        loss = evaluate(model, val_tokens, starts, args.context, args.batch_size, device)
        metrics = {
            "step": step,
            "tokens_seen": step * args.batch_size * args.context,
            "lr": lr,
            "validation_loss": loss,
        }
        target = args.out / f"step-{step:08d}.pt"
        temporary = target.with_suffix(".tmp")
        checkpoint = {"model": model.state_dict(), "metrics": metrics}
        if step in args.fork_at:
            checkpoint["training_state"] = {
                "config": config,
                "optimizer": optimizer.state_dict(),
                "sampler": json.dumps(sampler.bit_generator.state),
            }
        torch.save(checkpoint, temporary)
        temporary.replace(target)
        print(json.dumps(metrics), flush=True)
        model.train()

    save(start_step, start_lr)
    for step in range(start_step + 1, args.steps + 1):
        lr = args.lr * min(1.0, step / max(1, args.warmup))
        optimizer.param_groups[0]["lr"] = lr
        offsets = sampler.integers(0, len(train_tokens) - args.context, args.batch_size)
        inputs, targets = read_windows(train_tokens, offsets, args.context, device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs)
        token_loss(logits, targets).backward()
        if frozen_qk is not None:
            qkv.grad[: 2 * args.width].zero_()
        optimizer.step()
        if frozen_qk is not None:
            # AdamW momentum and weight decay can still move zero-gradient rows.
            with torch.no_grad():
                qk.copy_(frozen_qk)
        if step in early_steps or step in args.fork_at or step % args.save_every == 0 or step == args.steps:
            save(step, lr)


@torch.inference_mode()
def probe(model, tokens, starts, positions, context, device):
    maps = []
    for start in starts:
        inputs, _ = read_windows(tokens, [start], context, device)
        _, layers = model(inputs, positions)
        maps.append(torch.stack(layers).squeeze(1))
    return torch.stack(maps, dim=2)  # layers, heads, windows, queries, keys


def analyze(args):
    config = json.loads((args.run / "config.json").read_text())
    val_path = args.val or Path(config["val"])
    if fingerprint(val_path) != config["val_sha256"]:
        raise ValueError("Validation data must match the original run")
    tokens = load_tokens(val_path, config["vocab_size"])
    context = config["context"]
    device = torch.device(args.device)
    starts = validation_starts(tokens, context, args.windows, config["data_seed"])
    positions = torch.arange(max(1, context // 4), context, args.query_stride, device=device)
    model = Transformer(config).to(device).eval()
    checkpoints = {
        int(path.stem.removeprefix("step-")): path
        for path in sorted(args.run.glob("step-*.pt"))
    }
    if not checkpoints:
        raise ValueError("No checkpoints found")
    pairs = {step: [lag for lag in args.lags if step - lag in checkpoints] for step in checkpoints}
    missing = set(args.lags) - {lag for lags in pairs.values() for lag in lags}
    if missing:
        raise ValueError(f"No checkpoint pairs for lags {sorted(missing)}")
    positions_cpu = positions.cpu()
    keys = torch.arange(context)
    nearby = (keys <= positions_cpu[:, None]) & (keys >= positions_cpu[:, None] - 7)

    # Retain a checkpoint's maps only until its last requested comparison.
    last_use = {
        step - lag: step
        for step in sorted(checkpoints)
        for lag in pairs[step]
    }
    cache = {}
    rows = []
    metrics = []
    for step in sorted(checkpoints):
        checkpoint = torch.load(checkpoints[step], map_location="cpu", weights_only=True)
        metrics.append(checkpoint["metrics"])
        if not pairs[step] and step not in last_use:
            continue
        model.load_state_dict(checkpoint["model"])
        current = probe(model, tokens, starts, positions, context, device)
        first_mass = current[..., 0].mean(dim=(-1, -2))
        local_mass = (current * nearby).sum(-1).mean(dim=(-1, -2))
        for lag in pairs[step]:
            earlier_step = step - lag
            earlier = cache[earlier_step]
            tv = (0.5 * (current - earlier).abs().sum(-1)).flatten(2)
            mean = tv.mean(-1)
            p95 = torch.quantile(tv, 0.95, dim=-1)
            for layer in range(config["layers"]):
                for head in range(config["heads"]):
                    rows.append({
                        "seed": config["seed"],
                        "step": step,
                        "earlier_step": earlier_step,
                        "lag_steps": lag,
                        "tokens_seen": checkpoint["metrics"]["tokens_seen"],
                        "lag_tokens": lag * config["batch_size"] * context,
                        "layer": layer,
                        "head": head,
                        "tv_mean": mean[layer, head].item(),
                        "tv_p95": p95[layer, head].item(),
                        "first_token_mass": first_mass[layer, head].item(),
                        "local_mass": local_mass[layer, head].item(),
                    })
            if last_use[earlier_step] == step:
                del cache[earlier_step]
        if step in last_use:
            cache[step] = current

    for name, records in (("drift", rows), ("training", metrics)):
        with (args.run / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    settings = {
        "lags": args.lags,
        "validation_window_starts": starts.tolist(),
        "query_positions": positions_cpu.tolist(),
    }
    (args.run / "probe.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Wrote {len(rows)} head comparisons to {args.run / 'drift.csv'}")


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    data_dir = Path(__file__).resolve().parents[2] / "fineweb_data"
    training.add_argument("--train", type=Path, default=data_dir / "fineweb_train.pt")
    training.add_argument("--val", type=Path, default=data_dir / "fineweb_val.pt")
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--vocab-size", type=positive)
    training.add_argument("--mlp-depth", type=int, choices=(1, 2), default=1,
                          help="Number of residual MLPs per attention block")
    training.add_argument("--fork-at", type=positive, nargs="+", default=[])
    training.add_argument("--resume", type=Path)
    training.add_argument("--freeze-layer0-qk", action="store_true")
    for name, default in {
        "steps": 12000, "save-every": 250, "warmup": 200, "batch-size": 16,
        "context": 256, "width": 192, "heads": 6, "layers": 6,
        "val-windows": 64,
    }.items():
        training.add_argument(f"--{name}", type=positive, default=default)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--data-seed", type=int, default=42)
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--run", type=Path, required=True)
    analysis.add_argument("--val", type=Path, help="Path to the same validation data if moved")
    analysis.add_argument("--lags", type=positive, nargs="+", default=[250, 2000])
    analysis.add_argument("--windows", type=positive, default=8)
    analysis.add_argument("--query-stride", type=positive, default=4)
    analysis.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.command == "train":
        if args.context < 2 or not math.isfinite(args.lr) or args.lr <= 0:
            parser.error("training requires context >= 2 and a finite positive learning rate")
        if args.freeze_layer0_qk and args.resume is None:
            parser.error("--freeze-layer0-qk requires --resume")
        if args.freeze_layer0_qk and args.fork_at:
            parser.error("--fork-at is only for the unfrozen control")
        if any(step > args.steps for step in args.fork_at):
            parser.error("--fork-at steps must be within the training run")
        for name in ("train", "val"):
            path = getattr(args, name)
            if not path.is_file():
                parser.error(f"--{name} file not found: {path}. Pass the actual data file path.")
        if args.vocab_size is None:
            args.vocab_size = 50257 if args.train.suffix == ".pt" else 256
        train(args)
    else:
        args.lags = sorted(set(args.lags))
        analyze(args)


if __name__ == "__main__":
    main()
