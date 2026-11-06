"""Matched off/MLP comparison. Run from anywhere; see baseline_comparison.md."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--execute", action="store_true", help="Launch training; otherwise print commands only")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    steps = 3 if args.stage == "smoke" else 500
    eval_tokens = 32768 if args.stage == "smoke" else 262144
    name = f"baseline-{args.stage}-s{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    common = ["--precision", "fp16", "--loss-scale", "128", "--no-compile",
              "--device-batch-size", "1", "--virtual-ranks", "8",
              "--total-batch-size", "524288", "--schedule-steps", "500",
              "--max-train-steps", str(steps), "--eval-tokens", str(eval_tokens),
              "--seed", str(args.seed), "--mg-order", "oneway", "--mg-step-norm", "0.5",
              "--muon-weight-decay", "cautious", "--dropout", "0.1", "--probe-every", "0"]
    if args.stage == "smoke":
        # Exercise the previously problematic recurrent backward immediately.
        common += ["--iteration-schedule", "constant", "--num-iterations", "2"]
    arms = ("off", "mlp")
    # Each arm runs in its own process with the same initialization and loader seed.
    commands = [[sys.executable, str(here / "train.py"), *common,
                 "--mg-mode", arm, "--run-name", f"{name}-{arm}"] for arm in arms]
    import shlex
    for command in commands:
        print(shlex.join(command), flush=True)
    if not args.execute:
        print("Dry run only. Add --execute to train both arms sequentially on the selected GPU.")
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, WANDB_MODE="disabled")
    subprocess.run([sys.executable, "-c",
                    "import torch, numpy, wandb, tiktoken; "
                    "assert torch.cuda.is_available(), 'CUDA GPU required'; "
                    "print(torch.cuda.get_device_name(0))"], env=env, check=True)
    datasets = [root / "fineweb_data" / f"fineweb_{split}.pt" for split in ("train", "val")]
    for path in datasets:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run prepare_data.py first")
    output = root / "runs" / name
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=root, text=True)
    (output / "source.diff").write_text(diff)
    hashes = {}
    for path in datasets:
        digest = hashlib.sha256()
        with path.open("rb") as data:
            for chunk in iter(lambda: data.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[path.name] = digest.hexdigest()
    manifest = dict(revision=revision, commands=commands, stage=args.stage,
                    gpu=args.gpu, seed=args.seed, dataset_sha256=hashes,
                    source_diff="source.diff", primary_metric="raw_final_val_loss")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = {}
    for arm, command in zip(arms, commands):
        started = time.monotonic()
        subprocess.run(command, cwd=root, env=env, check=True)
        elapsed = time.monotonic() - started
        result = json.loads((root / "runs" / f"{name}-{arm}" / "result.json").read_text())
        if result["steps"] != steps:
            raise RuntimeError(f"{arm}: expected {steps} steps, got {result['steps']}")
        results[arm] = dict(result=result, process_wall_seconds=elapsed)
        # Save each completed arm even if the next one fails.
        (output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    print("\narm     raw final loss    reported loss    model variant          process minutes")
    for arm in arms:
        row = results[arm]
        r = row["result"]
        print(f"{arm:6}  {r['raw_final_val_loss']:.6f}          {r['val_loss']:.6f}         "
              f"{r['model_variant']:22} {row['process_wall_seconds'] / 60:.2f}")
    delta = results['mlp']['result']['raw_final_val_loss'] - results['off']['result']['raw_final_val_loss']
    print(f"MLP minus off, raw final loss: {delta:+.6f} (negative favors meta-gradient)")
    print(f"Artifacts: {output}")
    if args.stage == "smoke":
        print("Smoke completion checks execution only; do not interpret its loss difference as an efficacy result.")


if __name__ == "__main__":
    main()
