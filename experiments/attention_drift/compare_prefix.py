"""Compare time to a fixed validation loss for the prefix-freeze experiment."""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/prefix"))
    parser.add_argument("--target", type=float, default=5.90)
    parser.add_argument("--start-step", type=int, default=4000)
    parser.add_argument("--end-step", type=int, default=8000)
    args = parser.parse_args()

    summaries = []
    for run in ("control", "embeddings", "block0"):
        directory = args.runs / run
        metrics = []
        for path in sorted(directory.glob("step-*.pt")):
            row = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["metrics"]
            if args.start_step <= row["step"] <= args.end_step:
                metrics.append(row)
        if not metrics or metrics[0]["step"] != args.start_step or metrics[-1]["step"] != args.end_step:
            parser.error(f"{directory} needs checkpoints from step {args.start_step} to {args.end_step}")
        if any("training_seconds" not in row for row in metrics):
            parser.error(f"{directory} has checkpoints without training timings")
        first, last = metrics[0], metrics[-1]
        if args.target >= first["validation_loss"]:
            parser.error("Target must be lower than the starting validation loss")
        reached = next((row for row in metrics if row["validation_loss"] <= args.target), None)
        elapsed = last["training_seconds"] - first["training_seconds"]
        speed = 1000 * elapsed / (last["step"] - first["step"])
        target_step = reached["step"] if reached else "not reached"
        target_time = f'{reached["training_seconds"] - first["training_seconds"]:.3f}' if reached else ""
        summaries.append((run, f'{last["validation_loss"]:.6f}', f"{speed:.3f}", target_step, target_time))

    print("run,final_val_ce,train_ms_per_step,first_saved_step_at_target,train_seconds_to_target")
    for row in summaries:
        print(",".join(map(str, row)))


if __name__ == "__main__":
    main()
