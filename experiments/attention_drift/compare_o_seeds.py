"""Summarize matched output-projection freeze checkpoints across seeds."""

import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch


def loss_at(run, step):
    return torch.load(
        run / f"step-{step:08d}.pt", map_location="cpu", weights_only=True, mmap=True,
    )["metrics"]["validation_loss"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/o-seeds"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5])
    parser.add_argument("--step", type=int, default=8000)
    args = parser.parse_args()

    writer = csv.writer(sys.stdout)
    writer.writerow(("seed", "layer", "control_ce", "frozen_ce", "delta_ce"))
    controls = {
        seed: loss_at(args.runs / f"seed{seed}" / "control", args.step)
        for seed in args.seeds
    }
    for layer in args.layers:
        deltas = []
        for seed, control in controls.items():
            frozen = loss_at(args.runs / f"seed{seed}" / f"block{layer}", args.step)
            delta = frozen - control
            deltas.append(delta)
            writer.writerow((seed, layer, f"{control:.9f}", f"{frozen:.9f}", f"{delta:+.9f}"))
        wins = sum(delta < 0 for delta in deltas)
        print(f"Block {layer}: mean delta {statistics.mean(deltas):+.6f} nats, "
              f"range [{min(deltas):+.6f}, {max(deltas):+.6f}], "
              f"{wins}/{len(deltas)} seeds improve", file=sys.stderr)


if __name__ == "__main__":
    main()
