"""Print the validation loss gap at each matched freeze and control checkpoint."""

import argparse
import csv
import sys
from pathlib import Path

import torch


def checkpoints(directory):
    paths = {
        int(path.stem.removeprefix("step-")): path
        for path in directory.glob("step-*.pt")
    }
    if not paths:
        raise ValueError(f"No checkpoints in {directory}")
    return paths


def validation_loss(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return checkpoint["metrics"]["validation_loss"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()

    try:
        control = checkpoints(args.runs / "freeze-control")
        comparisons = []
        for run in ("freeze-2k", "freeze-4k"):
            frozen = checkpoints(args.runs / run)
            fork_step = min(frozen)
            matched_steps = sorted(frozen.keys() & control.keys())
            if fork_step not in control or not any(step > fork_step for step in matched_steps):
                raise ValueError(f"{run} needs matched checkpoints at and after the fork")
            comparisons.append((run, frozen, fork_step, matched_steps))
    except ValueError as error:
        parser.error(str(error))

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("run", "step", "steps_since_freeze", "frozen_loss", "control_loss", "gap_nats"))
    control_losses = {}
    for run, frozen, fork_step, steps in comparisons:
        for step in steps:
            if step not in control_losses:
                control_losses[step] = validation_loss(control[step])
            baseline = control_losses[step]
            loss = validation_loss(frozen[step])
            writer.writerow(
                (run, step, step - fork_step, f"{loss:.6f}", f"{baseline:.6f}", f"{loss - baseline:+.6f}")
            )


if __name__ == "__main__":
    main()
