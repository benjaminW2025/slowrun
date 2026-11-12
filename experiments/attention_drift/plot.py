"""Plot attention drift, validation loss, and learning rate for one run."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    drift = read_rows(args.run / "drift.csv")
    training = read_rows(args.run / "training.csv")
    groups = defaultdict(list)
    for row in drift:
        key = tuple(int(row[field]) for field in ("lag_steps", "layer", "head"))
        groups[key].append((int(row["tokens_seen"]), float(row["tv_mean"])))
    lags = sorted({key[0] for key in groups})
    layers = sorted({key[1] for key in groups})
    colors = plt.get_cmap("viridis", len(layers))

    fig, axes = plt.subplots(
        len(lags) + 2, sharex=True, figsize=(9, 2.3 * (len(lags) + 2)),
        layout="constrained",
    )
    for lag, axis in zip(lags, axes):
        for (group_lag, layer, head), points in sorted(groups.items()):
            if group_lag == lag:
                axis.plot(
                    *zip(*points), color=colors(layers.index(layer)), alpha=0.65,
                    label=f"Layer {layer}" if head == 0 else None,
                )
        axis.set_title(f"Attention drift over {lag:,} steps")
        axis.set_ylabel("Mean TV")
        axis.set_ylim(bottom=0)
    axes[0].legend(loc="upper right", ncols=min(len(layers), 3))
    tokens = [int(row["tokens_seen"]) for row in training]
    axes[-2].plot(tokens, [float(row["validation_loss"]) for row in training])
    axes[-2].set_ylabel("Validation loss (nats)")
    axes[-1].plot(tokens, [float(row["lr"]) for row in training])
    axes[-1].set_ylabel("Learning rate")
    axes[-1].set_xlabel("Training tokens seen")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(args.run / "drift.png", dpi=160)


if __name__ == "__main__":
    main()
