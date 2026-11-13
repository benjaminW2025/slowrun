"""Compare an evaluation-only W_O swap with training around a frozen W_O."""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch

from experiment import Transformer, evaluate, fingerprint, load_tokens, validation_starts


def checkpoint(run, step):
    path = run / f"step-{step:08d}.pt"
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)["model"]


def compare_seed(root, args, validation_cache):
    control = root / "control"
    config = json.loads((control / "config.json").read_text())
    for layer in args.layers:
        if not 0 <= layer < config["layers"]:
            raise ValueError(f"Block {layer} is outside the model in {control}")

    path = (args.val or Path(config["val"])).resolve()
    cache_key = (path, config["vocab_size"])
    if cache_key not in validation_cache:
        validation_cache[cache_key] = fingerprint(path), load_tokens(path, config["vocab_size"])
    sha256, tokens = validation_cache[cache_key]
    if sha256 != config["val_sha256"]:
        raise ValueError(f"{path} does not match the validation data for {control}")

    context = config["context"]
    starts = validation_starts(tokens, context, config["val_windows"], config["data_seed"])
    model = Transformer(config).to(args.device).eval()
    early = checkpoint(control, args.from_step)
    late = checkpoint(control, args.step)

    def loss():
        return evaluate(model, tokens, starts, context, config["batch_size"],
                        args.device)["validation_loss"]

    model.load_state_dict(late)
    control_loss = loss()
    for layer in args.layers:
        # Each swap starts from the same fully trained control.
        model.load_state_dict(late)
        projection = model.blocks[layer].attention.output.weight
        key = f"blocks.{layer}.attention.output.weight"
        with torch.no_grad():
            projection.copy_(early[key])
        swapped_loss = loss()

        frozen = checkpoint(root / f"block{layer}", args.step)
        if not torch.equal(frozen[key], early[key]):
            raise ValueError(f"{root}/block{layer} did not retain W_O from step {args.from_step}")
        model.load_state_dict(frozen)
        yield layer, control_loss, swapped_loss, loss()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/o-seeds"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 3, 5])
    parser.add_argument("--from-step", type=int, default=4000)
    parser.add_argument("--step", type=int, default=8000)
    parser.add_argument("--val", type=Path, help="Path to the same validation data if moved")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0 <= args.from_step < args.step:
        parser.error("Require 0 <= --from-step < --step")

    writer = csv.writer(sys.stdout)
    writer.writerow(("seed", "layer", "control_ce", "swap_ce", "freeze_ce",
                     "swap_delta_ce", "freeze_delta_ce"))
    deltas = {layer: ([], []) for layer in args.layers}
    validation_cache = {}
    try:
        for seed in args.seeds:
            for layer, control, swapped, frozen in compare_seed(
                args.runs / f"seed{seed}", args, validation_cache
            ):
                swap_delta, freeze_delta = swapped - control, frozen - control
                deltas[layer][0].append(swap_delta)
                deltas[layer][1].append(freeze_delta)
                writer.writerow((seed, layer, f"{control:.9f}", f"{swapped:.9f}", f"{frozen:.9f}",
                                 f"{swap_delta:+.9f}", f"{freeze_delta:+.9f}"))
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    for layer, (swaps, freezes) in deltas.items():
        print(f"Block {layer}: mean swap delta {statistics.mean(swaps):+.6f}, "
              f"mean freeze delta {statistics.mean(freezes):+.6f} nats", file=sys.stderr)


if __name__ == "__main__":
    main()
