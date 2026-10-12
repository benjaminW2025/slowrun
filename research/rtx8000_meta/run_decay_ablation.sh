#!/usr/bin/env bash
# Run all four arms sequentially on one GPU. Settings are identical across arms.
set -euo pipefail
ablation_gpu=${1:-0}
ablation_seed=${2:-42}
ablation_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for ablation_decay in cautious standard; do
  for ablation_arm in off mlp; do
    MUON_WEIGHT_DECAY="$ablation_decay" bash "$ablation_script_dir/run_training.sh" \
      "$ablation_arm" "$ablation_gpu" "$ablation_seed"
  done
done
