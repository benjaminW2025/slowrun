#!/usr/bin/env bash
# Compare ordering at equal tokens; symmetric uses twice the backward evaluations.
set -euo pipefail
sym_gpu=${1:-0}
sym_seed=${2:-42}
sym_arm=${3:-mlp}
sym_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Optional final argument runs just one arm when splitting across GPUs.
read -r -a sym_orders <<< "${4:-oneway symmetric}"
for sym_order in "${sym_orders[@]}"; do
  TRAIN_STEPS=100 SCHEDULE_STEPS=500 MUON_WEIGHT_DECAY=cautious \
  MG_ORDER="$sym_order" MG_STEP_NORM=0.5 MG_MLP_STEP_NORM=0.5 MG_V_STEP_NORM=0.025 \
    bash "$sym_script_dir/run_training.sh" "$sym_arm" "$sym_gpu" "$sym_seed"
done
