#!/usr/bin/env bash
# Screen V perturbation norms; all arms share the first 100 steps of a 500-step schedule.
set -euo pipefail
sweep_gpu=${1:-0}
sweep_seed=${2:-42}
# Optional third argument allows a subset on another GPU, e.g. "0.25 0.5".
read -r -a sweep_radii <<< "${3:-0 0.025 0.1 0.25 0.5}"
sweep_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for sweep_radius in "${sweep_radii[@]}"; do
  echo "V sweep: MLP=0.5 V=$sweep_radius, 100 steps / 500-step schedule, cautious decay, seed=$sweep_seed"
  TRAIN_STEPS=100 SCHEDULE_STEPS=500 MUON_WEIGHT_DECAY=cautious \
  MG_ORDER=oneway MG_STEP_NORM=0.5 MG_MLP_STEP_NORM=0.5 MG_V_STEP_NORM="$sweep_radius" \
    bash "$sweep_script_dir/run_training.sh" mlp_v "$sweep_gpu" "$sweep_seed"
done
