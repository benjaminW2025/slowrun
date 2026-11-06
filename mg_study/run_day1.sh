#!/usr/bin/env bash
# Day 1: off / descent / random on the tiny track, 3 seeds each (~9 x 15-20 min on one 8xH100 node).
# Usage (from the repo root):  bash mg_study/run_day1.sh            # all 9 runs
#                              SEEDS="42" bash mg_study/run_day1.sh  # quick single-seed pass
set -euo pipefail
export WANDB_MODE=${WANDB_MODE:-offline}   # set WANDB_API_KEY + WANDB_MODE=online to log to wandb
SEEDS=${SEEDS:-"42 43 44"}
# GPU selection: export CUDA_VISIBLE_DEVICES=0,1,2,3 to pick GPUs; NPROC defaults to how many are visible.
# The tiny-track MG step needs grad_accum == 1, i.e. device_batch_size * 2048 * NPROC == 524288,
# so with fewer than 8 GPUs the device batch is raised to keep the same global batch.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then NPROC=${NPROC:-$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)}; fi
NPROC=${NPROC:-8}
DBS=${DBS:-$((256 / NPROC))}
echo "NPROC=$NPROC device_batch_size=$DBS (global batch $((DBS * 2048 * NPROC)) tokens) GPUs=${CUDA_VISIBLE_DEVICES:-all}"
run () {  # name, extra args
  local name=$1; shift
  if [ -f "runs/$name/result.json" ]; then echo "skip $name (done)"; return; fi
  echo "=== $name ==="
  torchrun --standalone --nproc_per_node="$NPROC" tiny/train.py --run-name "$name" --device-batch-size "$DBS" "$@"
}
for s in $SEEDS; do
  run "mg_off_s$s"     --seed "$s" --mg-every 0
  run "mg_desc_s$s"    --seed "$s" --mg-log-every 50
  run "mg_random_s$s"  --seed "$s" --mg-log-every 50 --mg-sign random
done
python3 mg_study/summarize.py
