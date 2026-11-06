#!/usr/bin/env bash
# Day 1: off / descent / random on the tiny track, 3 seeds each (~9 x 15-20 min on one 8xH100 node).
# Usage (from the repo root):  bash mg_study/run_day1.sh            # all 9 runs
#                              SEEDS="42" bash mg_study/run_day1.sh  # quick single-seed pass
set -euo pipefail
export WANDB_MODE=${WANDB_MODE:-offline}   # set WANDB_API_KEY + WANDB_MODE=online to log to wandb
SEEDS=${SEEDS:-"42 43 44"}
run () {  # name, extra args
  local name=$1; shift
  if [ -f "runs/$name/result.json" ]; then echo "skip $name (done)"; return; fi
  echo "=== $name ==="
  torchrun --standalone --nproc_per_node=8 tiny/train.py --run-name "$name" "$@"
}
for s in $SEEDS; do
  run "mg_off_s$s"     --seed "$s" --mg-every 0
  run "mg_desc_s$s"    --seed "$s" --mg-log-every 50
  run "mg_random_s$s"  --seed "$s" --mg-log-every 50 --mg-sign random
done
python3 mg_study/summarize.py
