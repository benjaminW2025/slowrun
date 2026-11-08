#!/usr/bin/env bash
# Reduced-scale single-GPU testbed: the record's own tiny/train.py (SWA, dropout, MTP, recurrence, MG,
# best-epoch eval) at ~1/16-1/32 of the compute, for development on one GPU (L40). Run from the repo root:
#
#   CUDA_VISIBLE_DEVICES=1 bash mg_study/testbed.sh smoke             # ~5 min: prints tok/s and the ETA of a full run
#   CUDA_VISIBLE_DEVICES=1 bash mg_study/testbed.sh run NAME [args]   # one full testbed run -> runs/NAME/
#
# Scale knobs (environment): WIDTH (n_embd, default 384), HEADS (default 3 = head dim 128 as in the record),
# FRAC (fraction of the 100M training tokens, default 0.125), BATCH (global tokens/step, default 65536 = 32 seqs:
# about the record's step count at FRAC 0.125), DBS (sequences per micro-step = one emulated rank, default 8,
# so 4 emulated ranks). On one L40 the previous default (512 wide, 25% data) measured 1.57 s/step, ~90-110 min/run.
set -euo pipefail
MODE=${1:-}; shift || true
WIDTH=${WIDTH:-384}; HEADS=${HEADS:-3}; FRAC=${FRAC:-0.125}; BATCH=${BATCH:-65536}; DBS=${DBS:-8}
export WANDB_MODE=${WANDB_MODE:-offline}
TB="--n_embd $WIDTH --n_head $HEADS --train-frac $FRAC --total-batch-size $BATCH --device-batch-size $DBS --eval-tokens 2000000"
case "$MODE" in
  smoke) torchrun --standalone --nproc_per_node=1 tiny/train.py $TB --run-name tb_smoke --max-train-steps 40 "$@" ;;
  run)   NAME=$1; shift; torchrun --standalone --nproc_per_node=1 tiny/train.py $TB --run-name "$NAME" "$@" ;;
  *)     sed -n '2,10p' "$0"; exit 1 ;;
esac
