#!/usr/bin/env bash
# One fresh training run per invocation; launch on a separately allocated GPU.
set -euo pipefail
train_arm=${1:-off}
train_gpu=${2:-0}
train_seed=${3:-42}
train_steps=${TRAIN_STEPS:-500}
train_radius=${MG_STEP_NORM:-0.5}
train_mlp_radius=${MG_MLP_STEP_NORM:-0.5}
train_v_radius=${MG_V_STEP_NORM:-0.1}
case "$train_arm" in off|qk|v|mlp|qkv|mlp_v) ;; *) echo "Unknown arm: $train_arm" >&2; exit 2 ;; esac
train_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$train_root"
train_python=python
if [[ -x .venv/bin/python ]]; then train_python=.venv/bin/python; fi
train_tag="d${train_radius}"
if [[ "$train_arm" == mlp_v ]]; then train_tag="mlp${train_mlp_radius}-v${train_v_radius}"; fi
train_name="confirm-${train_arm}-${train_tag}-s${train_seed}-$(date +%Y%m%d-%H%M%S)-$$"
export CUDA_VISIBLE_DEVICES="$train_gpu"
export WANDB_MODE=disabled
"$train_python" research/rtx8000_meta/train.py \
  --mg-mode "$train_arm" --mg-step-norm "$train_radius" \
  --mg-mlp-step-norm "$train_mlp_radius" --mg-v-step-norm "$train_v_radius" \
  --precision fp16 --loss-scale 128 --no-compile \
  --device-batch-size 1 --virtual-ranks 8 --total-batch-size 524288 \
  --max-train-steps "$train_steps" --schedule-steps "$train_steps" \
  --eval-tokens 32768 --seed "$train_seed" --run-name "$train_name"
echo "Training results saved to runs/$train_name/result.json"
