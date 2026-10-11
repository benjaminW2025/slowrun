#!/usr/bin/env bash
# Start a fresh, read-only checkpoint profile from any working directory.
set -euo pipefail
profile_kind=${1:-muon}
profile_gpu=${2:-0}
profile_pairs=${3:-1}
profile_checkpoint=${4:-runs/audit-v-500/model.pt}
if [[ "$profile_kind" != muon && "$profile_kind" != abc ]]; then
  echo "Usage: bash research/rtx8000_meta/run_profile.sh {muon|abc} [gpu=0] [pairs=1] [checkpoint]" >&2
  exit 2
fi
profile_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$profile_root"
profile_python=python
if [[ -x .venv/bin/python ]]; then profile_python=.venv/bin/python; fi
profile_name="${profile_kind}-v-$(date +%Y%m%d-%H%M%S)-$$"
profile_extra=(--probe-muon-bprime)
if [[ "$profile_kind" == abc ]]; then
  profile_extra+=(--probe-abc --probe-transfer-radius 0.1)
fi
export CUDA_VISIBLE_DEVICES="$profile_gpu"
export WANDB_MODE=disabled
"$profile_python" research/rtx8000_meta/train.py \
  --init-model "$profile_checkpoint" \
  --mg-mode off --precision fp16 --loss-scale 128 --no-compile \
  --device-batch-size 1 --virtual-ranks 8 --total-batch-size 524288 \
  --schedule-steps 500 --max-train-steps 1 --eval-tokens 32768 \
  --iteration-schedule constant --num-iterations 2 --seed 42 \
  --probe-every 1 --probe-pairs "$profile_pairs" --probe-modes qk,v,mlp \
  --probe-radii 0.1,0.5 --probe-alignment-only --probe-only \
  "${profile_extra[@]}" --run-name "$profile_name"
"$profile_python" research/rtx8000_meta/summarize_probes.py \
  "runs/$profile_name/probes.jsonl" --muon-bprime --aggregate
if [[ "$profile_kind" == abc ]]; then
  "$profile_python" research/rtx8000_meta/summarize_probes.py \
    "runs/$profile_name/probes.jsonl" --transfer --aggregate
fi
echo "Profile saved to runs/$profile_name/probes.jsonl"
