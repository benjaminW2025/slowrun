#!/usr/bin/env bash
# Single-GPU MG study (proxy). Run from the repo root.
#
#   bash mg_study/run_proxy.sh test     # correctness tests on this GPU (~5-10 min)
#   bash mg_study/run_proxy.sh smoke    # 60 steps of descent: prints tok/s, memory, ETA for a full run
#   bash mg_study/run_proxy.sh step0    # off vs descent, seed 42: does MG help in the proxy at all?
#   bash mg_study/run_proxy.sh day1     # off / descent / random x seeds 42 43 44 (step0 runs are reused)
#   bash mg_study/run_proxy.sh summary
#
# Pick the GPU with CUDA_VISIBLE_DEVICES=0 (default: GPU 0). Extra args after the mode go to proxy.py,
# e.g. `bash mg_study/run_proxy.sh step0 --train-tokens 10000000` (use the same extras for every mode).
set -euo pipefail
MODE=${1:-}; shift || true
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=${PY:-python3}

run () {  # arm seed [extra...]
  local arm=$1 seed=$2; shift 2
  local out="runs_proxy/${arm}_s${seed}"
  if [ -f "$out/result.json" ]; then echo "skip $out (done)"; return; fi
  echo "=== $out ==="
  $PY mg_study/proxy.py --mg "$arm" --seed "$seed" --out "$out" "$@"
}

case "$MODE" in
  test)    $PY mg_study/test_proxy.py "$@" ;;
  smoke)   $PY mg_study/proxy.py --mg descent --max-steps 60 --log-every 10 --eval-every 1000000 --out runs_proxy/_smoke "$@"
           echo; echo "Full-run ETA = the 'eta' column x (total steps / 60) once tok/s has settled; see the [proxy] lines above." ;;
  step0)   run off 42 "$@"; run descent 42 "$@"; $PY mg_study/summarize_proxy.py ;;
  day1)    for s in 42 43 44; do run off "$s" "$@"; run descent "$s" "$@"; run random "$s" "$@"; done
           $PY mg_study/summarize_proxy.py ;;
  summary) $PY mg_study/summarize_proxy.py ;;
  *)       sed -n '2,12p' "$0"; exit 1 ;;
esac
