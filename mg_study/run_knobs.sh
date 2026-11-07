#!/usr/bin/env bash
# Knob experiment (seed 42, eval every 100) into runs_eta/, paired with off_s42 and comparable to desc_eta0.125_s42.
# Emulates descent at eta = TARGET (default 0.125) with a small probe EPS (default 0.0625):
#   odd only  (alpha = TARGET/EPS, beta = 0)     -> the Hessian term alone
#   even only (alpha = 0, beta = (TARGET/EPS)^2) -> the third-derivative term alone
#   both      (alpha = TARGET/EPS, beta = (TARGET/EPS)^2) -> sanity check: should be close to desc_eta0.125
# Pick EPS with mg_study/probe_precision.py first. Run from the repo root:
#   CUDA_VISIBLE_DEVICES=1 EPS=0.0625 nohup bash mg_study/run_knobs.sh > knobs.log 2>&1 &
set -euo pipefail
PY=${PY:-python3}
S=${SEED:-42}; EPS=${EPS:-0.0625}; TARGET=${TARGET:-0.125}
A=$($PY -c "print($TARGET / $EPS)"); B=$($PY -c "print(($TARGET / $EPS) ** 2)")
echo "probe eps=$EPS, emulating eta=$TARGET: alpha=$A beta=$B"
COMMON="--mg knobs --seed $S --eval-every 100 --mg-step-norm $EPS --mg-eta-schedule const ${EXTRA:-}"
run () {  # out alpha beta
  if [ -f "$1/result.json" ]; then echo "skip $1 (done)"; return; fi
  echo "=== $1 (alpha=$2 beta=$3) ==="
  $PY mg_study/proxy.py $COMMON --mg-alpha "$2" --mg-beta "$3" --out "$1"
}
[ -f runs_eta/off_s$S/result.json ] || $PY mg_study/proxy.py --mg off --seed $S --eval-every 100 ${EXTRA:-} --out runs_eta/off_s$S
run runs_eta/knob_odd_e${EPS}_s$S  "$A" 0
run runs_eta/knob_even_e${EPS}_s$S 0 "$B"
run runs_eta/knob_both_e${EPS}_s$S "$A" "$B"
$PY mg_study/summarize_proxy.py runs_eta
