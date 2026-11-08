#!/usr/bin/env bash
# eta-schedule experiment (seed 42, eval every 100), written into runs_eta/ next to the eta sweep so all
# arms are paired against the same off_s42 run. Run from the repo root:
#
#   CUDA_VISIBLE_DEVICES=1 nohup bash mg_study/run_eta_sched.sh > eta_sched.log 2>&1 &
#
# arms: descent with eta adaptive (start 0.5, keep cos(g_A, g_B(theta')) >= target), eta = 0.5 x LR multiplier,
# and eta constant at the adaptive run's mean eta (control: same average amount of MG, no cosine tracking).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"   # keep every cache/temp dir off $HOME
PY=${PY:-python3}
S=${SEED:-42}
COMMON="--seed $S --eval-every 100 ${EXTRA:-}"   # EXTRA: extra proxy.py args (tests only)
run () {  # out [args...]
  local out=$1; shift
  if [ -f "$out/result.json" ]; then echo "skip $out (done)"; return; fi
  echo "=== $out ==="
  $PY mg_study/proxy.py --mg descent $COMMON --out "$out" "$@"
}
[ -f runs_eta/off_s$S/result.json ] || $PY mg_study/proxy.py --mg off $COMMON --out runs_eta/off_s$S
run runs_eta/desc_adapt_s$S --mg-step-norm 0.5 --mg-eta-schedule adaptive
run runs_eta/desc_lrsched_s$S --mg-step-norm 0.5 --mg-eta-schedule lr
MEAN=$($PY -c "import json; print(round(json.load(open('runs_eta/desc_adapt_s$S/result.json'))['eta_mean'], 4))")
echo "adaptive mean eta = $MEAN"
run runs_eta/desc_constmean_s$S --mg-step-norm "$MEAN" --mg-eta-schedule const
$PY mg_study/summarize_proxy.py runs_eta
$PY - <<EOF
import json
for a in ["adapt", "lrsched", "constmean"]:
    eh = json.load(open(f"runs_eta/desc_{a}_s$S/result.json"))["eta_hist"]; n = len(eh)
    print(f"eta schedule {a:<10} " + "  ".join(f"Q{q + 1} {sum(eh[q * n // 4:(q + 1) * n // 4]) / max(1, n // 4):.4f}" for q in range(4)))
EOF
