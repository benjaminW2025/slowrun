"""Summarize runs_proxy/<arm>_s<seed>/result.json: val / clean-train / gap per arm (mean +- std over seeds),
paired differences vs off (same seed = same init and data order), and the logged cosines by quarter."""
import glob
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

root = sys.argv[1] if len(sys.argv) > 1 else "runs_proxy"
runs = defaultdict(dict)   # arm -> seed -> result
for f in sorted(glob.glob(os.path.join(root, "*_s*", "result.json"))):
    m = re.match(r"(.+)_s(\d+)$", os.path.basename(os.path.dirname(f)))
    if m:
        runs[m.group(1)][int(m.group(2))] = json.load(open(f))
if not runs:
    sys.exit(f"no results under {root}/")


def ms(xs):
    if not xs:
        return "-"
    return f"{st.mean(xs):.4f} +- {st.stdev(xs):.4f} (n={len(xs)})" if len(xs) > 1 else f"{xs[0]:.4f} (n=1)"


order = [a for a in ["off", "descent", "random", "ascent"] if a in runs] + sorted(set(runs) - {"off", "descent", "random", "ascent"})
print(f"{'arm':<9} {'final val':<26} {'best val':<26} {'train (clean)':<26} {'gap val-train':<26}")
for a in order:
    r = list(runs[a].values())
    print(f"{a:<9} {ms([x['final_val'] for x in r]):<26} {ms([x['best_val'] for x in r]):<26} "
          f"{ms([x['final_train_eval'] for x in r]):<26} {ms([x['final_gap'] for x in r]):<26}")

if "off" in runs:
    print("\npaired vs off (arm - off, same seed; negative = better):")
    for a in order:
        if a == "off":
            continue
        seeds = sorted(set(runs[a]) & set(runs["off"]))
        if not seeds:
            continue
        dv = [runs[a][s]["final_val"] - runs["off"][s]["final_val"] for s in seeds]
        db = [runs[a][s]["best_val"] - runs["off"][s]["best_val"] for s in seeds]
        dt = [runs[a][s]["final_train_eval"] - runs["off"][s]["final_train_eval"] for s in seeds]
        print(f"  {a:<9} final val {ms(dv)} | best val {ms(db)} | clean train {ms(dt)}  seeds {seeds}")

print("\ncos(g_A(theta), g_B(theta')) mean by quarter of training (pooled over seeds and groups):")
for a in order:
    by_sign = defaultdict(list)
    for x in runs[a].values():
        n = x["total_steps"]
        for step, sign, c in x["mg_cos"]:
            by_sign[sign].append((min(3, int(4 * step / n)), c))
    for sign in sorted(by_sign, reverse=True):
        q = [[c for k, c in by_sign[sign] if k == i] for i in range(4)]
        print(f"  {a:<9} {'+desc' if sign > 0 else '-asc':<6} " + "  ".join(f"Q{i + 1} {st.mean(v):+.4f}" if v else f"Q{i + 1}   -   " for i, v in enumerate(q)))
    if set(by_sign) == {1.0, -1.0}:
        d, s_ = [c for _, c in by_sign[1.0]], [c for _, c in by_sign[-1.0]]
        print(f"  {a:<9} SNR-cos ~ (desc+asc)/2 = {(st.mean(d) + st.mean(s_)) / 2:+.4f} | curvature term ~ (asc-desc)/2 = {(st.mean(s_) - st.mean(d)) / 2:+.4f}")
