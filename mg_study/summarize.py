"""Summarize the Day-1 runs: val loss per arm (mean +- std over seeds), late-training train loss, and the
logged cos(g_A, g_B(theta')) split by step sign. Reads runs/mg_*/result.json and runs/mg_*/terminal.log."""
import glob, json, os, re, statistics as st
from collections import defaultdict

def arm_of(name): return re.sub(r"_s\d+$", "", name)
val, best, trn, cos = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
for d in sorted(glob.glob("runs/mg_*")):
    name, arm = os.path.basename(d), arm_of(os.path.basename(d))
    rj, log = os.path.join(d, "result.json"), os.path.join(d, "terminal.log")
    if not os.path.exists(rj): print(f"(incomplete) {name}"); continue
    r = json.load(open(rj)); val[arm].append(r["val_loss"]); best[arm].append(r["best_val_loss"])
    lines = open(log, errors="ignore").read().splitlines()
    losses = [float(m.group(1)) for l in lines if l.startswith("step ") and (m := re.search(r"\| loss: ([\d.]+)", l))]
    if losses: trn[arm].append(st.mean(losses[-max(1, len(losses)//20):]))   # mean smoothed train loss over the last 5% of steps
    for l in lines:
        m = re.search(r"\[mg-snr\] step (\d+) \| cos\(gA,gB\) (-?[\d.]+) .*sign (\S+)", l)
        if m:
            cos[arm][m.group(3)].append((int(m.group(1)), float(m.group(2))))

def ms(x): return f"{st.mean(x):.4f} +- {st.stdev(x):.4f} (n={len(x)})" if len(x) > 1 else (f"{x[0]:.4f} (n=1)" if x else "-")
print(f"\n{'arm':<12} {'final val':<28} {'best val':<28} {'train (last 5%)':<28}")
for arm in sorted(val): print(f"{arm:<12} {ms(val[arm]):<28} {ms(best[arm]):<28} {ms(trn[arm]):<28}")

print("\ncos(g_A, g_B(theta')) by training phase (pooled over seeds); +desc = descent step, -asc = ascent step")
for arm in sorted(cos):
    for sign, pts in sorted(cos[arm].items()):
        pts.sort(); n = len(pts); q = [pts[i*n//4:(i+1)*n//4] for i in range(4)]
        print(f"  {arm:<12} {sign:<6} quarters: " + "  ".join(f"{st.mean(c for _, c in qq):+.4f}" for qq in q if qq))
if "mg_random" in cos and len(cos["mg_random"]) == 2:
    d, a = cos["mg_random"]["+desc"], cos["mg_random"]["-asc"]
    print(f"\n  random arm: SNR-cos ~ (desc+asc)/2 = {(st.mean(c for _, c in d)+st.mean(c for _, c in a))/2:+.4f}, "
          f"curvature term ~ (asc-desc)/2 = {(st.mean(c for _, c in a)-st.mean(c for _, c in d))/2:+.4f}")
