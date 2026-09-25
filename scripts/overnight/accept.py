#!/usr/bin/env python3
"""Join head-log 'Decode batch' lines to benchmark phase windows (UTC).

step_ms = 1000 * accept_len * running_req / gen_throughput, per logged interval."""
import json, re, statistics, subprocess, sys
from datetime import datetime
summ = json.load(open(sys.argv[1]))
ctn = sys.argv[2] if len(sys.argv) > 2 else "dsv41-head"
log = subprocess.run(["docker", "logs", "--since", "3h", ctn], capture_output=True, text=True)
lines = (log.stdout + log.stderr).splitlines()
pat = re.compile(r"^\[(\S+ \S+) TP0.*Decode batch, #running-req: (\d+).*accept len: ([\d.]+).*gen throughput \(token/s\): ([\d.]+)")
recs = []
for ln in lines:
    m = pat.search(ln)
    if m:
        recs.append((datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), int(m.group(2)), float(m.group(3)), float(m.group(4))))
out = {}
for name, ph in summ["phases"].items():
    a = datetime.strptime(ph["start"][:19], "%Y-%m-%dT%H:%M:%S"); b = datetime.strptime(ph["end"][:19], "%Y-%m-%dT%H:%M:%S")
    sel = [r for r in recs if a <= r[0] <= b and r[3] > 1.0]
    if not sel:
        print(f"{name:16s} no decode log lines"); continue
    acc = statistics.median(r[2] for r in sel)
    step = statistics.median(1000 * r[2] * r[1] / r[3] for r in sel)
    print(f"{name:16s} accept_len {acc:.2f}  step {step:.1f} ms  (n={len(sel)} intervals, bs={statistics.median(r[1] for r in sel)})")
    out[name] = {"accept_len": acc, "step_ms": step, "n": len(sel)}
json.dump(out, open(sys.argv[1].replace(".summary.json", ".accept.json"), "w"), indent=1)
