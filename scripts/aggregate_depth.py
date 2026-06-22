#!/usr/bin/env python3
"""Aggregate the depth sweep (scripts/run_depth_sweep.sh): best test acc
mean +/- std per (depth, structure), and the static-snc - random-sparse gap."""
import csv, glob, os, re, statistics, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "data", "depth_sweep")


def best(path):
    b = 0.0
    with open(path) as f:
        for row in csv.DictReader(f):
            b = max(b, float(row["test_acc"]))
    return b


groups = {}
for c in sorted(glob.glob(os.path.join(OUT, "d*_*.csv"))):
    m = re.match(r"d(\d)_(static-snc|random-sparse)_s\d+\.csv$", os.path.basename(c))
    if not m:
        continue
    groups.setdefault((int(m.group(1)), m.group(2)), []).append(best(c))


def fmt(v):
    return f"{statistics.mean(v):.4f} +/- {statistics.pstdev(v) if len(v) > 1 else 0:.4f}"


print(f"\n{'depth':>5}  {'static-snc':>22}  {'random-sparse':>22}  {'gap':>8}")
for d in sorted({k[0] for k in groups}):
    s = groups.get((d, "static-snc"), [])
    r = groups.get((d, "random-sparse"), [])
    gap = (statistics.mean(s) - statistics.mean(r)) if s and r else float("nan")
    hidden = ",".join(["256"] * d)
    print(f"{d:>5}  {fmt(s):>22}  {fmt(r):>22}  {gap:>+8.4f}   ({hidden})")
print()
