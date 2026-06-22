#!/usr/bin/env python3
"""Aggregate the multi-seed BPTT sweep: best test acc mean +/- std per config."""
import csv, glob, os, re, statistics, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "data", "bptt_seed")
LABEL = {"ss_d1": "static-snc  1h", "ss_d2": "static-snc  2h", "ss_d3": "static-snc  3h",
         "rs_d1": "random-spr  1h", "rs_d2": "random-spr  2h", "rs_d3": "random-spr  3h",
         "dense_d1": "dense       1h"}


def best(path):
    return max(float(r["test_acc"]) for r in csv.DictReader(open(path)))


groups = {}
for c in sorted(glob.glob(os.path.join(OUT, "*_s*.csv"))):
    tag = re.sub(r"_s\d+\.csv$", "", os.path.basename(c))
    groups.setdefault(tag, []).append(best(c))

print(f"\n{'config':<16} {'best test acc':<22} seeds")
for tag in ["dense_d1", "ss_d1", "rs_d1", "ss_d2", "rs_d2", "ss_d3", "rs_d3"]:
    v = groups.get(tag)
    if not v:
        continue
    m = statistics.mean(v)
    sd = statistics.pstdev(v) if len(v) > 1 else 0.0
    print(f"{LABEL.get(tag, tag):<16} {m:.4f} +/- {sd:.4f}      {len(v)}")
print()
