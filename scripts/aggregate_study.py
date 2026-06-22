#!/usr/bin/env python3
"""Aggregate the full-MNIST study (scripts/run_mnist_study.sh) into a table.

For each config it reports best test accuracy mean +/- std across seeds (best =
max over epochs/rounds, the usual early-stopping figure). Reads the per-run CSVs
in OUT_DIR; synapse counts come from the matching .log header.
"""
import csv
import glob
import os
import re
import statistics
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "data", "mnist_study")


def best_test(csv_path):
    best = 0.0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            best = max(best, float(row["test_acc"]))
    return best


def synapses(log_path):
    try:
        with open(log_path) as f:
            m = re.search(r"synapses=(\d+)", f.read())
            return int(m.group(1)) if m else None
    except OSError:
        return None


def collect(pattern):
    """pattern -> {config_label: ([best per seed], synapse_count)}"""
    groups = {}
    for c in sorted(glob.glob(os.path.join(OUT, pattern))):
        label = re.sub(r"_s\d+\.csv$", "", os.path.basename(c))
        b = best_test(c)
        syn = synapses(c[:-4] + ".log")
        g = groups.setdefault(label, ([], syn))
        g[0].append(b)
    return groups


def fmt(vals):
    m = statistics.mean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.4f} +/- {s:.4f}  (n={len(vals)})"


def section(title, pattern):
    print(f"\n=== {title} ===")
    g = collect(pattern)
    if not g:
        print("  (no results yet)")
        return
    width = max(len(k) for k in g)
    for label in sorted(g):
        best, syn = g[label]
        syn_s = f"{syn:>7d} syn" if syn else "    ? syn"
        print(f"  {label:<{width}}  {syn_s}   best_test = {fmt(best)}")


section("Experiment 1: frozen structure, equal budget (e-prop)", "exp1_*.csv")
section("Experiment 2: static vs dynamic co-training", "exp2_*.csv")
print()
