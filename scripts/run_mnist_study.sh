#!/usr/bin/env bash
# Full-MNIST multi-seed study (new-plan.md Phase 7 / Experiments 1 & 2).
#
# Runs two experiments on the full MNIST set (60k train / 10k t10k test), three
# seeds each, all as parallel single-threaded processes (the trainer is
# CPU-only; this machine has many cores). Per-run CSV + log land in OUT.
#
#   ./scripts/run_mnist_study.sh [OUT_DIR]
#   python3 scripts/aggregate_study.py [OUT_DIR]   # summarise when done
#
# Exp 1 (frozen structure, e-prop): does structure-aware locality beat random
#        sparsity at an equal synapse budget? dense is a higher-budget reference.
# Exp 2 (two-timescale co-training): does dynamic rewiring beat static structure
#        at a tight budget?
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build"
OUT="${1:-$ROOT/data/mnist_study}"
mkdir -p "$OUT"

DATA="--dataset mnist --data-dir $ROOT/data/mnist"
COMMON="$DATA --hidden 256 --num-steps 20 --num-train 60000 --num-test 10000 --encoder poisson"
SEEDS="1 2 3"
BUDGET=40000           # equal budget for the sparse structures in Exp 1
EPOCHS=10

pids=()

# --- Experiment 1: frozen structure, equal budget --------------------------
for s in $SEEDS; do
  "$BIN/snc_train" $COMMON --epochs $EPOCHS --structure dense --seed $s \
    --log-csv "$OUT/exp1_dense_s$s.csv" >"$OUT/exp1_dense_s$s.log" 2>&1 &
  pids+=($!)
  for st in random-sparse static-snc; do
    "$BIN/snc_train" $COMMON --epochs $EPOCHS --structure "$st" \
      --synapse-budget $BUDGET --seed $s \
      --log-csv "$OUT/exp1_${st}_s$s.csv" >"$OUT/exp1_${st}_s$s.log" 2>&1 &
    pids+=($!)
  done
done

# --- Experiment 2: static vs dynamic co-training at tight budgets ----------
for s in $SEEDS; do
  for B in 3000 6000; do
    case $B in 3000) G=200;; 6000) G=500;; esac
    "$BIN/snc_cotrain" $COMMON --outer $EPOCHS --inner 1 --structural-budget $B \
      --grow 0 --seed $s \
      --log-csv "$OUT/exp2_b${B}_static_s$s.csv" >"$OUT/exp2_b${B}_static_s$s.log" 2>&1 &
    pids+=($!)
    "$BIN/snc_cotrain" $COMMON --outer $EPOCHS --inner 1 --structural-budget $B \
      --grow $G --seed $s \
      --log-csv "$OUT/exp2_b${B}_dynamic_s$s.csv" >"$OUT/exp2_b${B}_dynamic_s$s.log" 2>&1 &
    pids+=($!)
  done
done

echo "launched ${#pids[@]} runs into $OUT; waiting..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "ALL DONE ($fail failures)"
