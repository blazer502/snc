#!/usr/bin/env bash
# Multi-seed surrogate-gradient BPTT sweep (error bars for experiments-mnist.md
# Experiment 4). depth {1,2,3} x structure {static-snc, random-sparse} + dense,
# full MNIST, several seeds, GPU (round-robined). Self-contained: waits for all
# runs then aggregates.
#
#   ./scripts/run_bptt_sweep.sh [OUT_DIR]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/data/bptt_seed}"
SEEDS="${SEEDS:-1 2 3}"
EPOCHS="${EPOCHS:-12}"
mkdir -p "$OUT"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -lt 1 ] && NGPU=1

COMMON="--num-train 60000 --num-test 10000 --epochs $EPOCHS --num-steps 20 \
  --batch 128 --lr 2e-3 --synapse-budget 40000"
CONFIGS=("static-snc 256 ss_d1" "static-snc 256,256 ss_d2" "static-snc 256,256,256 ss_d3"
         "random-sparse 256 rs_d1" "random-sparse 256,256 rs_d2" "random-sparse 256,256,256 rs_d3"
         "dense 256 dense_d1")

pids=(); gi=0
for s in $SEEDS; do
  for cfg in "${CONFIGS[@]}"; do
    set -- $cfg
    CUDA_VISIBLE_DEVICES=$((gi % NGPU)) python3 -u "$ROOT/python/train.py" \
      --structure "$1" --hidden "$2" $COMMON --seed "$s" \
      --log-csv "$OUT/$3_s$s.csv" >"$OUT/$3_s$s.log" 2>&1 &
    pids+=($!); gi=$((gi+1))
  done
done
echo "launched ${#pids[@]} BPTT runs into $OUT (ngpu=$NGPU); waiting..."
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "ALL DONE ($fail failures)"
python3 "$ROOT/scripts/aggregate_bptt.py" "$OUT"
