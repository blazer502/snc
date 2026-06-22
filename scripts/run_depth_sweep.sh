#!/usr/bin/env bash
# Depth sweep: does network depth help, and is the structure-aware advantage
# robust to depth? Frozen-structure e-prop on full MNIST, GPU, multi-seed.
#
#   ./scripts/run_depth_sweep.sh [OUT_DIR]
#   python3 scripts/aggregate_depth.py [OUT_DIR]
#
# Holds synapse budget fixed across depths, so deeper = budget spread thinner.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${SNC_BIN:-$ROOT/build-cuda}"
OUT="${1:-$ROOT/data/depth_sweep}"
SEEDS="${SEEDS:-1 2 3}"
mkdir -p "$OUT"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -lt 1 ] && NGPU=1

COMMON="--dataset mnist --data-dir $ROOT/data/mnist --synapse-budget 40000 \
  --num-train 60000 --num-test 10000 --num-steps 20 --epochs 12 \
  --device cuda --batch 32 --lr 1.0"

# depth -> hidden-width spec
hidden_for() { case "$1" in 1) echo "256";; 2) echo "256,256";; 3) echo "256,256,256";; esac; }

pids=(); gi=0
for d in 1 2 3; do
  H=$(hidden_for $d)
  for st in static-snc random-sparse; do
    for s in $SEEDS; do
      tag="d${d}_${st}_s${s}"
      CUDA_VISIBLE_DEVICES=$((gi % NGPU)) "$BIN/snc_train" $COMMON \
        --hidden "$H" --structure "$st" --seed "$s" \
        --log-csv "$OUT/$tag.csv" >"$OUT/$tag.log" 2>&1 &
      pids+=($!); gi=$((gi+1))
    done
  done
done
echo "launched ${#pids[@]} runs into $OUT (ngpu=$NGPU); waiting..."
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "ALL DONE ($fail failures)"
