#!/usr/bin/env bash
# Full-MNIST multi-seed study (new-plan.md Phase 7 / Experiments 1 & 2).
#
# Runs two experiments on the full MNIST set (60k train / 10k t10k test) over
# several seeds. Per-run CSV + log land in OUT.
#
#   ./scripts/run_mnist_study.sh [OUT_DIR]
#   python3 scripts/aggregate_study.py [OUT_DIR]
#
# Env knobs:
#   DEVICE=cuda|cpu     trainer device (default cuda; falls back to cpu if no GPU)
#   SNC_BIN=path        binary dir (default build-cuda for cuda, else build)
#   SEEDS="1 2 3 4 5"   seeds; EPOCHS=10; TRAIN_BATCH=32; GPU_LR=1.0; CPU_LR=0.08
#
# Exp 1 (frozen structure, e-prop): does structure-aware locality beat random
#        sparsity at an equal synapse budget? dense is a higher-budget reference.
# Exp 2 (two-timescale co-training): does dynamic rewiring beat static structure
#        at a tight budget? (Uses DEVICE too once snc_cotrain supports --device.)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-1 2 3 4 5}"
EPOCHS="${EPOCHS:-10}"
TRAIN_BATCH="${TRAIN_BATCH:-32}"
GPU_LR="${GPU_LR:-1.0}"
CPU_LR="${CPU_LR:-0.08}"
BUDGET=40000
OUT="${1:-$ROOT/data/mnist_study}"
mkdir -p "$OUT"

if [ "$DEVICE" = cuda ]; then BIN="${SNC_BIN:-$ROOT/build-cuda}"; else BIN="${SNC_BIN:-$ROOT/build}"; fi
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -lt 1 ] && NGPU=1

DATA="--dataset mnist --data-dir $ROOT/data/mnist"
COMMON="$DATA --hidden 256 --num-steps 20 --num-train 60000 --num-test 10000 --encoder poisson"

pids=(); gi=0
# snc_train with the right device flags; round-robins GPUs to avoid oversubscribe.
train() {  # args: tag structure [extra...]
  local tag="$1" st="$2"; shift 2
  local out="$OUT/$tag"
  if [ "$DEVICE" = cuda ]; then
    CUDA_VISIBLE_DEVICES=$((gi % NGPU)) "$BIN/snc_train" $COMMON --epochs $EPOCHS \
      --structure "$st" "$@" --device cuda --batch $TRAIN_BATCH --lr $GPU_LR \
      --log-csv "$out.csv" >"$out.log" 2>&1 &
    gi=$((gi+1))
  else
    "$BIN/snc_train" $COMMON --epochs $EPOCHS --structure "$st" "$@" --lr $CPU_LR \
      --log-csv "$out.csv" >"$out.log" 2>&1 &
  fi
  pids+=($!)
}

# --- Experiment 1: frozen structure, equal budget --------------------------
for s in $SEEDS; do
  train "exp1_dense_s$s"          dense          --seed $s
  train "exp1_random-sparse_s$s"  random-sparse  --synapse-budget $BUDGET --seed $s
  train "exp1_static-snc_s$s"     static-snc     --synapse-budget $BUDGET --seed $s
done

# snc_cotrain with the right device flags (same round-robin as train()).
cotrain() {  # args: tag budget grow seed
  local tag="$1" B="$2" G="$3" s="$4"; local out="$OUT/$tag"
  if [ "$DEVICE" = cuda ]; then
    CUDA_VISIBLE_DEVICES=$((gi % NGPU)) "$BIN/snc_cotrain" $COMMON --outer $EPOCHS \
      --inner 1 --structural-budget $B --grow $G --seed $s \
      --device cuda --batch $TRAIN_BATCH --lr $GPU_LR \
      --log-csv "$out.csv" >"$out.log" 2>&1 &
    gi=$((gi+1))
  else
    "$BIN/snc_cotrain" $COMMON --outer $EPOCHS --inner 1 --structural-budget $B \
      --grow $G --seed $s --lr $CPU_LR \
      --log-csv "$out.csv" >"$out.log" 2>&1 &
  fi
  pids+=($!)
}

# --- Experiment 2: static vs dynamic co-training at tight budgets ----------
for s in $SEEDS; do
  for B in 3000 6000; do
    case $B in 3000) G=200;; 6000) G=500;; esac
    cotrain "exp2_b${B}_static_s$s"  $B 0  $s
    cotrain "exp2_b${B}_dynamic_s$s" $B $G $s
  done
done

echo "launched ${#pids[@]} runs into $OUT (device=$DEVICE, ngpu=$NGPU); waiting..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "ALL DONE ($fail failures)"
