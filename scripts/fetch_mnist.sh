#!/usr/bin/env bash
# Fetch MNIST as uncompressed IDX files for the SNN substrate
# (snc_train / snc_cotrain --dataset mnist --data-dir data/mnist).
#
#   ./scripts/fetch_mnist.sh            # -> data/mnist/*-ubyte
#   ./scripts/fetch_mnist.sh /tmp/mn    # custom output dir
#
# Data is gitignored; run this once on a fresh checkout.
set -euo pipefail

OUT="${1:-data/mnist}"
BASE="https://storage.googleapis.com/cvdf-datasets/mnist"
FILES=(train-images-idx3-ubyte train-labels-idx1-ubyte
       t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte)

mkdir -p "$OUT"
for f in "${FILES[@]}"; do
  if [ -f "$OUT/$f" ]; then
    echo "[have] $f"
    continue
  fi
  echo "[get ] $f.gz"
  curl -fsSL "$BASE/$f.gz" -o "$OUT/$f.gz"
  gunzip -f "$OUT/$f.gz"
done
echo "MNIST ready in $OUT"
