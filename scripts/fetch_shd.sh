#!/usr/bin/env bash
# Fetch the Spiking Heidelberg Digits (SHD) dataset (Cramer et al. 2019) for the
# spiking SHD classifier (python/train_shd.py). Data is gitignored.
#
#   ./scripts/fetch_shd.sh            # -> data/shd/shd_{train,test}.h5
set -euo pipefail
OUT="${1:-data/shd}"
BASE="https://zenkelab.org/datasets"
mkdir -p "$OUT"
for f in shd_train shd_test; do
  if [ -f "$OUT/$f.h5" ]; then echo "[have] $f.h5"; continue; fi
  echo "[get ] $f.h5.gz"
  curl -fsSL "$BASE/$f.h5.gz" -o "$OUT/$f.h5.gz"
  gunzip -f "$OUT/$f.h5.gz"
done
echo "SHD ready in $OUT (needs python h5py to load)"
