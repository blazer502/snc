#!/usr/bin/env bash
# Fetch the Spiking Speech Commands (SSC) dataset (Cramer et al. 2019): 35-class
# spoken commands as cochlear spike trains -- the larger sibling of SHD.
# Used by python/train_shd.py --dataset ssc. Data is gitignored.
#
#   ./scripts/fetch_ssc.sh            # -> data/ssc/ssc_{train,test}.h5
set -euo pipefail
OUT="${1:-data/ssc}"
BASE="https://zenkelab.org/datasets"
mkdir -p "$OUT"
for f in ssc_train ssc_test; do
  if [ -f "$OUT/$f.h5" ]; then echo "[have] $f.h5"; continue; fi
  echo "[get ] $f.h5.gz"
  curl -fsSL "$BASE/$f.h5.gz" -o "$OUT/$f.h5.gz"
  gunzip -f "$OUT/$f.h5.gz"
done
echo "SSC ready in $OUT (needs python h5py to load)"
