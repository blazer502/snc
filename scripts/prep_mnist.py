#!/usr/bin/env python3
"""Pack V (multimodal validation) -- MNIST data prep.

Downloads MNIST (idx ubyte format), downsamples each 28x28 image to a
4x4 binary pattern (mean-pool then threshold), filters to digits in
{1,2,3,4} so every sample maps to a word in the existing kWords vocab
(`one`, `two`, `three`, `four`), and writes:

    data/mnist_train.csv   first N samples per digit
    data/mnist_test.csv    next M samples per digit

CSV format (no header): ``label,p0,p1,...,p15`` where ``label`` is the
spoken digit name and ``p_i`` is 0 or 1 (binary mean-pool indicator).

Run once before ``run_mnist.py``. Cached files in data/mnist_raw/ are
re-used on subsequent runs.
"""
from __future__ import annotations

import gzip
import os
import struct
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "mnist_raw"
OUT_DIR = ROOT / "data"

# CVDF mirror -- Yann LeCun's original site is intermittently down.
BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

# Existing vocab: "one"/"two"/"three"/"four" cover digits 1-4 cleanly.
DIGIT_TO_WORD = {1: "one", 2: "two", 3: "three", 4: "four"}
DIGITS = sorted(DIGIT_TO_WORD.keys())

# Per-class sample counts. Keep small so the harness runs in seconds.
N_TRAIN_PER_CLASS = 30
N_TEST_PER_CLASS = 20


def fetch(name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fname = FILES[name]
    out = RAW_DIR / fname
    if out.exists() and out.stat().st_size > 0:
        return out
    url = BASE + fname
    print(f"[prep] downloading {url}", flush=True)
    urllib.request.urlretrieve(url, out)
    return out


def read_idx(path: Path) -> list:
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic == 2051:  # images
            n, rows, cols = struct.unpack(">III", f.read(12))
            buf = f.read(n * rows * cols)
            return n, rows, cols, buf
        elif magic == 2049:  # labels
            n = struct.unpack(">I", f.read(4))[0]
            buf = f.read(n)
            return n, 0, 0, buf
        else:
            raise RuntimeError(f"unexpected magic {magic} in {path}")


def downsample_to_4x4(img: bytes, rows: int, cols: int) -> list:
    """Mean-pool 28x28 -> 4x4, then binarise at half of max-mean."""
    assert rows == 28 and cols == 28
    block_r = rows // 4
    block_c = cols // 4
    pooled = [0.0] * 16
    for br in range(4):
        for bc in range(4):
            s = 0
            for dr in range(block_r):
                for dc in range(block_c):
                    r = br * block_r + dr
                    c = bc * block_c + dc
                    s += img[r * cols + c]
            pooled[br * 4 + bc] = s / float(block_r * block_c)
    # Binarise at 25% of full intensity (255 * 0.25 = ~64). MNIST
    # strokes saturate, so this gives a cleaner stroke mask.
    return [1 if p >= 64.0 else 0 for p in pooled]


def write_split(name: str, rows: list[tuple[str, list[int]]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    with out.open("w") as f:
        for word, pixels in rows:
            f.write(word + "," + ",".join(str(p) for p in pixels) + "\n")
    print(f"[prep] wrote {len(rows):4d} rows -> {out}")


def build_split(images_path: Path, labels_path: Path,
                per_class: int) -> list[tuple[str, list[int]]]:
    n_img, rows, cols, img_buf = read_idx(images_path)
    n_lbl, _, _, lbl_buf = read_idx(labels_path)
    assert n_img == n_lbl
    counts = {d: 0 for d in DIGITS}
    out = []
    for i in range(n_img):
        d = lbl_buf[i]
        if d not in DIGIT_TO_WORD:
            continue
        if counts[d] >= per_class:
            continue
        img = img_buf[i * rows * cols:(i + 1) * rows * cols]
        pixels = downsample_to_4x4(img, rows, cols)
        out.append((DIGIT_TO_WORD[d], pixels))
        counts[d] += 1
        if all(counts[d] >= per_class for d in DIGITS):
            break
    return out


def main():
    train_imgs = fetch("train_images")
    train_lbls = fetch("train_labels")
    test_imgs = fetch("test_images")
    test_lbls = fetch("test_labels")

    train = build_split(train_imgs, train_lbls, N_TRAIN_PER_CLASS)
    test = build_split(test_imgs, test_lbls, N_TEST_PER_CLASS)
    write_split("mnist_train.csv", train)
    write_split("mnist_test.csv", test)


if __name__ == "__main__":
    main()
