#!/usr/bin/env python3
"""Pack V (multimodal validation) -- MNIST + voice harness.

Drives ``snc_chat`` to validate the brain on real-world handwriting
data while pairing each image with the spoken digit name (cochlea +
label). Three phases:

  1. Bootstrap   -- short babble + plain teach for the 4 digits, so
                    label/voice engrams exist before image binding.
                    Skipped when ``--load`` points to an already-trained
                    brain (e.g. ``lifetime_brain.snc``).
  2. Train       -- one ``image_teach`` + ``correct`` per training row.
                    A short sleep cycle is inserted every 20 rows.
  3. Test        -- one ``image_test`` per held-out row; the parsed
                    ``[image_test] top=<word>`` is compared to the true
                    label. Per-class and overall accuracy are reported.

Usage:
  python3 scripts/prep_mnist.py            # one-time: build CSVs
  python3 scripts/run_mnist.py             # bootstrap from scratch
  python3 scripts/run_mnist.py --load lifetime_brain.snc   # warm start

The brain state is persisted to ``mnist_brain.snc`` (or ``--save <p>``).
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAT_BIN = REPO_ROOT / "build" / "snc_chat"
DATA_DIR = REPO_ROOT / "data"

DIGIT_WORDS = ["one", "two", "three", "four"]

TEST_RE = re.compile(r"\[image_test\] top=(\S+)")
# After "rates=" the chat prints "<word>:<float>" pairs separated by spaces.
RATE_RE = re.compile(r"\[image_test\][^\n]*?rates=([^\n]+)")


def read_csv(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 17:
                continue
            label = parts[0]
            pixels = [float(x) for x in parts[1:]]
            rows.append((label, pixels))
    return rows


def bootstrap_cmds() -> list[str]:
    cmds = ["babble 30"]
    for w in DIGIT_WORDS:
        cmds.extend([f"teach {w}", "correct", f"teach {w}", "correct"])
    return cmds


def train_cmds(rows, mode: str = "multimodal", seed: int = 0) -> list[str]:
    """Build the training command sequence.

    Modes:
      multimodal -- ``image_teach`` only (label + voice + image)
      visual     -- ``image_teach_visual`` only (image + motor prime)
      curriculum -- first half multimodal, second half visual
      cross      -- alternate every other trial (Pack V-cross). Each
                    visual-only trial forces the image pathway to fire
                    motor on its own, so weights cannot stay redundant.
                    Damasio 1989 convergence zones: cross-modal binding
                    *requires* each modality to retain an independent
                    feed-forward path to the convergence cell.
      dropout    -- random per-trial pick between multimodal and visual
                    (de Sa & Ballard 1998 cross-modal LMS). Equivalent
                    to ``modality dropout`` in deep learning.
    """
    import random
    rng = random.Random(seed)
    cmds = []
    n = len(rows)
    half = n // 2
    for i, (label, pixels) in enumerate(rows):
        pixel_str = " ".join(f"{p:.3f}" for p in pixels)
        if mode == "visual":
            verb = "image_teach_visual"
        elif mode == "curriculum":
            verb = "image_teach" if i < half else "image_teach_visual"
        elif mode == "cross":
            verb = "image_teach" if i % 2 == 0 else "image_teach_visual"
        elif mode == "dropout":
            verb = ("image_teach" if rng.random() < 0.5
                    else "image_teach_visual")
        else:
            verb = "image_teach"
        cmds.append(f"{verb} {label} {pixel_str}")
        cmds.append("correct")
        if (i + 1) % 20 == 0:
            cmds.append("sleep 30 20")
    return cmds


def test_cmds(rows) -> list[str]:
    cmds = []
    for _, pixels in rows:
        pixel_str = " ".join(f"{p:.3f}" for p in pixels)
        cmds.append(f"image_test {pixel_str}")
    return cmds


def run_chat(load: Path | None, save: Path | None, commands: list[str]) -> str:
    args = [str(CHAT_BIN), "--no-log"]
    if load and load.exists():
        args.extend(["--load", str(load)])
    cmds = list(commands)
    if save:
        cmds.append(f"save {save}")
    cmds.append("quit")
    proc = subprocess.run(
        args,
        input="\n".join(cmds) + "\n",
        capture_output=True,
        text=True,
        timeout=900,
    )
    return proc.stdout


def parse_predictions(out: str) -> list[str]:
    return TEST_RE.findall(out)


def parse_rates(out: str) -> list[dict]:
    """Per-image dict of {word: rate} from each ``[image_test]`` line."""
    result = []
    for tail in RATE_RE.findall(out):
        d = {}
        for tok in tail.split():
            if ":" not in tok:
                continue
            k, _, v = tok.partition(":")
            try:
                d[k] = float(v)
            except ValueError:
                pass
        result.append(d)
    return result


def constrained_argmax(rates: dict, choices: list[str]) -> str:
    """Argmax over a subset of class names (e.g. only digit words)."""
    best_word = choices[0]
    best_rate = -1.0
    for w in choices:
        r = rates.get(w, 0.0)
        if r > best_rate:
            best_rate = r
            best_word = w
    return best_word


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--load", type=Path, default=None,
                   help="warm-start from this brain")
    p.add_argument("--save", type=Path, default=Path("mnist_brain.snc"))
    p.add_argument("--no-bootstrap", action="store_true")
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--epochs", type=int, default=1,
                   help="passes over the training set (Pack V-cross)")
    p.add_argument("--mode",
                   choices=("multimodal", "visual", "curriculum",
                            "cross", "dropout"),
                   default="multimodal",
                   help="training mode (Pack V-tune / Pack V-cross)")
    args = p.parse_args(argv[1:])

    train_rows = read_csv(DATA_DIR / "mnist_train.csv")
    test_rows = read_csv(DATA_DIR / "mnist_test.csv")
    if not train_rows or not test_rows:
        print("error: missing data/mnist_{train,test}.csv -- run prep_mnist.py")
        return 1
    print(f"[mnist] train={len(train_rows)}  test={len(test_rows)}")

    # Phase 1: bootstrap (only if no warm-start brain provided)
    if args.load is None and not args.no_bootstrap:
        print("[mnist] phase 1: bootstrap (no prior brain)")
        out = run_chat(None, args.save, bootstrap_cmds())
        load_path = args.save
    else:
        load_path = args.load if args.load else args.save

    # Phase 2: train on MNIST images + voice + label
    if not args.no_train:
        for ep in range(args.epochs):
            print(f"[mnist] phase 2 ep {ep+1}/{args.epochs}: "
                  f"train {len(train_rows)} samples (mode={args.mode})")
            out = run_chat(load_path, args.save,
                           train_cmds(train_rows, mode=args.mode,
                                      seed=ep))
            load_path = args.save

    # Phase 3: test (visual-only)
    print(f"[mnist] phase 3: test {len(test_rows)} samples")
    out = run_chat(load_path, args.save, test_cmds(test_rows))
    preds = parse_predictions(out)
    rates = parse_rates(out)
    if len(preds) != len(test_rows):
        print(f"warning: got {len(preds)} predictions for {len(test_rows)} tests")

    n = min(len(preds), len(test_rows))

    # Two scoring modes:
    #   open    : raw argmax over all 20 vocabulary classes (hardest -- the
    #             brain may pick e.g. "ball" if its pixel-engram dominates)
    #   forced  : argmax restricted to the 4 digit classes (proves the
    #             multimodal binding produced *some* signal even if it
    #             can't yet beat overlapping engrams)
    print()
    print(f"=== MNIST 4-class results (digits 1..4)  mode={args.mode} ===")
    for mode in ("open", "forced"):
        correct = 0
        per_class_total = Counter()
        per_class_correct = Counter()
        confusion = Counter()
        for i in range(n):
            true = test_rows[i][0]
            if mode == "open":
                pred = preds[i]
            else:
                pred = (constrained_argmax(rates[i], DIGIT_WORDS)
                        if i < len(rates) else preds[i])
            per_class_total[true] += 1
            confusion[(true, pred)] += 1
            if pred == true:
                correct += 1
                per_class_correct[true] += 1
        print(f"\n[{mode}] overall: {correct}/{n} = "
              f"{100.0*correct/max(1,n):.1f}%")
        for w in DIGIT_WORDS:
            c = per_class_correct[w]
            t = per_class_total[w]
            pct = (100.0 * c / t) if t else 0.0
            print(f"  {w:6s}  {c:3d}/{t:3d}  ({pct:5.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
