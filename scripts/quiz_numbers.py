#!/usr/bin/env python3
"""Pack V-quiz -- elementary-school-style number test.

After the brain has been trained multimodally on MNIST + voice (Pack
V-cross), this harness gives it a structured 3-section quiz on the
digits 1..4, mirroring how a primary-school worksheet pairs
*reading the word*, *recognising the printed digit*, and *recognising
handwriting* in one paper.

Sections
--------
A. **Symbolic recall** -- ``show <digit>``. Drives the label features
   directly; tests the pre-existing label -> motor pathway.
B. **Printed-digit visual** -- ``image_test`` on idealised 4x4 printed
   digit forms (NOT from MNIST). Different stroke geometry from
   handwriting, so this is a real transfer test: did the brain learn
   *number-shape* or just memorise MNIST pixel patterns?
C. **MNIST handwriting** -- ``image_test`` on held-out MNIST samples.
   Same distribution it trained on; the noisy real-world stimulus.

Output is a per-section + overall scorecard with closed-set
(4-digit-only) argmax, mirroring the multiple-choice format of a real
quiz. ``hear`` is intentionally omitted: the self-channel does not
project to motor in the current architecture, so an auditory section
would need a new acoustic-input pathway -- documented as a future pack.

Usage:
  python3 scripts/quiz_numbers.py [--brain mnist_brain.snc]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAT_BIN = REPO_ROOT / "build" / "snc_chat"

DIGIT_WORDS = ["one", "two", "three", "four"]

# Idealised printed-digit forms on a 4x4 grid.
#
#   index map:   0  1  2  3
#                4  5  6  7
#                8  9 10 11
#               12 13 14 15
#
# These are deliberately *different* from how MNIST mean-pools
# (handwritten 1s favour pixel 2/6/10/14; here 1 sits one column
# to the left, so the test verifies the brain learned a stroke
# concept rather than memorising the MNIST mean-pool centroid).
PRINTED = {
    "one": [
        0, 1, 0, 0,
        0, 1, 0, 0,
        0, 1, 0, 0,
        0, 1, 0, 0,
    ],
    "two": [
        0, 1, 1, 0,
        0, 0, 0, 1,
        0, 1, 1, 0,
        1, 1, 1, 0,
    ],
    "three": [
        0, 1, 1, 0,
        0, 0, 1, 1,
        0, 0, 1, 1,
        0, 1, 1, 0,
    ],
    "four": [
        0, 1, 0, 1,
        0, 1, 0, 1,
        0, 1, 1, 1,
        0, 0, 0, 1,
    ],
}

SHOW_RE = re.compile(r"\[show\] shown=(\S+)\s+said=(\S+)")
IT_RE = re.compile(r"\[image_test\][^\n]*?rates=([^\n]+)")


def run_chat(brain: Path, commands: list[str]) -> str:
    args = [str(CHAT_BIN), "--no-log"]
    if brain.exists():
        args.extend(["--load", str(brain)])
    cmds = list(commands) + ["quit"]
    proc = subprocess.run(
        args,
        input="\n".join(cmds) + "\n",
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc.stdout


def parse_show(out: str):
    return SHOW_RE.findall(out)


def parse_rates(out: str) -> list[dict]:
    result = []
    for tail in IT_RE.findall(out):
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
    best = choices[0]
    best_r = -1.0
    for w in choices:
        r = rates.get(w, 0.0)
        if r > best_r:
            best_r = r
            best = w
    return best


def read_mnist_test(path: Path, per_class: int = 4):
    """Pick the first ``per_class`` test rows for each digit."""
    counts = Counter()
    rows = []
    with path.open() as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 17:
                continue
            label = parts[0]
            if label not in DIGIT_WORDS:
                continue
            if counts[label] >= per_class:
                continue
            pixels = [float(x) for x in parts[1:]]
            rows.append((label, pixels))
            counts[label] += 1
            if all(counts[w] >= per_class for w in DIGIT_WORDS):
                break
    return rows


def build_quiz(mnist_rows):
    cmds = []
    # Section A: symbolic (show)
    for w in DIGIT_WORDS:
        cmds.append(f"show {w}")
    # Section B: printed digit
    for w in DIGIT_WORDS:
        pat = " ".join(f"{p:.3f}" for p in PRINTED[w])
        cmds.append(f"image_test {pat}")
    # Section C: MNIST handwriting
    for label, pixels in mnist_rows:
        pat = " ".join(f"{p:.3f}" for p in pixels)
        cmds.append(f"image_test {pat}")
    return cmds


def section_score(name: str, items: list[tuple[str, str]]) -> float:
    correct = sum(1 for true, pred in items if true == pred)
    total = len(items)
    pct = (100.0 * correct / total) if total else 0.0
    print(f"\n=== Section {name} ===")
    for i, (true, pred) in enumerate(items, 1):
        mark = "OK " if true == pred else "X  "
        print(f"  Q{i:2d}  truth={true:6s}  answer={pred:8s}  {mark}")
    print(f"  --> {correct}/{total} = {pct:.1f}%")
    return pct


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--brain", type=Path, default=Path("mnist_brain.snc"))
    p.add_argument("--mnist-test", type=Path,
                   default=REPO_ROOT / "data" / "mnist_test.csv")
    p.add_argument("--per-class", type=int, default=4,
                   help="MNIST samples per digit in section C")
    args = p.parse_args(argv[1:])

    if not args.brain.exists():
        print(f"error: brain {args.brain} not found. Run "
              "scripts/run_mnist.py first to produce a trained brain.")
        return 1

    mnist_rows = (read_mnist_test(args.mnist_test, args.per_class)
                  if args.mnist_test.exists() else [])
    if not mnist_rows:
        print("warning: no MNIST test data found; skipping Section C")

    out = run_chat(args.brain, build_quiz(mnist_rows))
    show_results = parse_show(out)
    rates = parse_rates(out)

    # Section A: 4 show items in DIGIT_WORDS order
    section_a = [(t, s) for t, s in show_results
                 if t in DIGIT_WORDS][: len(DIGIT_WORDS)]

    # Section B: first 4 image_test items are printed digits
    section_b = []
    for i, w in enumerate(DIGIT_WORDS):
        if i < len(rates):
            pred = constrained_argmax(rates[i], DIGIT_WORDS)
            section_b.append((w, pred))

    # Section C: remaining image_test items are MNIST handwriting
    section_c = []
    for i, (label, _) in enumerate(mnist_rows):
        idx = len(DIGIT_WORDS) + i
        if idx < len(rates):
            pred = constrained_argmax(rates[idx], DIGIT_WORDS)
            section_c.append((label, pred))

    print("=" * 60)
    print(" Elementary-school number quiz (digits 1..4) ")
    print("=" * 60)
    print(f"Brain: {args.brain}")
    a = section_score("A: read the word        (label features)", section_a)
    b = section_score("B: read the printed digit (idealised 4x4)", section_b)
    c = section_score("C: read the handwriting   (held-out MNIST)", section_c)
    overall = (a + b + c) / 3.0
    print()
    print("=" * 60)
    print(" SCORECARD")
    print("=" * 60)
    print(f"  Section A (symbolic):    {a:5.1f}%")
    print(f"  Section B (printed):     {b:5.1f}%")
    print(f"  Section C (handwriting): {c:5.1f}%")
    print(f"  Overall (avg of 3):      {overall:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
