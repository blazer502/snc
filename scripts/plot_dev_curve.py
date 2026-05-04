#!/usr/bin/env python3
"""
Plot the developmental trajectory written by `snc_demo schedule` to
`dev_curve.csv`. Shows side-by-side panels for:
  - Total / structural neuron count over time
  - Synapses formed vs pruned (cumulative + per-step deltas)
  - Per-area population maturation (motor / sensory / associate / language)
  - Spatial occupancy (NEURON+SYNAPSE voxel fraction)

Usage:
  python3 scripts/plot_dev_curve.py [csv_path]
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(path: Path):
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def col(rows, key, cast=float):
    return [cast(r[key]) for r in rows]


def main(argv):
    csv_path = Path(argv[1]) if len(argv) > 1 else Path("dev_curve.csv")
    if not csv_path.exists():
        print(f"missing {csv_path}", file=sys.stderr)
        return 1
    rows = load_csv(csv_path)
    if not rows:
        print("empty csv", file=sys.stderr)
        return 1

    step = col(rows, "step", int)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Developmental trajectory ({csv_path})")

    # Panel 1: neuron counts.
    ax = axes[0][0]
    ax.plot(step, col(rows, "total_neurons", int), label="seeded neurons")
    ax.plot(step, col(rows, "structural_neurons", int),
            label="structural (connected components)")
    ax.set_xlabel("step")
    ax.set_ylabel("neuron count")
    ax.set_title("Neuron population")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: synapses + per-step deltas.
    ax = axes[0][1]
    ax.plot(step, col(rows, "total_synapses", int), label="total synapses",
            color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(step, col(rows, "synapses_formed", int), alpha=0.5,
             color="tab:green", label="formed/step")
    ax2.plot(step, col(rows, "synapses_pruned", int), alpha=0.5,
             color="tab:red", label="pruned/step")
    ax.set_xlabel("step")
    ax.set_ylabel("total synapses")
    ax2.set_ylabel("delta / step")
    ax.set_title("Synaptogenesis vs. pruning")
    ax.grid(alpha=0.3)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")

    # Panel 3: per-area neurons (matures at staggered times).
    ax = axes[1][0]
    for area, color in [("motor", "tab:orange"),
                        ("sensory", "tab:green"),
                        ("associate", "tab:blue"),
                        ("language", "tab:purple")]:
        ax.plot(step, col(rows, f"{area}_n", int), label=area, color=color)
    ax.set_xlabel("step")
    ax.set_ylabel("neurons in area")
    ax.set_title("Per-area maturation (staggered)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 4: occupancy + max blob size.
    ax = axes[1][1]
    ax.plot(step, col(rows, "occupancy"), color="tab:purple",
            label="occupancy (frac)")
    ax2 = ax.twinx()
    ax2.plot(step, col(rows, "max_blob_size", int), color="tab:olive",
             label="max blob (voxels)")
    ax.set_xlabel("step")
    ax.set_ylabel("occupancy")
    ax2.set_ylabel("max connected blob size")
    ax.set_title("Spatial density")
    ax.grid(alpha=0.3)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.tight_layout()
    out = csv_path.with_suffix(".png")
    plt.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)
