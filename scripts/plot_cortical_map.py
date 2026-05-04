#!/usr/bin/env python3
"""
Cortical-map visualisation. Reads a position-features CSV (written by
`Simulator::dump_position_features_csv`) and renders two 3D panels:
  - left:  bin density coloured by mean fire-rate EMA, marker size
           scaled to neuron count per bin
  - right: dominant input channel per bin (argmax of tuning_curve),
           coloured categorically

Usage:
  python3 scripts/plot_cortical_map.py [csv_path] [region_size]

Defaults: vocab_position_features.csv, region_size 8.
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def main(argv):
    path = Path(argv[1] if len(argv) > 1 else "vocab_position_features.csv")
    region_size = int(argv[2]) if len(argv) > 2 else 8
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        return 1

    rows = []
    tuning_cols = []
    with path.open() as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        tuning_cols = [c for c in cols if c.startswith("tuning_")]
        for row in reader:
            rows.append(row)
    if not rows:
        print("empty csv", file=sys.stderr)
        return 1

    bx = np.array([int(r["bin_x"]) for r in rows]) * region_size
    by = np.array([int(r["bin_y"]) for r in rows]) * region_size
    bz = np.array([int(r["bin_z"]) for r in rows]) * region_size
    n = np.array([int(r["n_neurons"]) for r in rows])
    fr = np.array([float(r["mean_fire_rate_ema"]) for r in rows])

    fig = plt.figure(figsize=(14, 6))

    # Left: bin density / activity heatmap.
    ax1 = fig.add_subplot(121, projection="3d")
    sc = ax1.scatter(bx, by, bz, c=fr, s=np.maximum(8, n * 6),
                     cmap="viridis", alpha=0.85)
    plt.colorbar(sc, ax=ax1, shrink=0.6, label="mean fire_rate_ema")
    ax1.set_title(f"Cortical map: {len(rows)} bins (size ~ neuron count)")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

    # Right: dominant tuning channel.
    ax2 = fig.add_subplot(122, projection="3d")
    if tuning_cols:
        tuning = np.array(
            [[float(r[c]) for c in tuning_cols] for r in rows])
        tuning_total = tuning.sum(axis=1)
        # Bins with no tuning input get -1 (rendered grey).
        dominant = np.where(tuning_total > 0,
                            np.argmax(tuning, axis=1), -1)
        # Categorical colour map.
        n_channels = tuning.shape[1]
        cmap = plt.cm.tab20
        colors = []
        for d in dominant:
            if d < 0:
                colors.append((0.5, 0.5, 0.5, 0.3))
            else:
                colors.append(cmap(d / max(1, n_channels)))
        ax2.scatter(bx, by, bz, c=colors, s=np.maximum(8, n * 6),
                    alpha=0.85)
        ax2.set_title(f"Dominant input channel (of {n_channels})")
    else:
        ax2.text2D(0.5, 0.5, "no tuning_curve columns",
                   ha="center", transform=ax2.transAxes)
        ax2.set_title("no tuning data")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")

    plt.tight_layout()
    out = path.with_suffix(".png")
    plt.savefig(out, dpi=110)
    print(f"wrote {out}  ({len(rows)} bins, {len(tuning_cols)} tuning cols)")


if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)
