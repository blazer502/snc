#!/usr/bin/env python3
"""
Render a developmental animation from a series of position-features
snapshots. snc_demo (schedule mode) writes `schedule_pf_step<N>.csv`
every 100 steps; this script reads them in order and produces:
  - a multi-panel PNG showing the cortical map at each snapshot
    (always available; uses matplotlib only)
  - optionally a GIF if `imageio` is installed

Usage:
  python3 scripts/animate_cortical_map.py [pattern] [region_size]

Defaults:
  pattern     = "schedule_pf_step*.csv"
  region_size = 8
"""

import csv
import glob
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_csv(path):
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def step_from_name(p):
    m = re.search(r"step(\d+)", p.name)
    return int(m.group(1)) if m else 0


def main(argv):
    pattern = argv[1] if len(argv) > 1 else "schedule_pf_step*.csv"
    region_size = int(argv[2]) if len(argv) > 2 else 8

    paths = sorted(Path(".").glob(pattern), key=step_from_name)
    if not paths:
        print(f"no files match {pattern}", file=sys.stderr)
        return 1
    print(f"found {len(paths)} snapshots: "
          f"{paths[0].name} .. {paths[-1].name}")

    # Compute global axis ranges and color scale across all frames so
    # successive panels are directly comparable.
    all_x, all_y, all_z, all_fr = [], [], [], []
    snapshots = []
    for p in paths:
        rows = load_csv(p)
        if not rows:
            continue
        bx = np.array([int(r["bin_x"]) for r in rows]) * region_size
        by = np.array([int(r["bin_y"]) for r in rows]) * region_size
        bz = np.array([int(r["bin_z"]) for r in rows]) * region_size
        n = np.array([int(r["n_neurons"]) for r in rows])
        fr = np.array([float(r["mean_fire_rate_ema"]) for r in rows])
        snapshots.append((step_from_name(p), bx, by, bz, n, fr))
        all_x.extend(bx); all_y.extend(by); all_z.extend(bz)
        all_fr.extend(fr)

    if not snapshots:
        print("no frames had data", file=sys.stderr)
        return 1

    fr_max = max(all_fr) if all_fr else 1.0
    n_frames = len(snapshots)

    # Choose a grid layout for the multi-panel summary.
    cols = min(4, n_frames)
    rows_n = (n_frames + cols - 1) // cols
    fig = plt.figure(figsize=(4.5 * cols, 4.0 * rows_n))
    for idx, (step, bx, by, bz, n, fr) in enumerate(snapshots):
        ax = fig.add_subplot(rows_n, cols, idx + 1, projection="3d")
        sc = ax.scatter(bx, by, bz, c=fr, s=np.maximum(8, n * 4),
                        cmap="viridis", alpha=0.85, vmin=0, vmax=fr_max)
        ax.set_title(f"step {step} ({len(bx)} bins)")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.colorbar(sc, ax=fig.axes, shrink=0.4,
                 label="mean fire_rate_ema")
    out_summary = Path("cortical_map_evolution.png")
    plt.savefig(out_summary, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_summary}  ({n_frames} frames)")

    # Try to also produce a GIF if imageio is available.
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("(imageio not installed -- skipping GIF; "
              "PNG summary above is enough)")
        return 0

    frame_paths = []
    for step, bx, by, bz, n, fr in snapshots:
        f = plt.figure(figsize=(6, 5))
        a = f.add_subplot(111, projection="3d")
        a.scatter(bx, by, bz, c=fr, s=np.maximum(8, n * 5),
                  cmap="viridis", alpha=0.85, vmin=0, vmax=fr_max)
        a.set_title(f"step {step}")
        a.set_xlabel("x"); a.set_ylabel("y"); a.set_zlabel("z")
        fp = Path(f"_frame_{step:05d}.png")
        plt.savefig(fp, dpi=90, bbox_inches="tight")
        plt.close(f)
        frame_paths.append(fp)
    images = [imageio.imread(p) for p in frame_paths]
    out_gif = Path("cortical_map_evolution.gif")
    imageio.mimsave(out_gif, images, duration=0.6)
    for p in frame_paths:
        p.unlink()
    print(f"wrote {out_gif}")


if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)
