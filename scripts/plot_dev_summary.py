#!/usr/bin/env python3
"""
Combined developmental summary figure: dev_curve.csv trajectory metrics
on the top half, cortical-map snapshots on the bottom. Reads:
  - dev_curve.csv                    (snc_demo schedule output)
  - schedule_pf_step*.csv            (cortical-map snapshots, sampled
                                      every 100 steps by snc_demo)

Produces one PNG `dev_summary.png` so the whole developmental story
fits in a single figure that can be reviewed / shared.

Usage:
  python3 scripts/plot_dev_summary.py [region_size]
"""

import csv
import glob
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def col(rows, key, cast=float):
    return [cast(r[key]) for r in rows]


def step_from_name(p):
    m = re.search(r"step(\d+)", p.name)
    return int(m.group(1)) if m else 0


def main(argv):
    region_size = int(argv[1]) if len(argv) > 1 else 8
    dev_path = Path("dev_curve.csv")
    pf_paths = sorted(Path(".").glob("schedule_pf_step*.csv"),
                      key=step_from_name)

    if not dev_path.exists() and not pf_paths:
        print("nothing to plot: missing dev_curve.csv and "
              "schedule_pf_step*.csv", file=sys.stderr)
        return 1

    # Limit to ~3 cortical-map snapshots so the figure stays compact.
    n_snapshots = min(3, len(pf_paths))
    snap_idx = (np.linspace(0, len(pf_paths) - 1, n_snapshots, dtype=int)
                if pf_paths else np.array([]))

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.2],
                           hspace=0.42, wspace=0.32)

    # ---------- Top row: trajectory line plots ----------
    if dev_path.exists():
        dev = load_csv(dev_path)
        step = col(dev, "step", int)

        ax = fig.add_subplot(gs[0, 0])
        ax.plot(step, col(dev, "total_neurons", int), label="seeded")
        if "structural_neurons" in dev[0]:
            ax.plot(step, col(dev, "structural_neurons", int),
                    label="structural")
        ax.set_title("Neuron population")
        ax.set_xlabel("step"); ax.set_ylabel("count")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[0, 1])
        ax.plot(step, col(dev, "total_synapses", int), color="tab:blue")
        ax.set_title("Total synapses")
        ax.set_xlabel("step"); ax.set_ylabel("count")
        ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[0, 2])
        ax.plot(step, col(dev, "synapses_formed", int),
                color="tab:green", label="formed")
        ax.plot(step, col(dev, "synapses_pruned", int),
                color="tab:red", label="pruned")
        ax.set_title("Synaptogenesis vs pruning (per step)")
        ax.set_xlabel("step"); ax.set_ylabel("count")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[1, 0])
        for area, color in [("motor", "tab:orange"),
                            ("sensory", "tab:green"),
                            ("associate", "tab:blue"),
                            ("language", "tab:purple")]:
            if f"{area}_n" in dev[0]:
                ax.plot(step, col(dev, f"{area}_n", int),
                        label=area, color=color)
        ax.set_title("Per-area maturation (staggered)")
        ax.set_xlabel("step"); ax.set_ylabel("neurons in area")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[1, 1])
        if "occupancy" in dev[0]:
            ax.plot(step, col(dev, "occupancy"), color="tab:purple",
                    label="occupancy")
        if "max_blob_size" in dev[0]:
            ax2 = ax.twinx()
            ax2.plot(step, col(dev, "max_blob_size", int),
                     color="tab:olive", label="max blob (vox)")
            ax2.set_ylabel("max blob")
        ax.set_title("Spatial density")
        ax.set_xlabel("step"); ax.set_ylabel("occupancy frac")
        ax.grid(alpha=0.3)
        # Combined legend.
        h, l = ax.get_legend_handles_labels()
        try:
            h2, l2 = ax2.get_legend_handles_labels()  # noqa: F821
            h += h2; l += l2
        except NameError:
            pass
        ax.legend(h, l, fontsize=8, loc="upper left")

        ax = fig.add_subplot(gs[1, 2])
        if "spikes" in dev[0]:
            ax.plot(step, col(dev, "spikes", int), color="tab:gray")
            ax.set_title("Spikes per step")
            ax.set_xlabel("step"); ax.set_ylabel("spike count")
            ax.grid(alpha=0.3)
        else:
            ax.set_axis_off()

    # ---------- Bottom row: cortical-map snapshots ----------
    if n_snapshots > 0:
        # Compute global axis ranges so panels are comparable.
        all_x, all_y, all_z, all_fr, all_n = [], [], [], [], []
        for p in pf_paths:
            rows = load_csv(p)
            if not rows:
                continue
            all_x.extend(int(r["bin_x"]) * region_size for r in rows)
            all_y.extend(int(r["bin_y"]) * region_size for r in rows)
            all_z.extend(int(r["bin_z"]) * region_size for r in rows)
            all_fr.extend(float(r["mean_fire_rate_ema"]) for r in rows)
            all_n.extend(int(r["n_neurons"]) for r in rows)
        fr_max = max(all_fr) if all_fr else 1.0

        for i, idx in enumerate(snap_idx):
            p = pf_paths[idx]
            rows = load_csv(p)
            if not rows:
                continue
            bx = np.array([int(r["bin_x"]) for r in rows]) * region_size
            by = np.array([int(r["bin_y"]) for r in rows]) * region_size
            bz = np.array([int(r["bin_z"]) for r in rows]) * region_size
            n = np.array([int(r["n_neurons"]) for r in rows])
            fr = np.array([float(r["mean_fire_rate_ema"]) for r in rows])
            ax = fig.add_subplot(gs[2, i], projection="3d")
            sc = ax.scatter(bx, by, bz, c=fr, s=np.maximum(8, n * 4),
                            cmap="viridis", alpha=0.85,
                            vmin=0, vmax=fr_max)
            ax.set_title(f"step {step_from_name(p)} ({len(rows)} bins)")
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    plt.suptitle("Developmental summary", fontsize=14)
    out = Path("dev_summary.png")
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)
