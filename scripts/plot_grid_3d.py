#!/usr/bin/env python3
"""
3D voxel + connectome visualisation. Reads the three CSV files written
by `Simulator::dump_csv("prefix")` and renders:
  - left panel:  3D scatter of voxel states (NEURON / SYNAPSE / BLOCKED)
  - right panel: 3D plot of neuron somas connected by their synapses,
                 coloured by polarity, edge alpha proportional to weight

Usage:
  python3 scripts/plot_grid_3d.py [prefix]

Default prefix is `vocab` (i.e. expects vocab_voxels.csv etc.).
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_csv(path):
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# 2-bit grid cell states (matches BrainGrid::Cell).
STATE_LABEL = {1: "NEURON", 2: "SYNAPSE", 3: "BLOCKED"}
STATE_COLOR = {1: "tab:blue", 2: "tab:red", 3: "tab:gray"}

# Neuron polarity (matches NeuronPolarity).
POLARITY_LABEL = {0: "exc", 1: "PV", 2: "SST", 3: "VIP"}
POLARITY_COLOR = {0: "tab:blue", 1: "tab:red",
                  2: "tab:orange", 3: "tab:green"}


def main(argv):
    prefix = Path(argv[1]) if len(argv) > 1 else Path("vocab")
    voxels = load_csv(Path(f"{prefix}_voxels.csv"))
    neurons = load_csv(Path(f"{prefix}_neurons.csv"))
    synapses = load_csv(Path(f"{prefix}_synapses.csv"))
    print(f"voxels={len(voxels)}, neurons={len(neurons)}, "
          f"synapses={len(synapses)}")

    fig = plt.figure(figsize=(14, 7))

    # Left: voxel states.
    ax1 = fig.add_subplot(121, projection="3d")
    for state, label in STATE_LABEL.items():
        xs = [int(v["x"]) for v in voxels if int(v["state"]) == state]
        ys = [int(v["y"]) for v in voxels if int(v["state"]) == state]
        zs = [int(v["z"]) for v in voxels if int(v["state"]) == state]
        if xs:
            ax1.scatter(xs, ys, zs, label=f"{label} ({len(xs)})",
                        c=STATE_COLOR[state], s=4, alpha=0.45)
    ax1.set_title("Structural matrix (voxel states)")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax1.legend(loc="upper right")

    # Right: neuron graph -- somas + outgoing synapses (capped).
    ax2 = fig.add_subplot(122, projection="3d")
    by_id = {int(n["id"]): n for n in neurons}
    for pol, label in POLARITY_LABEL.items():
        xs = [int(n["soma_x"]) for n in neurons if int(n["polarity"]) == pol]
        ys = [int(n["soma_y"]) for n in neurons if int(n["polarity"]) == pol]
        zs = [int(n["soma_z"]) for n in neurons if int(n["polarity"]) == pol]
        if xs:
            ax2.scatter(xs, ys, zs, label=f"{label} ({len(xs)})",
                        c=POLARITY_COLOR[pol], s=14, alpha=0.85)

    # Edges: only the strongest 800 to keep the plot readable.
    sorted_syn = sorted(synapses, key=lambda s: -float(s["weight"]))[:800]
    for s in sorted_syn:
        pre = by_id.get(int(s["pre"]))
        post = by_id.get(int(s["post"]))
        if pre is None or post is None:
            continue
        x = [int(pre["soma_x"]), int(post["soma_x"])]
        y = [int(pre["soma_y"]), int(post["soma_y"])]
        z = [int(pre["soma_z"]), int(post["soma_z"])]
        w = float(s["weight"])
        # Inhibitory pre -> dashed red, excitatory pre -> solid blue.
        if int(pre["polarity"]) >= 1:
            color, ls = "tab:red", "--"
        else:
            color, ls = "tab:blue", "-"
        alpha = max(0.05, min(0.6, w / 1.5))
        ax2.plot(x, y, z, color=color, linestyle=ls, alpha=alpha,
                 linewidth=0.6)
    ax2.set_title("Connectome (somas + top-800 synapse weights)")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    out = Path(f"{prefix}_3d.png")
    plt.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)
