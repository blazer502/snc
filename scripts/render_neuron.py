#!/usr/bin/env python3
"""
Render the 3D shape of one neuron from a dumped SNC brain.

Workflow:
  1. Run a brief snc_chat session (default: bootstrap + teach mom),
     and emit a CSV dump of the brain via the `dump <prefix>` command.
  2. Parse <prefix>_voxels.csv (now carries owner + role per voxel
     after Pack 27).
  3. Filter to the chosen neuron id; render its body voxels in 3D
     with role-based colour (soma / dendrite / axon / axon-trunk).
  4. Save as PNG.

Usage:
  python3 scripts/render_neuron.py [neuron_id] [out_path]

If `neuron_id` is omitted, picks a representative pyramidal cell from
the cortical bulk (first INTERNAL EXCITATORY with a non-trivial body).
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D)


REPO_ROOT = Path(__file__).resolve().parent.parent
CHAT_BIN = REPO_ROOT / "build" / "snc_chat"
DUMP_PREFIX = REPO_ROOT / "neuron_dump"


def run_session_and_dump() -> None:
    """Run snc_chat briefly, teach a few words, and dump CSVs."""
    cmds = [
        "babble 30",
        "teach mom", "correct",
        "teach mom", "correct",
        "teach dad", "correct",
        "teach dad", "correct",
        "teach baby", "correct",
        f"dump {DUMP_PREFIX}",
        "quit",
    ]
    proc = subprocess.run(
        [str(CHAT_BIN), "--no-log"],
        input="\n".join(cmds) + "\n",
        capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        sys.exit(f"snc_chat exited {proc.returncode}")


def load_voxels(prefix: Path):
    """Read <prefix>_voxels.csv -> list of (x, y, z, state, owner, role)."""
    rows = []
    with open(f"{prefix}_voxels.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((
                int(row["x"]), int(row["y"]), int(row["z"]),
                int(row["state"]), int(row["owner"]), int(row["role"]),
            ))
    return rows


def load_neurons(prefix: Path):
    rows = []
    with open(f"{prefix}_neurons.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "id": int(row["id"]),
                "role": int(row["role"]),       # NeuronRole enum
                "polarity": int(row["polarity"]),
                "channel": int(row["channel"]),
                "soma": (int(row["soma_x"]), int(row["soma_y"]),
                         int(row["soma_z"])),
                "body_voxels": int(row["body_voxels"]),
            })
    return rows


def pick_default_neuron(neurons, voxels) -> int:
    """Pick a representative INTERNAL EXCITATORY cell with a non-trivial
    body (sprouted at least one extra voxel)."""
    by_owner = {}
    for (x, y, z, state, owner, role) in voxels:
        by_owner.setdefault(owner, []).append((x, y, z, state, role))
    candidates = []
    for n in neurons:
        if n["role"] != 0:               # 0 = INTERNAL
            continue
        if n["polarity"] != 0:           # 0 = EXCITATORY
            continue
        if n["body_voxels"] < 3:
            continue
        # Prefer cells in the cortical bulk (z away from input/output layers)
        sz = n["soma"][2]
        if sz < 5 or sz > 40:
            continue
        candidates.append(n)
    if not candidates:
        # Fall back to any neuron with body > 1
        candidates = [n for n in neurons if n["body_voxels"] > 1]
    if not candidates:
        sys.exit("no neurons with non-trivial body in dump")
    # Pick the one with the most body voxels
    candidates.sort(key=lambda n: -n["body_voxels"])
    return candidates[0]["id"]


SOMA_COLOUR = "#cc0000"   # bright red so the soma stands out among
                          # blue dendrites / orange axons
ROLE_COLOUR = {
    0: ("#9ecae1", "dendrite"),
    1: ("#fdae6b", "axon"),
    2: ("#bdbdbd", "axon trunk"),
}


def render_one(ax, target, own_voxels):
    sx, sy, sz = target["soma"]
    n_dend = sum(1 for v in own_voxels
                 if v[4] == 0 and not (v[0] == sx and v[1] == sy and v[2] == sz))
    n_ax   = sum(1 for v in own_voxels if v[4] == 1)
    n_trunk = sum(1 for v in own_voxels if v[4] == 2)
    pol_name = {0: "EXC", 1: "PV", 2: "SST", 3: "VIP"}.get(
        target.get("polarity", -1), "?")
    role_name = {0: "INTERNAL", 1: "INPUT", 2: "OUTPUT"}.get(
        target["role"], "?")
    ax.set_title(
        f"id={target['id']}  {pol_name}/{role_name}\n"
        f"soma=({sx},{sy},{sz})  body={target['body_voxels']}  "
        f"D={n_dend} A={n_ax} T={n_trunk}",
        fontsize=10)

    # Order voxels so the soma is rendered LAST (so it's not occluded
    # by dendrites drawn after it).
    soma_voxels = [v for v in own_voxels
                   if v[0] == sx and v[1] == sy and v[2] == sz]
    other_voxels = [v for v in own_voxels
                    if not (v[0] == sx and v[1] == sy and v[2] == sz)]

    for (x, y, z, state, role) in other_voxels:
        colour = ROLE_COLOUR.get(role, ("#cccccc", "?"))[0]
        ax.bar3d(x - 0.4, y - 0.4, z - 0.4, 0.8, 0.8, 0.8,
                 color=colour, edgecolor="#333", alpha=0.85,
                 shade=True)
    for (x, y, z, state, role) in soma_voxels:
        # Make the soma a slightly larger, brighter cube.
        ax.bar3d(x - 0.5, y - 0.5, z - 0.5, 1.0, 1.0, 1.0,
                 color=SOMA_COLOUR, edgecolor="black", alpha=1.0,
                 shade=True)

    pad = 3
    xs = [v[0] for v in own_voxels]
    ys = [v[1] for v in own_voxels]
    zs = [v[2] for v in own_voxels]
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_zlim(min(zs) - pad, max(zs) + pad)
    ax.set_xlabel("x", fontsize=8); ax.set_ylabel("y", fontsize=8)
    ax.set_zlabel("z", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=18, azim=-58)


def pick_three_examples(neurons, voxels) -> list[int]:
    """Pick three illustrative neurons: bulk EXC pyramidal, an
    interneuron (PV/SST/VIP), and a cell with the largest body."""
    by_owner = {}
    for (x, y, z, state, owner, role) in voxels:
        by_owner.setdefault(owner, []).append((x, y, z, state, role))
    pyr_candidates = [n for n in neurons
                      if n["role"] == 0 and n["polarity"] == 0
                      and n["body_voxels"] > 1
                      and 5 <= n["soma"][2] <= 40]
    inh_candidates = [n for n in neurons
                      if n["role"] == 0 and n["polarity"] != 0
                      and n["body_voxels"] > 1]
    big_candidates = sorted(neurons, key=lambda n: -n["body_voxels"])

    pick = []
    if pyr_candidates:
        pyr_candidates.sort(key=lambda n: -n["body_voxels"])
        pick.append(pyr_candidates[0]["id"])
    if inh_candidates:
        inh_candidates.sort(key=lambda n: -n["body_voxels"])
        pick.append(inh_candidates[0]["id"])
    for n in big_candidates:
        if n["id"] not in pick and n["body_voxels"] > 1:
            pick.append(n["id"])
            break
    while len(pick) < 3 and big_candidates:
        cand = big_candidates.pop(0)
        if cand["id"] not in pick:
            pick.append(cand["id"])
    return pick[:3]


def render_neuron(prefix: Path, neuron_id: int, out_path: Path) -> None:
    voxels = load_voxels(prefix)
    neurons = load_neurons(prefix)
    if neuron_id > 0:
        ids = [neuron_id]
    else:
        ids = pick_three_examples(neurons, voxels)

    fig = plt.figure(figsize=(5 * len(ids) + 1.5, 6.5))
    for i, nid in enumerate(ids):
        target = next((n for n in neurons if n["id"] == nid), None)
        if target is None:
            sys.exit(f"neuron id={nid} not found")
        own_voxels = [(x, y, z, state, role)
                      for (x, y, z, state, owner, role) in voxels
                      if owner == nid]
        if not own_voxels:
            sys.exit(f"neuron id={nid} has no voxels in dump")
        ax = fig.add_subplot(1, len(ids), i + 1, projection="3d")
        render_one(ax, target, own_voxels)

    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=SOMA_COLOUR, label="soma"),
               mpatches.Patch(color="#9ecae1", label="dendrite (role=0)"),
               mpatches.Patch(color="#fdae6b", label="axon (role=1)"),
               mpatches.Patch(color="#bdbdbd",
                              label="axon trunk (role=2 BLOCKED)")]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=9,
               frameon=False)
    fig.suptitle(
        "Brain model — extended neuron shapes "
        "(after Pack M v2 + Phase 1' + sprouting)",
        fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv):
    nid = int(argv[1]) if len(argv) > 1 else 0
    out = Path(argv[2]) if len(argv) > 2 else (
        REPO_ROOT / "docs" / "figures" / "08_extended_neuron_shape.png")
    if not (DUMP_PREFIX.parent / f"{DUMP_PREFIX.name}_voxels.csv").exists():
        run_session_and_dump()
    render_neuron(DUMP_PREFIX, nid, out)


if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)
