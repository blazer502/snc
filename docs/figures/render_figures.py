#!/usr/bin/env python3
"""
Render documentation figures for the SNC project.

Outputs go into docs/figures/. Each figure is a self-contained PNG
explaining one aspect of the computing model or the brain model.
Re-run to regenerate after changes.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
})


# --------------------------------------------------------------------------
# Figure 1 — 2-bit structural grid, packed words, owner map
# --------------------------------------------------------------------------
def fig_structural_grid():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: a 6x4 slice of cells, color-coded
    ax = axes[0]
    rng = np.random.default_rng(7)
    # 0 EMPTY (white), 1 NEURON (blue), 2 SYNAPSE (orange), 3 BLOCKED (grey)
    cells = rng.choice([0, 0, 0, 1, 1, 2, 3], size=(4, 6))
    cmap = {
        0: ("#ffffff", "·"),
        1: ("#9ecae1", "N"),
        2: ("#fdae6b", "S"),
        3: ("#bdbdbd", "X"),
    }
    for r in range(4):
        for c in range(6):
            color, label = cmap[int(cells[r, c])]
            ax.add_patch(plt.Rectangle((c, 3 - r), 1, 1,
                                        facecolor=color, edgecolor="#444"))
            ax.text(c + 0.5, 3 - r + 0.5, label, ha="center", va="center",
                    fontsize=11, color="#222")
    ax.set_xlim(0, 6); ax.set_ylim(0, 4); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Structural grid — 2 bits per voxel")
    legend_handles = [
        mpatches.Patch(facecolor="#ffffff", edgecolor="#444", label="0  EMPTY"),
        mpatches.Patch(facecolor="#9ecae1", edgecolor="#444", label="1  NEURON"),
        mpatches.Patch(facecolor="#fdae6b", edgecolor="#444", label="2  SYNAPSE"),
        mpatches.Patch(facecolor="#bdbdbd", edgecolor="#444", label="3  BLOCKED"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)

    # Right: 32 cells packed into one uint64
    ax = axes[1]
    ax.set_title("32 cells packed into one uint64")
    for i in range(32):
        v = i % 4  # cycle 0..3 just for illustration
        color = cmap[v][0]
        ax.add_patch(plt.Rectangle((i, 0), 1, 1,
                                    facecolor=color, edgecolor="#444"))
        ax.text(i + 0.5, 0.5, f"{v:02b}", ha="center", va="center",
                fontsize=7, color="#222")
    ax.text(0, 1.4, "bit 0..1", fontsize=9, color="#666")
    ax.text(31, 1.4, "bit 62..63", fontsize=9, color="#666", ha="right")
    ax.text(16, -0.6, "uint64 word",
            fontsize=10, color="#444", ha="center")
    ax.set_xlim(-0.5, 32.5); ax.set_ylim(-1.0, 2.0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["bottom"].set_visible(False); ax.spines["left"].set_visible(False)

    fig.suptitle("Computing model — structural matrix",
                 fontsize=12, fontweight="bold")
    fig.savefig(OUT / "01_structural_grid.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2 — One simulation step pipeline
# --------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    phases = [
        ("integrate\nincoming", "queue 1\n→ branch sums", "#cde6f5"),
        ("chemistry", "potential\n→ fire", "#cde6f5"),
        ("STDP", "LTP / LTD\n+ eligibility", "#fbe5b8"),
        ("hetero-\nsynaptic", "PSD damp", "#fbe5b8"),
        ("fire\ndispatch", "→ queue 4\n(energy gated)", "#cde6f5"),
        ("scheduler\ndispatch", "queue 4 → 1\n+ LTD on deliv.", "#cde6f5"),
        ("homeo-\nstatic", "Σw → target", "#fbe5b8"),
        ("sprouting", "extend NEURON\nvoxels", "#d6eecd"),
        ("synapto-\ngenesis", "NEURON → SYNAPSE\n(per-region cap)", "#d6eecd"),
        ("pruning", "spine\nretraction", "#d6eecd"),
        ("energy\nregen", "regions\n+ fire cost", "#cde6f5"),
    ]
    n = len(phases)
    box_w = 1.0; gap = 0.12
    x = 0
    for label, sub, color in phases:
        ax.add_patch(plt.Rectangle((x, 0), box_w, 1.0,
                                    facecolor=color, edgecolor="#333"))
        ax.text(x + box_w / 2, 0.65, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold")
        ax.text(x + box_w / 2, 0.32, sub, ha="center", va="center",
                fontsize=7.5, color="#333")
        if x + box_w + gap < n * (box_w + gap):
            ax.annotate("", xy=(x + box_w + gap - 0.02, 0.5),
                         xytext=(x + box_w + 0.02, 0.5),
                         arrowprops=dict(arrowstyle="->", color="#444",
                                         lw=1.0))
        x += box_w + gap

    # Legend bands
    ax.text(0, -0.55, "● parallel (OpenMP)", color="#3a5a78",
            fontsize=8.5)
    ax.text(2.5, -0.55, "● learning rules", color="#a07020",
            fontsize=8.5)
    ax.text(5.0, -0.55, "● structural mutation (serial)",
            color="#3a7050", fontsize=8.5)
    ax.set_xlim(-0.2, x); ax.set_ylim(-1.0, 1.4); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("One simulation step", fontsize=12, fontweight="bold")
    fig.savefig(OUT / "02_pipeline.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3 — Synaptic transmission queues 1..4
# --------------------------------------------------------------------------
def fig_queues():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    nodes = [
        (1.0, 2.0, "PRE\nsoma"),
        (4.0, 2.0, "AXON\nin transit"),
        (7.0, 2.0, "SYNAPSE\nvoxel"),
        (10.0, 2.0, "POST\nbranch + soma"),
    ]
    for (x, y, label) in nodes:
        ax.add_patch(plt.Circle((x, y), 0.7, facecolor="#cde6f5",
                                 edgecolor="#333"))
        ax.text(x, y, label, ha="center", va="center", fontsize=9)
    arrows = [
        (1.7, 2.0, 3.3, 2.0, "Q4\nfire dispatch\n(energy gated)"),
        (4.7, 2.0, 6.3, 2.0, "transit\nqueue\n(conduction\ndelay)"),
        (7.7, 2.0, 9.3, 2.0, "Q1\nincoming\nbranch[b]"),
    ]
    for (x0, y0, x1, y1, label) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="-|>", lw=1.5,
                                     color="#446"))
        ax.text((x0 + x1) / 2, y1 + 1.05, label, ha="center", va="bottom",
                fontsize=8.5, color="#333")

    # STDP & vesicle annotations
    ax.text(7, 0.6, "STDP fires here:\nLTP if pre→post,\nLTD on delivery",
            ha="center", va="top", fontsize=8.5, color="#a35",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fde7f0",
                      edgecolor="#a35"))
    ax.text(4, 0.6, "vesicle pool\n(Tsodyks-Markram\nshort-term plasticity)",
            ha="center", va="top", fontsize=8.5, color="#357",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#e7f0fd",
                      edgecolor="#357"))

    ax.set_xlim(-0.5, 11); ax.set_ylim(-0.5, 4); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Synaptic transmission — four explicit queue stages",
                 fontsize=12, fontweight="bold")
    fig.savefig(OUT / "03_queues.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 4 — Brain anatomy layout (toddler 64x64x48)
# --------------------------------------------------------------------------
def fig_anatomy():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Top-down (x, y) layout at the cortical floor / mid-depth.
    # 20-word brain: 20 rows of 4 label INPUTs at z=2 (stride 3 in y),
    # 4x4 image INPUTs at z=4, 2x10 motors at z=Z-3 (stride 6 in x).
    ax = axes[0]
    ax.set_title("Top-down layout (toddler 64×64, 20-word brain)\n"
                 "INPUTs at z=2 / 4, motors at z=Z-3")
    n_classes = 20
    # ext_in: 20 rows × 4 cols at y=3..60 stride 3 (z=2)
    for c in range(n_classes):
        for f in range(4):
            x = 3 + f * 4; y = 3 + c * 3
            ax.add_patch(plt.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                                        facecolor="#9ecae1",
                                        edgecolor="#3182bd", lw=0.5))
    # image_in: 4x4 at x=32..47, y=4..19 stride 5 (z=4)
    for r in range(4):
        for c in range(4):
            x = 32 + c * 5; y = 4 + r * 5
            ax.add_patch(plt.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                                        facecolor="#a1d99b",
                                        edgecolor="#31a354", lw=0.5))
    # motors: 2 rows × 10 cols at xm=4..58 stride 6, ym=Y/2-8 / +8
    for c in range(n_classes):
        col = c % 10; row = c // 10
        xm = 4 + col * 6; ym = (32 - 8) + row * 16
        ax.add_patch(plt.Circle((xm, ym), 0.8, facecolor="#fdae6b",
                                 edgecolor="#e6550d", lw=0.7))
        ax.text(xm + 1.2, ym, "M", fontsize=6, color="#a04020")
    # niches: radius 9 around each motor
    for c in range(n_classes):
        col = c % 10; row = c // 10
        xm = 4 + col * 6; ym = (32 - 8) + row * 16
        ax.add_patch(plt.Circle((xm, ym), 9, facecolor="none",
                                 edgecolor="#e6550d", lw=0.4,
                                 linestyle="--", alpha=0.4))
    ax.set_xlim(0, 64); ax.set_ylim(0, 64); ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(alpha=0.15)
    legend_handles = [
        mpatches.Patch(facecolor="#9ecae1", edgecolor="#3182bd",
                       label="80 label INPUTs (z=2)"),
        mpatches.Patch(facecolor="#a1d99b", edgecolor="#31a354",
                       label="16 image INPUTs (z=4)"),
        mpatches.Patch(facecolor="#fdae6b", edgecolor="#e6550d",
                       label="20 motor OUTPUTs (z=Z-3)"),
        mpatches.Patch(facecolor="none", edgecolor="#e6550d",
                       linestyle="--",
                       label="engram niches (r=9)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    # Cross-section: z stack
    ax = axes[1]
    ax.set_title("Z stack (cortical depth)")
    layers = [
        (0, 1.8, "fetal seed VZ + brainstem +\nthalamic relay + aversive",
         "#e7e1c8"),
        (1.8, 3.2, "label INPUTs (z=2)", "#9ecae1"),
        (3.2, 5.0, "image INPUTs (z=4)", "#a1d99b"),
        (5.0, 41.0, "cortical bulk\n(sprouting / synaptogenesis)",
         "#fafafa"),
        (41.0, 44.0, "self-perception INPUT (z=Z-4)", "#c6dbef"),
        (44.0, 46.0, "motor OUTPUTs (z=Z-3)", "#fdae6b"),
    ]
    for (z0, z1, label, color) in layers:
        ax.add_patch(plt.Rectangle((0, z0), 1, z1 - z0,
                                    facecolor=color, edgecolor="#333"))
        ax.text(1.05, (z0 + z1) / 2, label, va="center", fontsize=8.5)
    ax.set_xlim(0, 4); ax.set_ylim(48, 0)  # invert: z=0 at top
    ax.set_aspect("auto")
    ax.set_xticks([]); ax.set_ylabel("z (cortical depth)")
    ax.spines["top"].set_visible(True)

    fig.suptitle("Brain model — chat vocabulary anatomy",
                 fontsize=12, fontweight="bold")
    fig.savefig(OUT / "04_anatomy.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 5 — Developmental volume scaling
# --------------------------------------------------------------------------
def fig_developmental():
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    stages = [
        ("toddler",        (0, 30),    (64, 64, 48),  "#fee0b6"),
        ("early-child",    (30, 60),   (96, 96, 64),  "#fdae6b"),
        ("middle-child",   (60, 100),  (112, 112, 80),"#f16913"),
        ("preadolescent",  (100, 140), (128, 128, 96),"#a63603"),
    ]
    for (name, (s0, s1), (X, Y, Z), color) in stages:
        vol = X * Y * Z / 1000.0
        ax.add_patch(plt.Rectangle((s0, 0), s1 - s0, vol,
                                    facecolor=color,
                                    edgecolor="#333", alpha=0.85))
        ax.text((s0 + s1) / 2, vol + 60, name, ha="center", fontsize=9)
        ax.text((s0 + s1) / 2, vol / 2,
                f"{X}×{Y}×{Z}\n{vol:.0f}k voxels",
                ha="center", va="center", fontsize=8.5, color="#222")
    # Real human brain volume at each stage (rough, ml).
    real = [(15, 200), (45, 590), (80, 1000), (120, 1500)]
    rs, rv = zip(*real)
    ax2 = ax.twinx()
    ax2.plot(rs, rv, "o-", color="#08519c", lw=1.5, label="real human cortex (ml)")
    ax2.set_ylabel("real cortex volume (ml)")
    ax2.set_ylim(0, 1700)
    ax.set_xlabel("session (developmental time)")
    ax.set_ylabel("simulator grid volume (k voxels)")
    ax.set_xlim(0, 140); ax.set_ylim(0, 1700)
    ax.set_title("Developmental scaling — anchored to pre-adolescent peak",
                 fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right")
    fig.savefig(OUT / "05_developmental.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 6 — Engram allocation pipeline
# --------------------------------------------------------------------------
def fig_engram():
    fig, ax = plt.subplots(figsize=(11, 5.2))

    boxes = [
        (0.0, 2.4, 2.0, 1.2, "INTERNAL\nneurons",
         "candidate pool", "#cde6f5"),
        (3.0, 5.0, 3.0, 0.9,
         "score = fire_rate_ema × excitability_bias",
         "Pack 25 (CREB)", "#fbe5b8"),
        (3.0, 3.8, 3.0, 0.9,
         "× 0.5 same-session / × 0.1 cross-session",
         "Pack 25 (memory linking)", "#fbe5b8"),
        (3.0, 2.6, 3.0, 0.9,
         "× niche (2.0 in, 0.25 far) — floor 0.75 if bias>1.5",
         "Pack 23 + Pack 25.1", "#fbe5b8"),
        (7.0, 3.5, 2.2, 1.2,
         "top-K +\npersistent\nmembers",
         "Pack 20", "#cde6f5"),
        (10.0, 3.5, 2.4, 1.2,
         "permanent +\ntagged\n(silent? skip floor)",
         "Pack 19A / 25", "#d6eecd"),
    ]
    for (x, y, w, h, label, sub, color) in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h,
                                    facecolor=color, edgecolor="#333"))
        ax.text(x + w / 2, y + h * 0.6, label, ha="center", va="center",
                fontsize=8.5)
        ax.text(x + w / 2, y + h * 0.2, sub, ha="center", va="center",
                fontsize=7.5, color="#555", fontstyle="italic")

    arrows = [
        (2.0, 3.0, 3.0, 5.45),
        (2.0, 3.0, 3.0, 4.25),
        (2.0, 3.0, 3.0, 3.05),
        (6.0, 4.1, 7.0, 4.1),
        (9.2, 4.1, 10.0, 4.1),
    ]
    for (x0, y0, x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="->", color="#444"))

    ax.set_xlim(-0.3, 12.6); ax.set_ylim(2.0, 6.4); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Engram allocation — promote_engram pipeline",
                 fontsize=12, fontweight="bold")
    fig.savefig(OUT / "06_engram.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 7 — Neuron morphology stamps (Pack M v2 + Phase 1 + Phase 1')
# --------------------------------------------------------------------------
def fig_neuron_morphology():
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.5))
    cell_types = [
        ("Pyramidal (excitatory)",
         # (dx, dy, dz, role)  role 0 = DENDRITE, 1 = AXON
         [(0, 0,  0, "soma"),
          (0, 0,  1, "DENDRITE"),   # apical
          (1, 0,  0, "DENDRITE"),   # basal +x
          (-1, 0, 0, "DENDRITE"),   # basal -x
          (0, 1,  0, "DENDRITE"),   # basal +y
          (0,-1,  0, "DENDRITE"),   # basal -y
          (0, 0, -1, "AXON")],      # descending
         "DeFelipe et al. 2013 — apical + 4 basal dendrites + descending axon"),
        ("PV basket (inhibitory, perisomatic)",
         [(0, 0, 0, "soma"),
          ( 1, 0, 0, "AXON"),
          (-1, 0, 0, "AXON"),
          ( 0, 1, 0, "AXON"),
          ( 0,-1, 0, "AXON"),
          ( 0, 0, 1, "DENDRITE"),
          ( 0, 0,-1, "DENDRITE")],
         "Tremblay 2016 — dense local axonal arbor (perisomatic)"),
        ("SST Martinotti (inhibitory, dendritic)",
         [(0, 0, 0, "soma"),
          ( 0, 0, 1, "AXON"),
          ( 0, 0, 2, "AXON"),
          ( 1, 0, 0, "DENDRITE"),
          (-1, 0, 0, "DENDRITE")],
         "Tremblay 2016 — ascending axon to layer 1"),
        ("VIP (inhibitory, disinhibitory)",
         [(0, 0, 0, "soma"),
          ( 0,  1, 0, "AXON"),
          ( 0, -1, 0, "AXON"),
          ( 1, 0, 0, "DENDRITE"),
          (-1, 0, 0, "DENDRITE")],
         "Tremblay 2016 — local axon to other inhibitories"),
    ]
    role_colours = {
        "soma":     ("#444444", "S"),
        "DENDRITE": ("#9ecae1", "D"),
        "AXON":     ("#fdae6b", "A"),
    }

    def render_cell(ax, voxels, title, sub):
        # Use an x-z slice to show the morphology: most templates lie
        # in this plane (apical / axon along z; lateral processes
        # along x). y-only voxels are rendered as a small ring atop
        # the soma.
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.text(0.5, -0.18, sub, transform=ax.transAxes, ha="center",
                fontsize=8.5, color="#555", fontstyle="italic")
        # Plot in (dx, dz) plane.
        for (dx, dy, dz, role) in voxels:
            color, label = role_colours[role]
            if dy != 0 and dx == 0 and dz == 0:
                # y-only voxel: draw an outlined ring at soma position
                ax.add_patch(plt.Rectangle((dx - 0.4, dz - 0.4),
                                            0.8, 0.8,
                                            facecolor="none",
                                            edgecolor=color, lw=2,
                                            linestyle=":"))
                ax.text(dx, dz, label, ha="center", va="center",
                        fontsize=8, color=color)
            else:
                ax.add_patch(plt.Rectangle((dx - 0.45, dz - 0.45),
                                            0.9, 0.9,
                                            facecolor=color,
                                            edgecolor="#333", lw=0.7))
                ax.text(dx, dz, label, ha="center", va="center",
                        fontsize=10, color="white" if role != "DENDRITE"
                                                  else "#222",
                        fontweight="bold")
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_xlabel("dx (lateral)")
        ax.set_ylabel("dz (cortical depth)")
        ax.grid(alpha=0.2)
        ax.axhline(0, color="#aaa", lw=0.5)
        ax.axvline(0, color="#aaa", lw=0.5)

    render_cell(axes[0][0], cell_types[0][1], cell_types[0][0],
                cell_types[0][2])
    render_cell(axes[0][1], cell_types[1][1], cell_types[1][0],
                cell_types[1][2])
    render_cell(axes[1][0], cell_types[2][1], cell_types[2][0],
                cell_types[2][2])
    render_cell(axes[1][1], cell_types[3][1], cell_types[3][0],
                cell_types[3][2])

    legend_handles = [
        mpatches.Patch(facecolor="#444444", edgecolor="#333",
                       label="S — soma (DENDRITE-default)"),
        mpatches.Patch(facecolor="#9ecae1", edgecolor="#333",
                       label="D — dendrite voxel (receives synapses)"),
        mpatches.Patch(facecolor="#fdae6b", edgecolor="#333",
                       label="A — axon voxel (initiates synapses)"),
        mpatches.Patch(facecolor="none", edgecolor="#666",
                       linestyle=":",
                       label="dotted = y-axis voxel (out of x-z slice)"),
    ]
    fig.suptitle("Brain model — neuron morphology stamps "
                 "(Pack M v2 + Phase 1 + Phase 1')",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=9,
               frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    fig.savefig(OUT / "07_neuron_morphology.png", dpi=140)
    plt.close(fig)


def main():
    fig_structural_grid()
    fig_pipeline()
    fig_queues()
    fig_anatomy()
    fig_developmental()
    fig_engram()
    fig_neuron_morphology()
    print(f"wrote 7 figures into {OUT}")


if __name__ == "__main__":
    main()
