#!/usr/bin/env python3
"""
Multi-session "lifetime" harness. Drives snc_chat across many sessions,
each one a fresh `--load` invocation, and records per-session metrics.
The brain ages naturally: lifelong neurogenesis on each load, stage
transitions at sessions 30 / 60 / 100.

Default scenario: 15 sessions where each early session teaches one new
word and later sessions probe retention. After the run a CSV trajectory
and a single PNG summary are written.

Usage:
  python3 scripts/run_lifetime.py [num_sessions=15] [brain_path=lifetime_brain.snc]
"""

import csv
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
CHAT_BIN = REPO_ROOT / "build" / "snc_chat"

ALL_WORDS = [
    "mom", "dad", "baby", "ball", "dog", "cat",
    "hi", "bye", "yes", "no", "more", "stop",
]


def run_session(brain_path: Path, commands: list, fresh: bool = False) -> str:
    args = [str(CHAT_BIN), "--no-log"]
    if not fresh:
        args.extend(["--load", str(brain_path)])
    proc = subprocess.run(
        args,
        input="\n".join(commands) + "\n",
        capture_output=True,
        text=True,
        timeout=180,
    )
    return proc.stdout


SHOW_RE   = re.compile(r"\[show\] shown=(\S+)\s+said=(\S+)")
STATUS_RE = re.compile(r"\[status\] step=(\d+)\s+neurons=(\d+)\s+synapses=(\d+)\s+"
                       r"structural-blobs=(\d+)\s+bins=(\d+).*?"
                       r"grid=(\d+)x(\d+)x(\d+)")
# `[stage] session N (toddler)` is printed every load; the `-> X` form only
# fires on transitions, so prefer the steady-state form.
STAGE_RE  = re.compile(r"\[stage\] session \d+ \(([^)]+)\)")


def parse_metrics(out: str, session: int) -> dict:
    correct = 0
    total = 0
    word_results = {w: "?" for w in ALL_WORDS}
    for shown, said in SHOW_RE.findall(out):
        if shown in word_results:
            word_results[shown] = said
            total += 1
            if said == shown:
                correct += 1
    m = STATUS_RE.search(out)
    step      = int(m.group(1)) if m else 0
    neurons   = int(m.group(2)) if m else 0
    synapses  = int(m.group(3)) if m else 0
    blobs     = int(m.group(4)) if m else 0
    bins      = int(m.group(5)) if m else 0
    grid_x    = int(m.group(6)) if m else 0
    grid_y    = int(m.group(7)) if m else 0
    grid_z    = int(m.group(8)) if m else 0
    stage_match = STAGE_RE.search(out)
    stage = stage_match.group(1) if stage_match else "—"
    return {
        "session": session,
        "stage": stage,
        "step": step,
        "neurons": neurons,
        "synapses": synapses,
        "structural_blobs": blobs,
        "bins": bins,
        "show_correct": correct,
        "show_total": total,
        "show_acc": (correct / total) if total else 0.0,
        "grid_x": grid_x, "grid_y": grid_y, "grid_z": grid_z,
        "word_results": word_results,
    }


def session_commands(session: int, brain_path: Path) -> list:
    """Curriculum:
       - Session 1: bootstrap (babble + first 2 words)
       - Sessions 2..12: each teaches one new word + reviews all so far
       - Sessions 13+: pure review (all 12 words shown)
    """
    cmds = []
    if session == 1:
        cmds.append("babble 30")
        for w in ALL_WORDS[:2]:
            cmds.extend([f"teach {w}", "correct",
                          f"teach {w}", "correct"])
        taught = ALL_WORDS[:2]
    elif session <= 12:
        cmds.append("babble 8")
        new_word = ALL_WORDS[session - 1] if session - 1 < len(ALL_WORDS) else None
        if new_word and new_word not in ALL_WORDS[:session - 1]:
            cmds.extend([f"teach {new_word}", "correct",
                          f"teach {new_word}", "correct"])
        taught = ALL_WORDS[:session]
    else:
        # Pure review session: skip babble. Random-motor firing during
        # the no-new-teaching window adds noise that competes with
        # engram readout on subsequent shows; explicit imagery
        # rehearsal also hurt in measurement (each imagine triggers
        # sprouting / synaptogenesis that dilutes the engram readout).
        # The brain consolidates better when given silent space +
        # probes only.
        taught = ALL_WORDS

    # Probe: show every taught word.
    for w in taught:
        cmds.append(f"show {w}")
    cmds.extend(["status", f"save {brain_path}", "quit"])
    return cmds


def main(argv):
    n_sessions = int(argv[1]) if len(argv) > 1 else 15
    brain_path = Path(argv[2]) if len(argv) > 2 else Path("lifetime_brain.snc")

    # Clean slate.
    for ext in ("", ".meta"):
        p = Path(str(brain_path) + ext)
        if p.exists():
            p.unlink()

    metrics = []
    print(f"running {n_sessions} sessions on {brain_path}")
    for s in range(1, n_sessions + 1):
        cmds = session_commands(s, brain_path)
        out = run_session(brain_path, cmds, fresh=(s == 1))
        m = parse_metrics(out, s)
        metrics.append(m)
        print(f"  session {s:3d}  stage={m['stage']:>13s}  "
              f"grid={m['grid_x']}x{m['grid_y']}x{m['grid_z']}  "
              f"neurons={m['neurons']}  syns={m['synapses']}  "
              f"show={m['show_correct']}/{m['show_total']}  "
              f"acc={m['show_acc']*100:5.1f}%")

    # Save CSV. Per-word columns hold the literal `said` value for
    # each probe (or `?` if that word wasn't shown this session); a
    # word column equals the word's name when the brain got it right.
    csv_path = Path("lifetime_trajectory.csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "stage", "step", "neurons", "synapses",
                     "structural_blobs", "bins",
                     "grid_x", "grid_y", "grid_z",
                     "show_correct", "show_total", "show_acc",
                     *ALL_WORDS])
        for m in metrics:
            w.writerow([m["session"], m["stage"], m["step"], m["neurons"],
                         m["synapses"], m["structural_blobs"], m["bins"],
                         m["grid_x"], m["grid_y"], m["grid_z"],
                         m["show_correct"], m["show_total"],
                         f"{m['show_acc']:.4f}",
                         *(m["word_results"][w_] for w_ in ALL_WORDS)])
    print(f"wrote {csv_path}")

    # Plot summary.
    sessions = [m["session"] for m in metrics]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Lifetime trajectory ({n_sessions} sessions)")

    ax = axes[0][0]
    ax.plot(sessions, [m["show_acc"] * 100 for m in metrics],
            marker="o", color="tab:green")
    ax.set_title("Show accuracy (%)")
    ax.set_xlabel("session"); ax.set_ylabel("accuracy %")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(sessions, [m["neurons"] for m in metrics],
            marker="s", color="tab:blue", label="seeded")
    ax.plot(sessions, [m["structural_blobs"] for m in metrics],
            marker="^", color="tab:orange", label="structural")
    ax.set_title("Neuron count")
    ax.set_xlabel("session"); ax.set_ylabel("count")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.plot(sessions, [m["synapses"] for m in metrics],
            marker="d", color="tab:purple")
    ax.set_title("Total synapses")
    ax.set_xlabel("session"); ax.set_ylabel("count")
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    vol = [m["grid_x"] * m["grid_y"] * m["grid_z"] / 1000.0 for m in metrics]
    ax.plot(sessions, vol, marker="o", color="tab:red")
    ax.set_title("Brain volume (k voxels)")
    ax.set_xlabel("session"); ax.set_ylabel("k voxels")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = Path("lifetime_summary.png")
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")

    # Per-word retention heatmap. Rows = words in teaching order,
    # columns = sessions. Cell value: 1 = correct, 0 = wrong, NaN = not
    # probed this session (so it shows blank in the imshow).
    import numpy as np
    grid = np.full((len(ALL_WORDS), len(metrics)), np.nan)
    for j, m in enumerate(metrics):
        for i, word in enumerate(ALL_WORDS):
            said = m["word_results"][word]
            if said == "?":
                continue
            grid[i, j] = 1.0 if said == word else 0.0
    fig2, ax = plt.subplots(figsize=(max(8, len(metrics) * 0.4),
                                      0.4 * len(ALL_WORDS) + 1.5))
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#eeeeee")
    im = ax.imshow(np.ma.masked_invalid(grid), aspect="auto",
                   cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(ALL_WORDS)))
    ax.set_yticklabels(ALL_WORDS)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([str(m["session"]) for m in metrics])
    ax.set_xlabel("session"); ax.set_ylabel("word")
    ax.set_title("Per-word retention "
                 "(green=correct, red=wrong, grey=not probed)")
    plt.tight_layout()
    out2 = Path("lifetime_per_word.png")
    plt.savefig(out2, dpi=110, bbox_inches="tight")
    print(f"wrote {out2}")


if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)
