"""Deterministic self-tests for the Developmental Multicenter SNC v1.

Every check is seeded, so results are reproducible. Run with:

    python3 python/train_nursery.py --selftest
"""
import os
import tempfile

import numpy as np

from .agent import AgentConfig, DevelopmentalAgent
from .centers import write_graph_bin
from .nursery import Nursery, Obj
from .tasks import run_forgetting, run_naming, run_navigation


def _check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def _env_determinism():
    a = Nursery(objects=[Obj(0, color=1, shape=2, x=1, y=0)], seed=0)
    b = Nursery(objects=[Obj(0, color=1, shape=2, x=1, y=0)], seed=0)
    _check("env: same object -> identical features",
           np.array_equal(a.by_id[0].features(), b.by_id[0].features()))
    # body API (convention: True on success). move into an occupied cell fails;
    # push relocates an object.
    w = Nursery(width=4, height=1, objects=[Obj(0, color=0, shape=0, x=1, y=0)])
    w.place_agent(0, 0, facing="E")
    moved = w.feet.move("E")      # into (1,0) which is occupied -> blocked
    _check("env: move is blocked by an occupied cell", (not moved) and (w.ax, w.ay) == (0, 0))
    _check("env: eyes see the object in front", w.eyes.look()["oid"] == 0)
    pushed = w.hands.push(0)      # push object from (1,0) to (2,0)
    _check("env: hands.push relocates a movable object", pushed and w.by_id[0].x == 2)


def _graph_roundtrip():
    ag = DevelopmentalAgent(7, [[0, 1, 2, 3], [4, 5, 6]], AgentConfig(seed=0, v_size=32))
    gd = ag.graph.to_graph_dict(ag.export_edges())
    from snc.graph import load_graph
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "mc.bin")
        write_graph_bin(p, gd)
        g2 = load_graph(p)
    _check("graph: per-neuron center survives the substrate binary round-trip",
           np.array_equal(g2["center"], ag.graph.center_of()))
    _check("graph: center partition sizes are right (32 visual + 7 language)",
           int((g2["center"] == 0).sum()) == 32 and int((g2["center"] == 1).sum()) == 7)


def _learning_and_transfer():
    prop = run_naming(AgentConfig(structure="modular", plasticity="structural", seed=0), data_seed=0)
    off = run_naming(AgentConfig(structure="modular", plasticity="structural",
                                 pathway=False, seed=0), data_seed=0)
    _check("learn: modular+structural names trained objects (train_acc == 1.0)",
           prop["train_acc"] == 1.0)
    _check("recall: word->object recall works from co-activation training (>=0.9)",
           prop["retrieval_acc"] >= 0.9)
    _check("lesion: removing the cross-modal pathway floors naming (< 0.4)",
           off["train_acc"] < 0.4)


def _structural_efficiency():
    s = run_forgetting(AgentConfig(plasticity="structural", seed=0), data_seed=0)
    w = run_forgetting(AgentConfig(plasticity="weightonly", seed=0), data_seed=0)
    m = run_forgetting(AgentConfig(structure="merged", seed=0), data_seed=0)
    _check("continual: structural retains classes (all_acc >= 0.8)", s["all_acc"] >= 0.8)
    _check("budget: structural uses fewer synapses than weight-only and dense",
           s["synapses"] < w["synapses"] < m["synapses"])
    _check("budget: dense pathways cost >3x the structural synapses",
           m["synapses"] > 3 * s["synapses"])

    # structural growth actually happens and is logged.
    ag = DevelopmentalAgent(7, [[0, 1, 2, 3], [4, 5, 6]], AgentConfig(seed=0))
    tgt = np.zeros(7, dtype=np.float32); tgt[0] = 1; tgt[4] = 1
    for _ in range(30):
        ag.teach(Obj(0, 0, 0).features(), tgt)
    reports = ag.consolidate()
    _check("structural: consolidation grows synapses and logs a StructReport",
           len(reports) == 2 and sum(r.grown for r in reports) > 0)


def _navigation():
    on = run_navigation(AgentConfig(seed=0), data_seed=0, cross_modal=True,
                        motor_episodes=1200, fetch_trials=200)
    off = run_navigation(AgentConfig(seed=0), data_seed=0, cross_modal=False,
                         motor_episodes=1200, fetch_trials=200)
    _check("motor: reward-modulated three-factor rule learns to navigate (>=0.9)",
           on["motor_success"] >= 0.9)
    _check("fetch: cross-modal recall lets the agent fetch the named object (>=0.7)",
           on["fetch_success"] >= 0.7)
    _check("fetch: lesioning the cross-modal pathway drops fetching toward chance",
           off["fetch_success"] < 0.45 and on["fetch_success"] > off["fetch_success"] + 0.3)


def run_selftest() -> int:
    print("Developmental Multicenter SNC v1 -- self-test")
    for section in (_env_determinism, _graph_roundtrip,
                    _learning_and_transfer, _structural_efficiency, _navigation):
        print(f"\n{section.__name__[1:]}:")
        section()
    print("\nAll self-tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_selftest())
