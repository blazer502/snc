"""CLI for the Developmental Multicenter SNC v1 nursery experiments.

    python3 python/train_nursery.py                      # full comparison suite
    python3 python/train_nursery.py --seeds 4 --demo     # quick run + a narrated episode
    python3 python/train_nursery.py --selftest           # assertions only (fast)

See docs/nursery-v1.md for the experimental design and expected results.
"""
import argparse
import sys

from dev_snc.agent import AgentConfig, DevelopmentalAgent
from dev_snc.experiment import (format_navigation, format_tables,
                                run_navigation_suite, run_suite)
from dev_snc.tasks import run_naming


def demo():
    """Narrate one developmental agent: show its centers, teach it, watch a
    structural-plasticity step, then let it name and retrieve."""
    from dev_snc.nursery import COLORS, SHAPES, Nursery, Obj
    cfg = AgentConfig(seed=0)
    nc, ns = len(COLORS), len(SHAPES)
    agent = DevelopmentalAgent(nc + ns, [list(range(nc)), list(range(nc, nc + ns))], cfg)
    print(agent.graph.describe())
    print()

    world = Nursery(objects=[Obj(0, color=0, shape=1, x=1, y=0)])  # a red square
    o = world.by_id[0]
    import numpy as np
    tgt = np.zeros(nc + ns, dtype=np.float32); tgt[o.color] = 1; tgt[nc + o.shape] = 1
    for _ in range(60):
        agent.teach(o.features(), tgt)
    reports = agent.consolidate()
    for rep in reports:
        print(rep)
    pred = agent.name(o.features())
    print(f"\n  sees {o.name!r} -> names it "
          f"({COLORS[pred[0]]}, {SHAPES[pred[1] - nc]})")
    print(f"  hears 'red' -> picks object #{agent.retrieve(0, [o.features()])} of 1 candidate")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8, help="number of seeds to average")
    ap.add_argument("--naming-epochs", type=int, default=40)
    ap.add_argument("--forget-epochs", type=int, default=25)
    ap.add_argument("--nav-seeds", type=int, default=4, help="seeds for the navigation table (0 to skip)")
    ap.add_argument("--demo", action="store_true", help="also narrate one agent")
    ap.add_argument("--selftest", action="store_true", help="run assertions and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        from dev_snc.selftest import run_selftest
        return run_selftest()

    if args.demo:
        demo()
        print("\n" + "=" * 68 + "\n")

    results = run_suite(range(args.seeds), args.naming_epochs, args.forget_epochs)
    print(format_tables(results))
    if args.nav_seeds > 0:
        print()
        print(format_navigation(run_navigation_suite(range(args.nav_seeds))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
