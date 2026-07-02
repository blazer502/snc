"""The core comparison (Phase 4; plan §9 baselines/ablations, §13 comparison).

Runs the proposed system against its ablations over several seeds and prints two
tables -- cross-modal grounding and continual learning -- with synapse counts, so
the *connectivity budget* is visible next to accuracy. Fully deterministic given
the seed list.

    proposed     : modular + structural  -- separate centers, grow/prune + consolidate
    weight-only  : modular + weightonly   -- same centers/start budget, fixed mask (no topology)
    matched-final: weight-only at ~the structural system's *final* budget (isolates
                   adaptivity from raw synapse count)
    dense        : dense (all-to-all) pathways -- a dense-connectivity upper bound
    pathway-off  : cross-modal lesion       -- no inter-center connectivity (chance floor)

`dense` densifies only the pathway masks; the centers and the fixed sparse visual
reservoir are unchanged, so it is a dense-pathway ceiling, not a merged monolith.
"""
import numpy as np

from .agent import AgentConfig
from .tasks import run_forgetting, run_naming

CONDITIONS = [
    ("proposed  (modular+structural)", dict(structure="modular", plasticity="structural", pathway=True)),
    ("weight-only (same start, no topo)", dict(structure="modular", plasticity="weightonly", pathway=True)),
    ("weight-only (matched final ~310)", dict(structure="modular", plasticity="weightonly", pathway=True,
                                              vl_budget=155, lv_budget=155)),
    ("dense pathways (ceiling)",         dict(structure="merged",  plasticity="structural", pathway=True)),
    ("pathway-off (lesion)",             dict(structure="modular", plasticity="structural", pathway=False)),
]


def _mean_std(rows, key):
    a = np.array([r[key] for r in rows], dtype=float)
    return np.nanmean(a), np.nanstd(a)


def run_suite(seeds=range(8), naming_epochs=40, forget_epochs=25):
    results = []
    for label, kw in CONDITIONS:
        naming = [run_naming(AgentConfig(seed=s, **kw), epochs=naming_epochs, data_seed=s)
                  for s in seeds]
        forget = [run_forgetting(AgentConfig(seed=s, **kw), epochs=forget_epochs, data_seed=s)
                  for s in seeds]
        results.append({"label": label, "naming": naming, "forget": forget})
    return results


def format_tables(results) -> str:
    L = []
    L.append("Cross-modal grounding  (naming task, mean over seeds)")
    L.append(f"  {'condition':<34} {'naming':>7} {'gen':>6} {'recall':>7} {'synapses':>9}")
    for r in results:
        tr, _ = _mean_std(r["naming"], "train_acc")
        gn, _ = _mean_std(r["naming"], "gen_acc")
        rt, _ = _mean_std(r["naming"], "retrieval_acc")
        sy, _ = _mean_std(r["naming"], "synapses")
        L.append(f"  {r['label']:<34} {tr:>7.2f} {gn:>6.2f} {rt:>7.2f} {int(sy):>9}")
    L.append("  (recall is associative -- the reverse pathway is trained by co-activation)")
    L.append("")
    L.append("Continual learning  (class-incremental, mean over seeds)")
    L.append(f"  {'condition':<34} {'retention':>11} {'all_acc':>8} {'synapses':>9}")
    for r in results:
        rt, rs = _mean_std(r["forget"], "early_retention")
        aa, _ = _mean_std(r["forget"], "all_acc")
        sy, _ = _mean_std(r["forget"], "synapses")
        L.append(f"  {r['label']:<34} {rt:>7.2f}+-{rs:.2f} {aa:>8.2f} {int(sy):>9}")
    return "\n".join(L)
