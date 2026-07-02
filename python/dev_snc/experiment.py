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


def run_navigation_suite(seeds=range(4), name_epochs=40, motor_episodes=1500):
    """Fetch-the-named-object with the cross-modal pathway on vs. off (lesion)."""
    from .tasks import run_navigation
    out = {}
    for cm in (True, False):
        out[cm] = [run_navigation(AgentConfig(seed=s), data_seed=s, cross_modal=cm,
                                  name_epochs=name_epochs, motor_episodes=motor_episodes)
                   for s in seeds]
    return out


def format_navigation(nav) -> str:
    L = ["Embodied navigation  (fetch the named object, mean over seeds)"]
    L.append(f"  {'condition':<28} {'motor_success':>13} {'fetch_success':>13} {'avg_steps':>10}")
    for cm, label in [(True, "cross-modal on (recall)"), (False, "cross-modal off (lesion)")]:
        rows = nav[cm]
        ms, _ = _mean_std(rows, "motor_success")
        fs, _ = _mean_std(rows, "fetch_success")
        st, _ = _mean_std(rows, "avg_steps")
        L.append(f"  {label:<28} {ms:>13.2f} {fs:>13.2f} {st:>10.1f}")
    mr, _ = _mean_std(nav[True], "motor_random")
    L.append(f"  (motor skill learned by a reward-modulated three-factor rule; uniform-random "
             f"policy = {mr:.2f}. fetch needs recall.)")
    return "\n".join(L)


def run_permanence_suite(seeds=range(4), name_epochs=40, motor_episodes=1500):
    """Search for a hidden named object with the episodic memory on vs. lesioned."""
    from .tasks import run_permanence
    out = {}
    for mem in (True, False):
        out[mem] = [run_permanence(AgentConfig(seed=s), data_seed=s, memory=mem,
                                   name_epochs=name_epochs, motor_episodes=motor_episodes)
                    for s in seeds]
    return out


def format_permanence(perm) -> str:
    L = ["Object permanence  (search for a hidden named object, mean over seeds)"]
    L.append(f"  {'condition':<28} {'search_success':>15}")
    for mem, label in [(True, "episodic memory on"), (False, "memory lesioned")]:
        ss, _ = _mean_std(perm[mem], "search_success")
        L.append(f"  {label:<28} {ss:>15.2f}")
    n = perm[True][0]["n_objects"]
    L.append(f"  (memory binds name->location while observing; lesioned -> guess among "
             f"{n} seen sites = 1/{n})")
    return "\n".join(L)


def run_consolidation_suite(seeds=range(8), noise=0.3, episodes=12):
    """Sleep/replay consolidation of noisy episodic property observations."""
    from .tasks import run_consolidation
    return [run_consolidation(AgentConfig(seed=s), noise=noise, episodes=episodes, data_seed=s)
            for s in seeds]


def format_consolidation(rows) -> str:
    L = ["Sleep/replay consolidation  (property learning, mean over seeds)"]
    L.append(f"  {'condition':<34} {'seen types':>12} {'unseen combos':>14}")
    ss, _ = _mean_std(rows, "semantic_seen")
    sn, _ = _mean_std(rows, "semantic_novel")
    ms, _ = _mean_std(rows, "episodic_majority_seen")
    es, _ = _mean_std(rows, "episodic_single_seen")
    en, _ = _mean_std(rows, "episodic_novel")
    L.append(f"  {'sleep-consolidated semantic':<34} {ss:>12.2f} {sn:>14.2f}")
    L.append(f"  {'episodic, aggregate at query':<34} {ms:>12.2f} {en:>14.2f}")
    L.append(f"  {'episodic, single transient trace':<34} {es:>12.2f} {en:>14.2f}")
    noise = rows[0]["noise"]
    L.append(f"  (label noise p={noise}: single-trace ceiling ~= {1 - noise:.2f}, chance = 0.50. "
             f"aggregation denoises 'seen'; only the consolidated rule abstracts to 'unseen combos'.)")
    return "\n".join(L)


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
