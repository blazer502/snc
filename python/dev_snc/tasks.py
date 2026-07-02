"""Developmental tasks and metrics (Phase 4; plan §7.3-7.4, §13 tasks).

Two curricula run over the digital nursery, each returning a metrics dict:

* naming  -- learn compositional object names (color word + shape word) from a
  subset of color x shape combinations, then measure (a) naming accuracy on the
  trained combos, (b) *generalization* to held-out combos, and (c) word->object
  *recall* (say a word, pick the matching object). Both directions are trained on
  every episode from the same co-observation; recall reads the reverse pathway
  (Hebbian co-activation), so it is associative recall, not transfer from the
  naming pathway. (b) is the compositional cross-modal signal (plan RQ5, §13 1&4).

* forgetting -- introduce objects a few classes at a time (class-incremental),
  then re-test the earlier classes. Retention of earlier classes is the continual
  learning signal (plan §13 task 5).

Data is built from a fixed seed independent of the agent, so every baseline and
ablation sees exactly the same objects and curriculum.
"""
import numpy as np

from .agent import AgentConfig, DevelopmentalAgent
from .nursery import COLORS, SHAPES, Nursery, Obj, feature_dim


def _all_combos():
    """One object per (color, shape) combination."""
    objs = []
    oid = 0
    for c in range(len(COLORS)):
        for s in range(len(SHAPES)):
            objs.append(Obj(oid=oid, color=c, shape=s))
            oid += 1
    return objs


def _split_covering(objs, holdout_frac, rng):
    """Hold out a fraction of combos while guaranteeing every color and every
    shape still appears in the training set (else its word can't be learned)."""
    nc, ns = len(COLORS), len(SHAPES)
    n_hold = max(1, int(round(holdout_frac * len(objs))))
    for _ in range(200):
        perm = rng.permutation(len(objs))
        hold = set(perm[:n_hold].tolist())
        train = [o for i, o in enumerate(objs) if i not in hold]
        if len({o.color for o in train}) == nc and len({o.shape for o in train}) == ns:
            test = [o for i, o in enumerate(objs) if i in hold]
            return train, test
    raise RuntimeError("could not build a covering split")


# ---------------------------------------------------------------------------
# Task 1: compositional naming + generalization + reverse retrieval
# ---------------------------------------------------------------------------
def run_naming(cfg: AgentConfig, epochs: int = 40, data_seed: int = 0,
               holdout_frac: float = 0.3) -> dict:
    nc, ns = len(COLORS), len(SHAPES)
    groups = [list(range(nc)), list(range(nc, nc + ns))]   # color block, shape block
    rng = np.random.default_rng(data_seed)
    train, test = _split_covering(_all_combos(), holdout_frac, rng)

    cfg = _with_in_dim(cfg)
    agent = DevelopmentalAgent(n_words=nc + ns, groups=groups, cfg=cfg)

    def target(o):
        t = np.zeros(nc + ns, dtype=np.float32)
        t[o.color] = 1.0
        t[nc + o.shape] = 1.0
        return t

    # Two-timescale: structural step between epochs, ending on weight-training so
    # the reported weights are settled (not a trailing structural reset).
    grown = pruned = 0
    for e in range(epochs):
        if e > 0:
            for r in agent.consolidate():
                grown += r.grown
                pruned += r.pruned
        for i in rng.permutation(len(train)):
            o = train[i]
            agent.teach(o.features(), target(o))

    def naming_acc(objs):
        if not objs:
            return float("nan")
        ok = 0
        for o in objs:
            pred = agent.name(o.features())               # [color_word, shape_word]
            ok += int(pred[0] == o.color and pred[1] == nc + o.shape)
        return ok / len(objs)

    # word->object recall: from a color/shape word, pick the matching object.
    def retrieval_acc():
        allobjs = train + test
        ok = tot = 0
        for c in range(nc):                               # color words
            cands = [next(o for o in allobjs if o.color == cc) for cc in range(nc)]
            ok += int(agent.retrieve(c, [o.features() for o in cands]) == c)
            tot += 1
        for s in range(ns):                               # shape words
            cands = [next(o for o in allobjs if o.shape == ss) for ss in range(ns)]
            ok += int(agent.retrieve(nc + s, [o.features() for o in cands]) == s)
            tot += 1
        return ok / tot

    return {
        "task": "naming", "structure": cfg.structure, "plasticity": cfg.plasticity,
        "pathway": cfg.pathway,
        "train_acc": naming_acc(train),
        "gen_acc": naming_acc(test),
        "retrieval_acc": retrieval_acc(),
        "synapses": agent.synapse_count(), "grown": grown, "pruned": pruned,
    }


# ---------------------------------------------------------------------------
# Task 2: continual (class-incremental) learning / catastrophic forgetting
# ---------------------------------------------------------------------------
def run_forgetting(cfg: AgentConfig, epochs: int = 25, data_seed: int = 0,
                   phases: int = 6) -> dict:
    """Introduce the objects a few classes at a time (class-incremental), then
    re-test the earlier classes. `early_retention` is end-of-training accuracy on
    every class introduced before the final phase (measured over many objects, so
    low variance); `all_acc` is accuracy over all classes (plan §13 task 5)."""
    objs = _all_combos()
    L = len(objs)                                         # one atomic label per object
    groups = [list(range(L))]
    rng = np.random.default_rng(data_seed)
    perm = rng.permutation(L)
    # np.array_split keeps every object even when L is not divisible by phases.
    phase_objs = [[objs[i] for i in idx] for idx in np.array_split(perm, phases)]

    cfg = _with_in_dim(cfg)
    agent = DevelopmentalAgent(n_words=L, groups=groups, cfg=cfg)

    def target(o):
        t = np.zeros(L, dtype=np.float32)
        t[o.oid] = 1.0
        return t

    def acc(group):
        return float(np.mean([agent.name(o.features())[0] == o.oid for o in group])) \
            if group else float("nan")

    for p in range(phases):
        for _ in range(epochs):
            for i in rng.permutation(len(phase_objs[p])):
                o = phase_objs[p][i]
                agent.teach(o.features(), target(o))
            agent.consolidate()           # structural step after each epoch

    early = [o for p in range(phases - 1) for o in phase_objs[p]]  # all but the last phase
    return {
        "task": "forgetting", "structure": cfg.structure, "plasticity": cfg.plasticity,
        "pathway": cfg.pathway,
        "early_retention": acc(early), "all_acc": acc(objs),
        "synapses": agent.synapse_count(),
    }


def _with_in_dim(cfg: AgentConfig) -> AgentConfig:
    if cfg.in_dim != feature_dim():
        cfg = AgentConfig(**{**cfg.__dict__, "in_dim": feature_dim()})
    return cfg
