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
from .nursery import COLORS, SHAPES, Obj, feature_dim


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


# ---------------------------------------------------------------------------
# Task 3: embodied navigation -- fetch the named object
# ---------------------------------------------------------------------------
def _manhattan(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


def _train_motor(agent, rng, grid, max_steps, episodes):
    """Reward-modulated motor training on an empty grid: reach a random target
    cell, shaped by the change in Manhattan distance (shared by nav + permanence)."""
    from .navigation import ACTIONS
    from .nursery import Nursery
    for _ in range(episodes):
        world = Nursery(grid, grid)
        ax, ay = int(rng.integers(grid)), int(rng.integers(grid))
        world.place_agent(ax, ay)
        tx, ty = ax, ay
        while (tx, ty) == (ax, ay):
            tx, ty = int(rng.integers(grid)), int(rng.integers(grid))
        for _step in range(max_steps):
            code = agent.spatial.encode(world.ax, world.ay, tx, ty)
            a, pi = agent.motor.act(code)
            d0 = _manhattan(world.ax, world.ay, tx, ty)
            world.feet.move(ACTIONS[a])
            r = float(d0 - _manhattan(world.ax, world.ay, tx, ty))
            reached = (world.ax, world.ay) == (tx, ty)
            if reached:
                r += 5.0
            agent.motor.learn(code, a, pi, r)
            if reached:
                break


def run_navigation(cfg: AgentConfig, name_epochs: int = 40, motor_episodes: int = 1500,
                   fetch_trials: int = 300, data_seed: int = 0, cross_modal: bool = True,
                   grid: int = 7, n_objects: int = 4, max_steps: int = 30) -> dict:
    """Learn names, learn to walk (reward-modulated motor), then fetch a named
    object: recall identifies which object the word means, the spatial+motor loop
    walks to it. `cross_modal` toggles the language<->vision pathway; with it off,
    target identification is chance and fetching should fail even though the motor
    skill is intact (plan RQ5 -> embodied behavior)."""
    from .navigation import ACTIONS, NavAgent
    from .nursery import COLORS, SHAPES, Nursery, Obj

    nc, ns = len(COLORS), len(SHAPES)
    n_objects = min(n_objects, nc)                 # at most one object per distinct colour
    if n_objects + 1 > grid * grid:
        raise ValueError("grid too small to place objects and the agent on distinct cells")
    groups = [list(range(nc)), list(range(nc, nc + ns))]
    cfg = AgentConfig(**{**cfg.__dict__, "in_dim": feature_dim(), "pathway": cross_modal})
    agent = NavAgent(nc + ns, groups, cfg, seed=cfg.seed)
    rng = np.random.default_rng(data_seed)

    # 1) teach compositional names so color-word recall works.
    combos = _all_combos()

    def name_target(o):
        t = np.zeros(nc + ns, dtype=np.float32)
        t[o.color] = 1.0
        t[nc + o.shape] = 1.0
        return t

    for e in range(name_epochs):
        if e > 0:
            agent.consolidate()
        for i in rng.permutation(len(combos)):
            o = combos[i]
            agent.teach_name(o.features(), name_target(o))

    # 2) learn to walk: reward-modulated motor training on an empty grid.
    def rand_cell():
        return int(rng.integers(grid)), int(rng.integers(grid))

    _train_motor(agent, rng, grid, max_steps, motor_episodes)

    # 3a) motor-only navigation success (reach a given cell, empty grid), for the
    # trained greedy policy and, as a reproducible chance baseline, a uniform-random
    # policy under the identical protocol (same grid, step budget, early stop).
    def motor_eval(policy):
        ok = 0
        for _ in range(200):
            world = Nursery(grid, grid)
            ax, ay = rand_cell()
            world.place_agent(ax, ay)
            tx, ty = ax, ay
            while (tx, ty) == (ax, ay):
                tx, ty = rand_cell()
            for _step in range(max_steps):
                world.feet.move(ACTIONS[policy(world, tx, ty)])
                if (world.ax, world.ay) == (tx, ty):
                    break
            ok += (world.ax, world.ay) == (tx, ty)
        return ok / 200

    motor_success = motor_eval(agent.greedy_action)
    motor_random = motor_eval(lambda w, tx, ty: int(rng.integers(len(ACTIONS))))

    # 3b) fetch: identify the named object by recall, then navigate to it.
    f_ok = 0
    steps = 0
    for _ in range(fetch_trials):
        used = set()

        def free_cell():
            while True:
                c = rand_cell()
                if c not in used:
                    used.add(c)
                    return c

        cols = list(rng.permutation(nc))[:n_objects]
        objs = [Obj(oid=k, color=int(c), shape=int(rng.integers(ns)), x=(xy := free_cell())[0], y=xy[1])
                for k, c in enumerate(cols)]
        world = Nursery(grid, grid, objects=objs)
        world.place_agent(*free_cell())
        cmd_color = int(rng.choice(cols))
        correct = next(o for o in objs if o.color == cmd_color)
        identified = agent.identify(cmd_color, objs)   # recall (chance if pathway off)
        step = 0
        for step in range(max_steps):
            if _manhattan(world.ax, world.ay, identified.x, identified.y) <= 1:
                break
            world.feet.move(ACTIONS[agent.greedy_action(world, identified.x, identified.y)])
        steps += step + 1
        f_ok += _manhattan(world.ax, world.ay, correct.x, correct.y) <= 1

    return {
        "task": "navigation", "cross_modal": cross_modal, "n_objects": n_objects,
        "motor_success": motor_success, "motor_random": motor_random,
        "fetch_success": f_ok / fetch_trials,
        "avg_steps": steps / fetch_trials,
    }


# ---------------------------------------------------------------------------
# Task 4: object permanence -- search for a named object that is now hidden
# ---------------------------------------------------------------------------
def run_permanence(cfg: AgentConfig, name_epochs: int = 40, motor_episodes: int = 1500,
                   trials: int = 300, data_seed: int = 0, memory: bool = True,
                   grid: int = 7, n_objects: int = 4, max_steps: int = 30) -> dict:
    """See objects and name them, binding each name to its location in episodic
    memory; then the objects are hidden and the agent must *search* for a named
    one it can no longer see. With the memory center it recalls the location and
    walks there; lesioned, it can only guess among the seen locations (plan §7.3
    stage 2 object permanence)."""
    from .centers import Center
    from .navigation import ACTIONS, NavAgent
    from .memory import EpisodicMemory
    from .nursery import COLORS, SHAPES, Nursery, Obj

    nc, ns = len(COLORS), len(SHAPES)
    n_objects = min(n_objects, nc)
    if n_objects + 1 > grid * grid:
        raise ValueError("grid too small to place objects and the agent on distinct cells")
    groups = [list(range(nc)), list(range(nc, nc + ns))]
    cfg = AgentConfig(**{**cfg.__dict__, "in_dim": feature_dim()})
    agent = NavAgent(nc + ns, groups, cfg, seed=cfg.seed)
    # register the episodic memory as a fifth center in the graph metadata.
    agent.graph.add_center(Center("episodic", "episodic", 2 * grid, "hebbian", "slow"))
    agent.graph.connect("language", "episodic", "excitatory")
    agent.graph.connect("episodic", "spatial", "predictive")
    rng = np.random.default_rng(data_seed)

    combos = _all_combos()

    def name_target(o):
        t = np.zeros(nc + ns, dtype=np.float32)
        t[o.color] = 1.0
        t[nc + o.shape] = 1.0
        return t

    for e in range(name_epochs):
        if e > 0:
            agent.consolidate()
        for i in rng.permutation(len(combos)):
            o = combos[i]
            agent.teach_name(o.features(), name_target(o))

    _train_motor(agent, rng, grid, max_steps, motor_episodes)

    def rand_cell():
        return int(rng.integers(grid)), int(rng.integers(grid))

    ok = 0
    for _ in range(trials):
        used = set()

        def free_cell():
            while True:
                c = rand_cell()
                if c not in used:
                    used.add(c)
                    return c

        cols = list(rng.permutation(nc))[:n_objects]
        objs = [Obj(oid=k, color=int(c), shape=int(rng.integers(ns)), x=(xy := free_cell())[0], y=xy[1])
                for k, c in enumerate(cols)]
        start = free_cell()

        # observe through the body API: see the scene, name each object, and bind
        # name -> location in episodic memory. (scene() is a whole-scene view, so
        # the agent's pose does not affect what is observed.)
        obs_world = Nursery(grid, grid, objects=objs)
        mem = EpisodicMemory(nc, grid)
        for s in obs_world.eyes.scene():
            color_word = int(agent.lang.name(s["features"])[0])
            mem.write(color_word, s["x"], s["y"])

        # the objects are now hidden; a colour is commanded and the agent searches
        # an empty grid from memory alone.
        cmd_color = int(rng.choice(cols))
        target = next(o for o in objs if o.color == cmd_color)
        qx, qy = mem.query(cmd_color) if memory else mem.random_location(rng)

        world = Nursery(grid, grid)
        world.place_agent(*start)
        for _step in range(max_steps):
            if (world.ax, world.ay) == (qx, qy):
                break
            world.feet.move(ACTIONS[agent.greedy_action(world, qx, qy)])
        ok += (world.ax, world.ay) == (target.x, target.y)

    return {"task": "permanence", "memory": memory, "n_objects": n_objects,
            "search_success": ok / trials}
