"""The developmental agent and its dense-pathway baseline (Phase 2-3; plan §13).

A DevelopmentalAgent is two specialized centers -- a visual center (a fixed sparse
spiking reservoir) and a language center (word neurons) -- joined by two separately
learnable inter-center pathways:

    visual --W_vl (naming: local delta rule over per-group softmax)--> language
    language --W_lv (recall: local Hebbian co-activation)--> visual

Both pathways are trained on every teaching episode from the *same* co-observation
(an object is seen and named at once). They are independent weight matrices --
naming is discriminative, recall is an unsupervised associative memory -- so the
agent grounds a word<->object association *bidirectionally* from shared
developmental experience, with no task-level (retrieval-labeled) supervision (the
plan's RQ5). This is learned bidirectional grounding, NOT free transfer from a
trained direction to an untrained one: both directions are trained.

The rules are two-factor in v1 (pre-activity x post-signal). Each update also
carries a `reward` factor, but it is held at 1.0 throughout v1 (supervised
teaching), so reward/neuromodulation is scaffolded, not yet exercised.

The same class expresses every baseline/ablation through AgentConfig (plan §9):

    structure = "modular" | "merged"     # sparse plastic pathways vs dense (all-to-all) pathways
    plasticity = "structural" | "weightonly"   # grow/prune the pathway vs fixed sparse mask
    pathway   = True | False             # cross-modal pathway present, or lesioned to chance

"merged" only makes the two pathway masks dense (all-to-all) -- a dense-connectivity
upper bound / capacity ceiling. The centers and the fixed sparse visual reservoir
are unchanged, so it is a dense-pathway ablation, not one homogeneous population.
"""
from dataclasses import dataclass

import numpy as np

from .centers import Center, MulticenterGraph
from .plasticity import SparseConn, VisualEncoder, softmax


@dataclass
class AgentConfig:
    structure: str = "modular"      # modular | merged
    plasticity: str = "structural"  # structural | weightonly
    pathway: bool = True
    v_size: int = 256               # visual center neurons
    v_active: int = 12              # k-winners that spike per object
    in_dim: int = 7                 # feature dim (color + shape one-hots)
    vl_budget: int = 200            # V->L synapses (sparse pathways)
    lv_budget: int = 200            # L->V synapses
    lr: float = 0.3                 # delta-rule learning rate (naming)
    lr_hebb: float = 0.2            # Hebbian rate (retrieval)
    grow: int = 60                  # synapses grown per consolidation
    prune: int = 60                 # synapses pruned per consolidation
    seed: int = 0


class DevelopmentalAgent:
    def __init__(self, n_words: int, groups, cfg: AgentConfig):
        """n_words: size of the language center.
        groups: list of index lists partitioning the word neurons into
                mutually-exclusive decision groups (e.g. [color_ids, shape_ids]);
                naming makes one choice per group."""
        self.cfg = cfg
        self.L = n_words
        self.groups = [list(g) for g in groups]
        self.rng = np.random.default_rng(cfg.seed)
        dense = (cfg.structure == "merged")
        structural = (cfg.plasticity == "structural")

        self.vis = VisualEncoder(cfg.in_dim, cfg.v_size, cfg.v_active, seed=cfg.seed)
        # naming: post = word (L), pre = visual (V)
        self.vl = SparseConn("visual->lang", cfg.v_size, self.L, cfg.vl_budget,
                             structural=structural, dense=dense, seed=cfg.seed + 1)
        # retrieval: post = visual (V), pre = word (L)
        self.lv = SparseConn("lang->visual", self.L, cfg.v_size, cfg.lv_budget,
                             structural=structural, dense=dense, seed=cfg.seed + 2)
        self._xv_mean = np.zeros(cfg.v_size, dtype=np.float32)
        self._seen = 0

        # Center metadata graph (plan step 3). visual ids [0,V), language ids [V,V+L).
        self.graph = MulticenterGraph()
        self.graph.add_center(Center("visual", "visual", cfg.v_size,
                                     learning_rule="fixed", timescale="fast"))
        self.graph.add_center(Center("language", "language", self.L,
                                     learning_rule="hebbian", timescale="medium"))
        if cfg.pathway:
            self.graph.connect("visual", "language", "excitatory")
            self.graph.connect("language", "visual", "predictive")

    # -- learning ---------------------------------------------------------
    def teach(self, features: np.ndarray, target: np.ndarray, reward: float = 1.0) -> None:
        """One supervised naming episode: see an object and hear its word(s).

        target: {0,1}^L with one active word per decision group. `reward` is the
        third (neuromodulatory) factor; v1 leaves it at 1.0 (see module docstring),
        so the rules below reduce to two-factor pre x post updates."""
        if not self.cfg.pathway:
            return
        xv = self.vis.encode(features)                 # sparse visual spikes
        target = np.asarray(target, dtype=np.float32)

        # Naming (V->L): local delta rule with per-group softmax (two-factor; reward=1).
        logits = self.vl.forward(xv)
        prob = np.zeros(self.L, dtype=np.float32)
        for g in self.groups:
            prob[g] = softmax(logits[g])
        err = target - prob                            # postsynaptic error signal
        dW = self.cfg.lr * reward * np.outer(err, xv)  # pre=xv, post=err, mod=reward
        self.vl.W += dW * self.vl.M * self.vl.gate     # gate freezes consolidated synapses
        self.vl.accumulate(np.abs(err) + target, xv)   # structural growth signal

        # Recall (L->V): local Hebbian toward the centered visual code (reward=1).
        self._seen += 1
        self._xv_mean += (xv - self._xv_mean) / self._seen
        centered = xv - self._xv_mean
        dH = self.cfg.lr_hebb * reward * np.outer(centered, target)  # post=visual, pre=word
        self.lv.W += dH * self.lv.M * self.lv.gate
        self.lv.accumulate(xv, target)

        self.graph.centers["language"].age += 1

    def consolidate(self):
        """Slow-clock structural step on the plastic pathways (plan §6.2 slow path)."""
        if not self.cfg.pathway:
            return []
        reports = [
            self.vl.structural_step(self.cfg.grow, self.cfg.prune),
            self.lv.structural_step(self.cfg.grow, self.cfg.prune),
        ]
        return reports

    # -- use --------------------------------------------------------------
    def name(self, features: np.ndarray) -> np.ndarray:
        """Name an object from vision alone: one chosen word index per group."""
        if not self.cfg.pathway:
            return np.array([int(self.rng.choice(g)) for g in self.groups])
        logits = self.vl.forward(self.vis.encode(features))
        return np.array([g[int(np.argmax(logits[g]))] for g in self.groups])

    def retrieve(self, word_index: int, candidates) -> int:
        """From a word alone, pick the matching object among candidate feature
        vectors -- language -> vision *recall*. The reverse (L->V) pathway is
        trained by co-activation on every teaching episode (not by retrieval-labeled
        examples), so this is associative recall, not transfer from the naming path."""
        if not self.cfg.pathway:
            return int(self.rng.integers(len(candidates)))
        pred = self.lv.W[:, word_index] * self.lv.M[:, word_index]   # expected visual code
        scores = [float(pred @ self.vis.encode(c)) for c in candidates]
        return int(np.argmax(scores))

    # -- inspection -------------------------------------------------------
    def export_edges(self) -> np.ndarray:
        """Current inter-center synapses as global (pre, post) ids for the
        substrate round-trip (visual ids [0,V), language ids [V, V+L))."""
        V = self.cfg.v_size
        edges = []
        post, pre = np.nonzero(self.vl.M)        # visual -> language
        edges.append(np.stack([pre, post + V], axis=1))
        post, pre = np.nonzero(self.lv.M)        # language -> visual
        edges.append(np.stack([pre + V, post], axis=1))
        return np.concatenate(edges, axis=0) if edges else np.zeros((0, 2), int)

    def synapse_count(self) -> int:
        return self.vl.num_synapses() + self.lv.num_synapses()
