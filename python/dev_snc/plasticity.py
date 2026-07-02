"""Center dynamics, local learning, and structural plasticity (Phase 3; plan §6).

Three ingredients, matching the plan's three timescales (§6.2):

* VisualEncoder -- the *fast* path. A fixed random projection + k-winners-take-all
  gives a sparse spiking code of an object's features. The representation is not
  trained in v1: the plan's thesis is that intelligence lives in the connectivity
  *among* centers, not inside a single learned blob, so the visual code is a fixed
  reservoir and all learning happens on the inter-center pathways.

* SparseConn -- the *medium* path. A masked linear map between two centers whose
  weights change by a local rule (pre-activity x post-signal, plus a reward factor
  the agent holds at 1.0 in v1, so two-factor in practice) -- the same family as the
  repo's e-prop / three-factor learning (docs/three-factor-reward.md). The rule is
  applied by the agent; this class owns the weights, the structural mask, the
  consolidation state, and the co-activation statistics.

* structural_step -- the *slow* path. Between learning it prunes the weakest
  synapses and grows new ones toward co-active neuron pairs (DeepR-style rewiring,
  mirroring src/structure/connectome.cpp). The budget is NOT pinned: growth is
  capped by how many co-active unconnected pairs exist, so once the useful ones are
  connected pruning outpaces growth and the pathway settles to a sparse,
  task-sustained floor. That emergent pruning is what makes connectivity itself a
  learning result (plan §6.3, RQ4).
"""
from dataclasses import dataclass

import numpy as np


def kwta(drive: np.ndarray, k: int) -> np.ndarray:
    """k-winners-take-all: exactly min(k, #positive) entries spike (1.0), rest 0.

    A stand-in for the lateral inhibition that keeps a center's code sparse. Only
    positive drives can win; ties are broken deterministically by index, so the
    result always has at most k ones."""
    x = np.zeros_like(drive)
    if k <= 0:
        return x
    pos = np.flatnonzero(drive > 0)
    if pos.size <= k:
        x[pos] = 1.0
        return x
    order = pos[np.argsort(drive[pos], kind="stable")]   # ascending, stable tie-break
    x[order[-k:]] = 1.0
    return x


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class VisualEncoder:
    """Fixed random sparse spiking code of an object's feature vector."""

    def __init__(self, in_dim: int, size: int, active: int, seed: int = 0):
        self.size = size
        self.active = active
        rng = np.random.default_rng(seed)
        # Fixed projection; each column normalized so every feature contributes.
        self.P = rng.standard_normal((size, in_dim)).astype(np.float32)
        self.P /= np.linalg.norm(self.P, axis=0, keepdims=True) + 1e-8

    def encode(self, features: np.ndarray) -> np.ndarray:
        return kwta(self.P @ features.astype(np.float32), self.active)


@dataclass
class StructReport:
    """One structural-plasticity step's outcome (mirrors snc::StructReport)."""
    pathway: str
    grown: int
    pruned: int
    synapses: int

    def __str__(self) -> str:
        return (f"[struct] {self.pathway:<10} grew={self.grown:<4} "
                f"pruned={self.pruned:<4} synapses={self.synapses}")


class SparseConn:
    """A masked (post x pre) connection with local weights + a structural mask.

    `structural=False` freezes the mask (weight-only learning); `dense=True`
    connects everything and never sparsifies (the merged/homogeneous baseline)."""

    def __init__(self, name: str, n_pre: int, n_post: int, budget: int,
                 structural: bool = True, dense: bool = False, seed: int = 0,
                 protect: int = 2, consolidate_thresh: float = 0.6,
                 consol_gate: float = 0.05):
        self.name = name
        self.n_pre, self.n_post = n_pre, n_post
        self.structural = structural and not dense
        self.dense = dense
        self.protect = protect              # rounds a new synapse is prune-immune
        self.consolidate_thresh = consolidate_thresh
        self.consol_gate = consol_gate      # plasticity left on a consolidated synapse
        self.rng = np.random.default_rng(seed)
        self.W = np.zeros((n_post, n_pre), dtype=np.float32)
        if dense:
            self.M = np.ones((n_post, n_pre), dtype=bool)
        else:
            self.M = self._random_mask(budget)
        # Structural age per synapse; the initial mask starts mature (prunable).
        self.age = self.M.astype(np.int32) * protect
        # Consolidation: once a synapse's weight matures it is protected from
        # pruning AND its plasticity is gated down, so later learning cannot erode
        # it (plan §6.2-6.3 consolidation; SI/EWC-like). `gate` multiplies the
        # local weight update the agent applies.
        self.consolidated = np.zeros((n_post, n_pre), dtype=bool)
        self.gate = np.ones((n_post, n_pre), dtype=np.float32)
        # Co-activation accumulator (post x pre), decayed each structural step.
        self.coact = np.zeros((n_post, n_pre), dtype=np.float32)
        self.budget = int(self.M.sum())

    def _random_mask(self, budget: int) -> np.ndarray:
        total = self.n_pre * self.n_post
        budget = min(budget, total)
        idx = self.rng.choice(total, size=budget, replace=False)
        m = np.zeros(total, dtype=bool)
        m[idx] = True
        return m.reshape(self.n_post, self.n_pre)

    def eff(self) -> np.ndarray:
        """Effective weights (masked)."""
        return self.W * self.M

    def forward(self, pre: np.ndarray) -> np.ndarray:
        return self.eff() @ pre

    def accumulate(self, post_act: np.ndarray, pre_act: np.ndarray) -> None:
        """Record a co-activation event for structural growth (Hebbian outer product)."""
        self.coact += np.outer(post_act, pre_act)

    def num_synapses(self) -> int:
        return int(self.M.sum())

    def structural_step(self, grow: int, prune: int, w_init: float = 0.01,
                        decay: float = 0.5) -> StructReport:
        """Prune the weakest connected synapses, then grow toward the most
        co-active *still*-unconnected pairs. Growth is capped by the number of
        co-active candidates, so the budget is not fixed (see the module note);
        just-pruned edges are excluded from growth so grown/pruned count genuinely
        distinct synapses."""
        if not self.structural:
            self.coact *= decay
            return StructReport(self.name, 0, 0, self.num_synapses())

        grown = pruned = 0
        victims = None
        # Consolidate: synapses whose weight has matured become protected -- immune
        # to pruning and largely frozen (gate down) so later tasks cannot erode them.
        newly = self.M & (~self.consolidated) & (np.abs(self.W) >= self.consolidate_thresh)
        self.consolidated |= newly
        self.gate[newly] = self.consol_gate

        # Prune: weakest |W| among connected synapses old enough to be eligible and
        # not consolidated (young synapses are protected so they can mature first).
        if prune > 0:
            eligible = np.argwhere(self.M & (self.age >= self.protect) & (~self.consolidated))
            if len(eligible):
                mag = np.abs(self.W[eligible[:, 0], eligible[:, 1]])
                victims = eligible[np.argsort(mag)[:prune]]
                self.M[victims[:, 0], victims[:, 1]] = False
                self.W[victims[:, 0], victims[:, 1]] = 0.0
                self.age[victims[:, 0], victims[:, 1]] = 0
                pruned = len(victims)
        # Grow: highest co-activation among pairs that are unconnected and were
        # not just pruned this step.
        if grow > 0:
            cand = self.coact.copy()
            cand[self.M] = -np.inf          # only unconnected pairs
            if victims is not None:
                cand[victims[:, 0], victims[:, 1]] = -np.inf  # don't regrow just-pruned
            flat = cand.ravel()
            n_pos = int((flat > 0).sum())
            take = min(grow, n_pos)
            if take > 0:
                sel = np.argpartition(flat, -take)[-take:]
                sel = sel[flat[sel] > 0]
                rows, cols = np.unravel_index(sel, self.W.shape)
                self.M[rows, cols] = True
                self.W[rows, cols] = w_init * self.rng.standard_normal(len(rows)).astype(np.float32)
                self.age[rows, cols] = 0
                grown = len(rows)

        self.age[self.M] += 1
        self.coact *= decay
        return StructReport(self.name, grown, pruned, self.num_synapses())
