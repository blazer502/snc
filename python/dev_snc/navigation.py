"""Embodied navigation: spatial + motor centers (plan §5.2, §7.3 stages 1&4).

Two more centers turn the v1 agent from a namer into an actor:

* SpatialCenter -- encodes the target's direction relative to the agent as a
  sparse spike code (which way is the goal). Fixed, like the visual reservoir.

* MotorCenter -- a policy over the four move actions, trained by a
  reward-modulated three-factor rule: the local eligibility (action-taken minus
  policy) x spatial activity, gated by (reward - running baseline). This is
  REINFORCE-with-baseline expressed as a three-factor synaptic update -- the
  plan's motor center (reinforcement learning, local eligibility traces, motor
  babbling -> skill). Unlike v1's naming rule (reward held at 1.0), here reward is
  a real varying third factor.

NavAgent composes the v1 language recall pathway with these centers so the agent
can *fetch a named object*: recall says which object the word means, the
spatial+motor loop walks there. This is cross-modal grounding driving embodied
behavior (plan RQ5 -> action).
"""
import numpy as np

from .agent import AgentConfig, DevelopmentalAgent
from .centers import Center, MulticenterGraph
from .nursery import DIR_ORDER
from .plasticity import softmax

# Motor action index -> compass move, aligned with nursery.DIRS.
ACTIONS = DIR_ORDER  # ("N", "E", "S", "W")


class SpatialCenter:
    """Relative-direction code: one spiking neuron per (sign dx, sign dy) cell."""
    size = 9

    @staticmethod
    def encode(ax: int, ay: int, tx: int, ty: int) -> np.ndarray:
        sx = int(np.sign(tx - ax))
        sy = int(np.sign(ty - ay))
        c = np.zeros(SpatialCenter.size, dtype=np.float32)
        c[(sx + 1) * 3 + (sy + 1)] = 1.0
        return c


class MotorCenter:
    """Policy over move actions, trained by a reward-modulated three-factor rule."""

    def __init__(self, n_actions: int, spatial_dim: int, lr: float = 0.2,
                 baseline_lr: float = 0.01, seed: int = 0):
        self.n = n_actions
        self.W = np.zeros((n_actions, spatial_dim), dtype=np.float32)
        self.lr = lr
        self.baseline_lr = baseline_lr
        self.baseline = 0.0
        self.rng = np.random.default_rng(seed)

    def policy(self, code: np.ndarray) -> np.ndarray:
        return softmax(self.W @ code)

    def act(self, code: np.ndarray):
        """Sample an action from the policy; returns (action, policy). Greedy
        (evaluation) action selection is done by NavAgent.greedy_action."""
        pi = self.policy(code)
        return int(self.rng.choice(self.n, p=pi)), pi

    def learn(self, code: np.ndarray, action: int, pi: np.ndarray, reward: float) -> None:
        """Three-factor update: post = (action indicator - policy), pre = spatial
        code, modulator = (reward - baseline). REINFORCE-with-baseline, local."""
        onehot = np.zeros(self.n, dtype=np.float32)
        onehot[action] = 1.0
        elig = (onehot - pi)[:, None] * code[None, :]
        self.W += self.lr * (reward - self.baseline) * elig
        self.baseline += self.baseline_lr * (reward - self.baseline)


class NavAgent:
    """Language recall (v1) + spatial + motor centers for fetching named objects."""

    def __init__(self, n_words: int, groups, cfg: AgentConfig, lr_motor: float = 0.2,
                 seed: int = 0):
        self.lang = DevelopmentalAgent(n_words, groups, cfg)
        self.spatial = SpatialCenter()
        self.motor = MotorCenter(len(ACTIONS), self.spatial.size, lr=lr_motor, seed=seed + 7)

        # A four-center graph (plan §5.2): the metadata now spans perception,
        # language, space, and action.
        self.graph = MulticenterGraph()
        self.graph.add_center(Center("visual", "visual", cfg.v_size, "fixed", "fast"))
        self.graph.add_center(Center("language", "language", n_words, "hebbian", "medium"))
        self.graph.add_center(Center("spatial", "spatial", self.spatial.size, "fixed", "fast"))
        self.graph.add_center(Center("motor", "motor", len(ACTIONS), "reward_modulated", "fast"))
        if cfg.pathway:
            self.graph.connect("visual", "language", "excitatory")
            self.graph.connect("language", "visual", "predictive")
        self.graph.connect("spatial", "motor", "excitatory")

    # -- language (delegate to v1) ----------------------------------------
    def teach_name(self, features, target, reward: float = 1.0) -> None:
        self.lang.teach(features, target, reward)

    def consolidate(self):
        return self.lang.consolidate()

    def identify(self, word_index: int, objects):
        """Pick which object a word refers to, via language->vision recall."""
        idx = self.lang.retrieve(word_index, [o.features() for o in objects])
        return objects[idx]

    # -- motor navigation -------------------------------------------------
    def greedy_action(self, world, tx: int, ty: int) -> int:
        """Best policy action toward (tx,ty) that actually moves (skips blocked
        moves so a greedy agent never wedges against a wall/object)."""
        code = self.spatial.encode(world.ax, world.ay, tx, ty)
        order = np.argsort(self.motor.W @ code)[::-1]
        for a in order:
            dx, dy = _delta(int(a))
            nx, ny = world.ax + dx, world.ay + dy
            if world.in_bounds(nx, ny) and (nx, ny) not in world.by_cell:
                return int(a)
        return int(order[0])


def _delta(action: int):
    from .nursery import DIRS
    return DIRS[ACTIONS[action]]
