"""Episodic memory center (plan §5.2 memory centers, §7.3 stage 2).

A fast, one-shot associative store binding a word to a location -- the agent's
record that "the red thing was at (2, 3)". It is written by Hebbian co-activation
the moment an object is seen and named, and read back by word. Because the binding
persists after the object is hidden, the agent can search for something it can no
longer perceive: **object permanence**.

Unlike the slow structural pathways (which learn across many episodes), episodic
memory is a fast within-episode store -- the plan's separation of timescales
(§6.2): the structural pathways are the slow developmental memory, this is the
fast event memory.
"""
import numpy as np


class EpisodicMemory:
    """Word -> location associative memory over a `grid` x `grid` world.

    The location code is one-hot(x) concatenated with one-hot(y); the memory is a
    (2*grid, n_words) Hebbian weight matrix. Writing a word at (x, y) adds that
    location code into the word's column; querying reads the column back and
    decodes x and y by argmax over each half."""

    def __init__(self, n_words: int, grid: int):
        self.n_words = n_words
        self.grid = grid
        self.W = np.zeros((2 * grid, n_words), dtype=np.float32)
        self.locations: list = []      # every observed (x, y), for the lesion baseline

    def _loc_code(self, x: int, y: int) -> np.ndarray:
        if not (0 <= x < self.grid and 0 <= y < self.grid):
            raise ValueError(f"location ({x}, {y}) out of bounds for grid {self.grid}")
        c = np.zeros(2 * self.grid, dtype=np.float32)
        c[x] = 1.0
        c[self.grid + y] = 1.0
        return c

    def write(self, word: int, x: int, y: int) -> None:
        """One-shot Hebbian bind of a word to a location."""
        self.W[:, word] += self._loc_code(x, y)
        self.locations.append((x, y))

    def query(self, word: int):
        """Recall the location bound to a word (argmax over each coordinate)."""
        col = self.W[:, word]
        return int(np.argmax(col[:self.grid])), int(np.argmax(col[self.grid:]))

    def random_location(self, rng):
        """A location the agent saw *some* object at, but without the word binding
        -- the memory-lesion baseline (knows objects existed, not which is which)."""
        return self.locations[int(rng.integers(len(self.locations)))]


class SemanticMemory:
    """Durable, consolidated knowledge (plan §5.2 semantic memory, §6.2 sleep).

    A Hebbian evidence vector over the visual code that predicts a binary property.
    It is not written during experience; it is built during a *sleep* phase by
    replaying the episodic traces (`consolidate`). Because replay accumulates
    evidence, two things emerge that transient episodic memory cannot provide:
    repeated noisy observations are averaged into a robust rule (denoising), and,
    since the readout is linear over the (attribute-structured) visual code, the
    rule generalizes to novel instances it never saw (abstraction)."""

    def __init__(self, code_size: int):
        self.W = np.zeros(code_size, dtype=np.float32)
        self.mask = None    # None = dense; a boolean keep-mask after prune()

    def consolidate(self, traces, replays: int = 1) -> None:
        """Replay episodic traces into semantic evidence. `traces` is a list of
        (visual_code, label) with label in {-1, +1}."""
        for _ in range(replays):
            for code, label in traces:
                self.W += label * code

    def prune(self, budget: int) -> None:
        """Structural consolidation (plan §6.3): keep only the `budget` most-
        evidenced synapses and prune the rest, compacting the consolidated rule
        into durable *sparse structure* on the substrate's slow clock -- rather
        than a dense readout. Evidence magnitude is the growth/keep signal."""
        keep = np.zeros(self.W.size, dtype=bool)
        if budget > 0:
            keep[np.argsort(np.abs(self.W))[self.W.size - budget:]] = True
        self.mask = keep

    def predict(self, code: np.ndarray) -> int:
        """Predict the binary property for a visual code (0/1)."""
        w = self.W if self.mask is None else self.W * self.mask
        return int((w @ code) >= 0)

    def num_synapses(self) -> int:
        return int(self.W.size if self.mask is None else self.mask.sum())
