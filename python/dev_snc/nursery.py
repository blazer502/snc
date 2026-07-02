"""Digital nursery: a deterministic 2D grid world + body API (Phase 1).

This is the embodied environment of the Developmental Multicenter SNC plan
(docs/developmental-multicenter-snc.md, §7 "Digital Nursery" and §13 "Minimal
Publishable Prototype"). It is intentionally small: a grid of attribute-bearing
objects and a thin *body API* -- eyes / feet / hands / speech / memory -- that a
developmental agent perceives and acts through. The purpose is not a game engine
but *enough embodiment for sensorimotor and cross-modal learning*.

Perception is object-centric for v1 (plan §12.4 "minimum useful"): the visual
signal of an object is a one-hot(color) + one-hot(shape) feature vector, not raw
pixels. That keeps the environment deterministic and the learning signal clean
while still forcing a *learned* mapping from a distributed visual code (produced
by the visual center) to language, which is where the cross-modal claim lives.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

COLORS = ("red", "green", "blue", "yellow")
SHAPES = ("circle", "square", "triangle")

# Compass directions: (dx, dy) with +y pointing "south" (row index grows down).
DIRS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
DIR_ORDER = ("N", "E", "S", "W")


def feature_dim() -> int:
    """Dimension of an object's visual feature vector (one-hot color+shape)."""
    return len(COLORS) + len(SHAPES)


@dataclass
class Obj:
    """An object in the nursery. Attributes drive perception and naming."""
    oid: int
    color: int          # index into COLORS
    shape: int          # index into SHAPES
    x: int = 0
    y: int = 0
    movable: bool = True

    def features(self) -> np.ndarray:
        """one-hot(color) concatenated with one-hot(shape)."""
        f = np.zeros(feature_dim(), dtype=np.float32)
        f[self.color] = 1.0
        f[len(COLORS) + self.shape] = 1.0
        return f

    @property
    def color_name(self) -> str:
        return COLORS[self.color]

    @property
    def shape_name(self) -> str:
        return SHAPES[self.shape]

    @property
    def name(self) -> str:
        return f"{self.color_name}_{self.shape_name}"


# --- Body API -------------------------------------------------------------
# Each sub-API is a thin view over the Nursery. They mirror plan §7.2 so the
# agent interacts through a body, not through privileged environment access.

class Eyes:
    def __init__(self, world: "Nursery"):
        self.w = world
        self.attended: Optional[int] = None

    def observe(self) -> Optional[dict]:
        """Object-centric observation of whatever occupies the agent's cell."""
        o = self.w.by_cell.get((self.w.ax, self.w.ay))
        return None if o is None else {"oid": o.oid, "features": o.features()}

    def look(self) -> Optional[dict]:
        """Observe the cell directly in front of the agent (facing dir)."""
        dx, dy = DIRS[self.w.facing]
        o = self.w.by_cell.get((self.w.ax + dx, self.w.ay + dy))
        return None if o is None else {"oid": o.oid, "features": o.features()}

    def focus(self, oid: int) -> Optional[dict]:
        """Attend to a specific object regardless of position (overt attention)."""
        self.attended = oid
        o = self.w.by_id.get(oid)
        return None if o is None else {"oid": oid, "features": o.features()}

    def scene(self) -> list:
        """All visible objects with positions (for the spatial center)."""
        return [{"oid": o.oid, "x": o.x, "y": o.y, "features": o.features()}
                for o in self.w.objects]


class Feet:
    def __init__(self, world: "Nursery"):
        self.w = world

    def move(self, direction: str) -> bool:
        """Step one cell in a compass direction. Returns True if the agent moved,
        False if blocked (out of bounds or an object occupies the target cell).
        Body-API convention: every action returns True on success."""
        dx, dy = DIRS[direction]
        nx, ny = self.w.ax + dx, self.w.ay + dy
        if not self.w.in_bounds(nx, ny) or (nx, ny) in self.w.by_cell:
            return False  # blocked
        self.w.ax, self.w.ay = nx, ny
        return True

    def forward(self) -> bool:
        return self.move(self.w.facing)

    def turn(self, direction: str) -> None:
        self.w.facing = direction


class Hands:
    def __init__(self, world: "Nursery"):
        self.w = world

    def push(self, oid: int) -> bool:
        """Push an object one cell along the agent's facing direction. Succeeds
        only if the object is in front, movable, and the target cell is free."""
        o = self.w.by_id.get(oid)
        if o is None or not o.movable:
            return False
        dx, dy = DIRS[self.w.facing]
        if (o.x, o.y) != (self.w.ax + dx, self.w.ay + dy):
            return False  # not in front of the agent
        tx, ty = o.x + dx, o.y + dy
        if not self.w.in_bounds(tx, ty) or (tx, ty) in self.w.by_cell:
            return False
        if self.w.by_cell.get((o.x, o.y)) is not o:
            return False  # stale/duplicate occupant; don't corrupt the grid
        del self.w.by_cell[(o.x, o.y)]
        o.x, o.y = tx, ty
        self.w.by_cell[(tx, ty)] = o
        return True


class Speech:
    """Agent's spoken output channel. The environment only records the last
    utterance; scoring lives in the task (the teacher grades the answer)."""
    def __init__(self, world: "Nursery"):
        self.w = world
        self.last_answer = None

    def answer(self, words) -> None:
        self.last_answer = tuple(words) if isinstance(words, (list, tuple)) else (words,)

    def ask(self, question: str) -> None:
        self.w.pending_question = question


class Memory:
    """A trivial associative store standing in for the plan's memory API. The
    episodic/semantic *centers* live in the agent; this is just the body-level
    remember/recall handle so the API surface is complete."""
    def __init__(self, world: "Nursery"):
        self.w = world
        self._store: dict = {}

    def remember(self, key, value) -> None:
        self._store[key] = value

    def recall(self, key):
        return self._store.get(key)


class Teacher:
    """The nursery's caregiver: names objects, issues commands, gives reward.

    Teaching signals are compositional -- an object is named by its color word
    AND its shape word -- so the language center can, in principle, generalize
    to unseen color x shape combinations (plan RQ5, §13 task 4)."""
    def __init__(self, world: "Nursery"):
        self.w = world

    def present(self, oid: int) -> dict:
        """Return the supervised teaching signal for an object: its visual
        features plus the color/shape words the caregiver speaks."""
        o = self.w.by_id[oid]
        return {"oid": oid, "features": o.features(),
                "color_word": o.color, "shape_word": o.shape, "name": o.name}

    @staticmethod
    def reward(correct: bool) -> float:
        return 1.0 if correct else -1.0


class Nursery:
    """A small grid world holding objects, an agent pose, and the body API."""

    def __init__(self, width: int = 5, height: int = 5, objects=None, seed: int = 0):
        self.W, self.H = width, height
        self.rng = np.random.default_rng(seed)
        self.objects = list(objects) if objects else []
        self.by_id = {o.oid: o for o in self.objects}
        self.by_cell = {}
        for o in self.objects:
            if (o.x, o.y) in self.by_cell:
                raise ValueError(f"two objects share cell {(o.x, o.y)}")
            self.by_cell[(o.x, o.y)] = o
        self.ax, self.ay = 0, 0
        self.facing = "E"
        self.pending_question = None
        # body API
        self.eyes = Eyes(self)
        self.feet = Feet(self)
        self.hands = Hands(self)
        self.speech = Speech(self)
        self.memory = Memory(self)
        self.teacher = Teacher(self)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.W and 0 <= y < self.H

    def add(self, o: Obj) -> Obj:
        if (o.x, o.y) in self.by_cell:
            raise ValueError(f"cell {(o.x, o.y)} already occupied")
        self.objects.append(o)
        self.by_id[o.oid] = o
        self.by_cell[(o.x, o.y)] = o
        return o

    def place_agent(self, x: int, y: int, facing: str = "E") -> None:
        self.ax, self.ay = x, y
        self.facing = facing
