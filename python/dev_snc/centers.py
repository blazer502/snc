"""Specialized centers and the multicenter SNC graph (Phase 2; plan §5.2-5.3).

A *center* is a named pool of spiking neurons carrying its own modality, local
learning rule, timescale, and developmental age (plan §5.2). The multicenter
graph partitions neurons into centers and declares the inter-center *pathways*
along which cross-modal signals flow (plan §5.3 "inter-center connectivity as the
core model").

This is where the plan's step 3 -- "center metadata in the SNC graph
representation" -- lands at the developmental-layer level. Parallel to the
substrate's per-neuron role/sign/channel arrays (include/snc/snn_graph.hpp), we
add a per-neuron `center` id; parallel to nothing yet in the substrate, we add a
per-synapse pathway/`signal_type` tag. `to_graph_dict()` emits exactly the field
layout the substrate uses (pre/post/delays/role/channel + center), and
`write_graph_bin()` serialises it in the same 'SNCG' binary that snc_export
writes, so a multicenter graph round-trips through the substrate's own format.
"""
from dataclasses import dataclass

import numpy as np

# The eight center modalities of plan §5.2.
MODALITIES = ("visual", "auditory", "spatial", "motor",
              "language", "episodic", "semantic", "reward")

# Substrate role codes (mirror snc::GraphRole in include/snc/snn_graph.hpp).
ROLE_INTERNAL, ROLE_INPUT, ROLE_OUTPUT = 0, 1, 2

MAGIC = 0x534E4347  # 'SNCG' -- matches src/bench/snc_export.cpp


@dataclass
class Center:
    """One specialized learning center: a contiguous pool of neurons."""
    name: str
    modality: str
    size: int
    learning_rule: str = "hebbian"   # fixed | hebbian | reward_hebbian | predictive
    timescale: str = "fast"          # fast | medium | slow
    begin: int = 0                   # first neuron id (assigned by the graph)
    age: int = 0                     # developmental age = updates experienced

    def __post_init__(self):
        if self.modality not in MODALITIES:
            raise ValueError(f"unknown modality {self.modality!r}")

    @property
    def end(self) -> int:
        return self.begin + self.size

    @property
    def ids(self) -> range:
        return range(self.begin, self.end)


@dataclass
class Pathway:
    """A declared connection channel between two centers (or a center to itself).

    The actual synapses are grown/pruned by structural plasticity at runtime, so
    a Pathway is only the *type* of the connection, not its current edge set."""
    src: str
    dst: str
    signal_type: str = "excitatory"  # excitatory | inhibitory | modulatory | predictive
    plastic: bool = True             # may structural plasticity grow/prune it?

    @property
    def kind(self) -> str:
        return "intra" if self.src == self.dst else "inter"


class MulticenterGraph:
    """Neuron partition into centers plus the declared inter-center pathways.

    This is a lightweight *spec*: it owns topology metadata, not weights or spike
    state (those live in the runtime, mirroring the substrate's split between the
    immutable SNNGraph and its runtimes)."""

    def __init__(self):
        self.centers: dict = {}          # name -> Center
        self.pathways: list = []         # list[Pathway]
        self._order: list = []           # center names in id order
        self.N = 0

    # -- construction -----------------------------------------------------
    def add_center(self, center: Center) -> Center:
        if center.name in self.centers:
            raise ValueError(f"duplicate center {center.name!r}")
        center.begin = self.N
        self.N += center.size
        self.centers[center.name] = center
        self._order.append(center.name)
        return center

    def connect(self, src: str, dst: str, signal_type: str = "excitatory",
                plastic: bool = True) -> Pathway:
        for c in (src, dst):
            if c not in self.centers:
                raise ValueError(f"unknown center {c!r}")
        p = Pathway(src, dst, signal_type, plastic)
        self.pathways.append(p)
        return p

    # -- metadata views ---------------------------------------------------
    def center_of(self) -> np.ndarray:
        """Per-neuron center index (size N), in the graph's center order."""
        arr = np.full(self.N, -1, dtype=np.int32)
        for idx, name in enumerate(self._order):
            c = self.centers[name]
            arr[c.begin:c.end] = idx
        return arr

    def center_index(self, name: str) -> int:
        return self._order.index(name)

    def modality_of(self) -> np.ndarray:
        """Per-neuron modality index into MODALITIES (size N)."""
        arr = np.full(self.N, -1, dtype=np.int32)
        for name in self._order:
            c = self.centers[name]
            arr[c.begin:c.end] = MODALITIES.index(c.modality)
        return arr

    def describe(self) -> str:
        lines = [f"MulticenterGraph: N={self.N} neurons, "
                 f"{len(self.centers)} centers, {len(self.pathways)} pathways"]
        for name in self._order:
            c = self.centers[name]
            lines.append(f"  center {name:<9} [{c.modality:<8}] "
                         f"ids {c.begin}:{c.end} n={c.size} "
                         f"rule={c.learning_rule} ts={c.timescale} age={c.age}")
        for p in self.pathways:
            lines.append(f"  pathway {p.src} -> {p.dst} "
                         f"[{p.kind}, {p.signal_type}, plastic={p.plastic}]")
        return "\n".join(lines)

    # -- substrate round-trip --------------------------------------------
    def to_graph_dict(self, edges, delays=None) -> dict:
        """Emit the substrate field layout for a concrete edge set.

        `edges` is an (S, 2) int array of (pre, post) neuron ids (the currently
        instantiated synapses across all pathways). Roles/channels are best-effort
        (INTERNAL / -1); the point of the round-trip is that per-neuron `center`
        survives the substrate's own serialization."""
        edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
        S = edges.shape[0]
        if delays is None:
            delays = np.ones(S, dtype=np.int32)
        return {
            "N": self.N, "S": S, "n_in": 0, "n_out": 0,
            "pre": edges[:, 0].astype(np.int32),
            "post": edges[:, 1].astype(np.int32),
            "delays": np.asarray(delays, dtype=np.int32),
            "role": np.full(self.N, ROLE_INTERNAL, dtype=np.int32),
            "channel": np.full(self.N, -1, dtype=np.int32),
            "center": self.center_of(),
        }


def write_graph_bin(path: str, g: dict) -> None:
    """Serialize a graph dict in the substrate's 'SNCG' binary layout, extended
    with a trailing per-neuron center[N] block (see src/bench/snc_export.cpp and
    the backward-compatible reader in python/snc/graph.py)."""
    N, S = int(g["N"]), int(g["S"])
    center = g.get("center", np.full(N, -1, dtype=np.int32))
    with open(path, "wb") as f:
        np.array([MAGIC, N, S, int(g.get("n_in", 0)), int(g.get("n_out", 0))],
                 dtype="<i4").tofile(f)
        np.asarray(g["pre"], dtype="<i4").tofile(f)
        np.asarray(g["post"], dtype="<i4").tofile(f)
        np.asarray(g["delays"], dtype="<i4").tofile(f)
        np.asarray(g["role"], dtype="<i4").tofile(f)
        np.asarray(g["channel"], dtype="<i4").tofile(f)
        np.asarray(center, dtype="<i4").tofile(f)
