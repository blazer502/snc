# Considered: 2-bit → 1-bit structural voxels

A design study of whether the `BrainGrid` structural matrix can drop from **2
bits/cell to 1 bit/cell**, recovering the cell *type* from a rule. Conclusion up
front: **keep 2 bits** — the change is a net loss at realistic scales and risks
the brain engine; this note records the analysis and the rule, in case grids
ever need to scale far enough to revisit it.

## Current encoding

`BrainGrid` packs 2 bits/cell, 32 cells per `uint64`, double-buffered, with four
states:

| code | state | meaning | used for |
|---|---|---|---|
| 0 | EMPTY | free space | sprouting target, volume exclusion |
| 1 | NEURON | soma / dendrite / axon tissue | growth, AXON×DENDRITE synaptogenesis |
| 2 | SYNAPSE | a synaptic-contact voxel | "already a synapse" check |
| 3 | BLOCKED | glia / scaffold / axon-trunk | **blocks** synapse formation |

All four are read in hot structural paths — e.g. synaptogenesis explicitly tests
"target is *not* BLOCKED and *not* already SYNAPSE" before forming a contact
(`simulator.cpp`), and growth/volume-exclusion count NEURON neighbours.

## The 1-bit proposal and its rule

Store only **occupancy** (1 bit, 64 cells/`uint64`) and recover the type:

```
bit == 0                      -> EMPTY
bit == 1 and v ∈ synapse_pos  -> SYNAPSE     (synapse_pos: set of SynapseEdge.pos)
bit == 1 and v ∈ blocked_set  -> BLOCKED     (blocked_set: glia/scaffold/trunk)
bit == 1 otherwise            -> NEURON      (covered by Neuron.body)
```

## Why it does not pay off

- **Memory savings are tiny and likely negative.** The grid is `X·Y·Z·2` bits
  (×2 buffers): 128³ ≈ 1 MB total, 512³ ≈ 67 MB. Halving saves 0.5 MB / 33 MB —
  while the *side structures* the rule needs cost more: `synapse_pos` for ~10⁵
  synapses is ~1–2 MB on its own, exceeding the 128³ grid saving. Net memory can
  *increase*. (Neuron/synapse state, not the grid, dominates the engine's memory.)
- **Hot paths get slower.** Type is needed per-voxel inside synaptogenesis and
  growth loops. A 2-bit grid answers in one `O(1)` masked read; the 1-bit rule
  needs a hash-set lookup per occupied voxel. Net compute loss.
- **BLOCKED has no other home.** SYNAPSE lives on `SynapseEdge.pos` and NEURON on
  `Neuron.body`, but BLOCKED (glia/scaffold/axon-trunk) is *only* in the grid —
  the rule would force a new `blocked_set` structure, re-adding the memory we
  tried to save.
- **Risk vs reward.** `BrainGrid` underpins the brain demos (100% baseline). A
  core encoding change for ≤1 MB at typical sizes is poor risk/reward.
- **Substrate relevance is low.** For the current SNN-substrate direction the
  grid is *legacy*: only `compile_from_simulator` reads it, and the runtime works
  on the compact CSR `SNNGraph`. Optimising the grid is off the critical path.

## Recommendation

- **Keep the 2-bit grid.** It is already compact (32 cells/word), `O(1)`-typed,
  and not a memory bottleneck.
- **If** very large grids (≥512³) ever make grid memory matter, the right move is
  a **sparse / active-voxel** representation (store only occupied voxels and their
  type, iterate active regions) — which also cuts per-step work — rather than
  1-bit-occupancy + type lookups.
- A genuine 1-bit grid is only clean if BLOCKED is **dropped** entirely (a model
  simplification that changes synaptogenesis and would need its own controlled
  experiment against the 100% baseline) — not recommended as a free swap.
