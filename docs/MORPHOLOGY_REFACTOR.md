# Morphology refactor — what to invert when Pack M lands

Every shipped pack between Pack 19A and Pack 25.1 was built on top of the
**point-soma + isotropic-blob** model: each neuron is a single soma voxel
that `sprouting_phase` extends as a uniform-random walk over 6-neighbours.
When Pack M lands real neuron shapes (cell-type-specific morphology stamped
at birth, with dendrite / axon / axon-trunk roles per voxel), several
existing implementations need to be **inverted** — refactored so they
operate on the actual cell shape instead of the soma point.

This doc is a checklist. Each entry: what the pack does today, what
assumption it makes about morphology, what changes when shapes are real.

The audit was generated 2026-05-04 against commit `58968b8` (Pack 25.1 +
docs + roadmap, Pack M v1 design preserved but unimplemented).

---

## Phase 1 — Inversions REQUIRED (these are wrong without shapes)

### 1. `sprouting_phase` (`src/simulator.cpp:1439`)

**Today**: pick a random body voxel, pick a random of 6 neighbours,
extend NEURON state into the empty neighbour. Direction is uniform over
±x, ±y, ±z. The neuron grows as an isotropic blob.

**Assumption**: a neuron is just an "owned blob"; no preferred axis.

**With real shapes**: sprouting must be **directionally biased per cell
type**. Pyramidal cells extend their apical dendrite preferentially +z
(toward pia); their axon descends -z and then branches laterally in
target layers. PV basket cells extend a dense local axonal arbor (short
±x / ±y from a hub voxel). SST Martinotti cells extend ascending +z. VIP
cells extend lateral. Each cell's morphology template defines a
*growth bias vector* per voxel role.

**Inversion**: replace the uniform `nbr(rng_)` direction pick with a
weighted sample over the 6 directions, weights derived from the cell's
polarity and the source voxel's role (dendrite vs axon). Add a
per-cell-type `growth_bias[6]` table.

### 2. `synaptogenesis_phase` (`src/simulator.cpp:1502`)

**Today**: any NEURON-NEURON contact between two distinct owners can
become a SYNAPSE; the outgoing edge is from the *active* neuron to the
neighbour. There's no notion of "axon side" vs "dendrite side".

**Assumption**: every body voxel can be either pre or post — the
direction is decided by which neuron is firing at that step.

**With real shapes**: real synapses form only at **AXON of pre × DENDRITE
of post** contacts. An axon-axon meeting doesn't form a synapse; two
dendrites meeting don't form one either. The pre/post direction is
pinned by morphology, not by recent firing.

**Inversion**: extend `Neuron::body` to carry a per-voxel role tag
(dendrite / axon). Synaptogenesis only triggers when one side is AXON
and the other side is DENDRITE; the AXON owner is automatically pre.
This removes the "which neuron is active right now" tiebreak and
replaces it with anatomical correctness.

### 3. `install_synapse` (`src/simulator.cpp:491`)

**Today**: places a SYNAPSE voxel at the post's soma (or at a chosen
location) and creates an outgoing edge. Bypasses morphology entirely.

**Assumption**: a synapse can sit "anywhere reasonable".

**With real shapes**: the synapse voxel should land on an actual
axon-of-pre × dendrite-of-post **contact**. For labelled-line installs
where pre and post don't yet have overlapping shapes (e.g. ext_in →
motor across the cortex), `install_synapse` should *grow* a short axonal
process from pre and accept it onto an existing dendrite of post — or
warn that no contact exists.

**Inversion (incremental)**: keep the existing edge-creation API for
back-compat, but emit a `synapse_pos` that lies on a verified
axon-dendrite contact when one exists. For long-range installs, pre's
morphology gets an extra axon-trunk extension reaching post's dendritic
field at install time.

### 4. `Neuron::body` storage

**Today**: `std::vector<Voxel>` — just positions, no roles.

**With real shapes**: needs to carry voxel role per entry, or be split
into `dendrite_body` / `axon_body`.

**Inversion options**:
- `std::vector<MorphologyVoxel>` (with `role` field) — minimal change,
  but every existing iterator over `body` needs to read the role.
- Two parallel vectors `dendrite_body` and `axon_body`. Cleaner for
  iteration but more state to persist.

Persist the voxel role through the SNC10 save format bump.

### 5. Dendritic branches (`Neuron::n_branches`, `branch_potential`)

**Today**: each neuron has `n_branches` independent integrators; each
synapse picks a branch index at install time. Branch indices have no
spatial meaning — branch 0 vs branch 1 is just "different integrator
pool".

**Assumption**: branches are abstract integrator slots.

**With real shapes**: branch indices should map to **physical
dendrite groups**. Branch 0 = apical dendrite, branch 1 = basal dendrite,
branch 2 = oblique, etc. A synapse formed on an apical-dendrite voxel
automatically routes to branch 0; basal-dendrite voxel → branch 1.

**Inversion**: in `synaptogenesis_phase`, derive the branch index from
the post's dendrite voxel position relative to its soma. Apical voxels
(dz > 0) → branch 0, basal (dz ≤ 0, |dx|+|dy| > 0) → branch 1, oblique
→ branch 2. Existing `install_synapse(branch)` argument becomes a hint
that `synaptogenesis_phase` ignores when morphology is available.

---

## Phase 2 — Inversions OPTIONAL (work fine without shapes, better with)

### 6. `set_engram_region` + niche multiplier in `promote_engram` (Pack 23 + 25.1)

**Today**: per-class niche is a sphere centered on (x, y, z). The
distance check uses `n.soma.x` etc.

**With real shapes**: a candidate cell's *dendritic field* might
overlap the niche even when the soma is outside. Real cortex couldn't
care less about soma position when the dendrites stretch elsewhere.

**Inversion (later)**: replace soma-distance with min-distance over the
cell's body voxels. Costs O(body) per candidate; cheap.

### 7. `apply_position_prior` (`src/simulator.cpp`)

**Today**: BCM `activity_baseline` for a newborn is averaged from
neighbours in the same region bin (8³ voxels by default).

**With real shapes**: neighbours could be defined as *cells whose
dendrites contact the newborn's dendritic field*. Richer biological
analogue (BDNF / TNF-α retrograde signalling actually travels along
dendritic contacts).

**Inversion (later)**: low priority. Bin-based prior is already
plausible.

### 8. CSV dump (`dump_csv`) and visualisation scripts

**Today**: dumps NEURON voxels and SYNAPSE voxels with owner ids.
Visualisation scripts colour by owner id, producing isotropic blobs.

**With real shapes**: should also dump per-voxel role (dendrite / axon
/ trunk). Visualisation scripts can then colour by role and show actual
neuron morphology — this is what the user wants to see.

**Inversion**: extend `dump_csv` to emit a `role` column;
`scripts/render_anatomy.py` uses it to colour dendrites blue, axons
green, trunks grey. Quick win once Pack M lands.

### 9. Save / load format (SNC9 → SNC10)

**Today**: SNC9 persists `Neuron::body` as a flat list of voxels.

**With real shapes**: must persist the per-voxel role too.

**Inversion**: bump magic to "SNC10", add a uint8 role per body voxel.
Backwards-compatible reader can default role=DENDRITE for SNC9 files.

---

## Phase 3 — Implementations that DON'T NEED inversion

These are per-edge, per-channel, or per-region — they don't reason about
neuron shape and stay correct as-is:

- **Engram protection** (Pack 19A): `permanent` flag on `SynapseEdge`.
  Per-edge.
- **Persistent engram membership** (Pack 20): `engram_members_[c]` is a
  list of neuron ids. Per-cell, not per-voxel.
- **Saccadic vision** (Pack 21): drives 16 image-INPUT neurons. The
  vision *organ* is per-channel, not per-shape.
- **Spaced rehearsal curriculum** (Pack 24): pure scheduling, no
  spatial reasoning.
- **CREB allocation + memory linking + silent engrams** (Pack 25):
  per-neuron `excitability_bias` and per-class session id.
- **Pack 25.1 niche floor**: still operates on soma distance; will
  pick up the morphology improvement automatically when (6) lands.
- **STDP, BCM, heterosynaptic damp, homeostatic scaling, eligibility**:
  all per-edge or per-cell, no morphology dependence.
- **Energy field**: per 8³ region. Cell shapes interact with it
  implicitly via `sprout_cost` — no API change.
- **Sleep modes** (`sleep_consolidate`, `sleep_replay_patterns`,
  `sleep_sws_replay`, `sleep_rem_replay`): drive INPUT neurons with
  patterns. Per-channel.
- **Reward learning** (`apply_reward`, `apply_reward_per_class`,
  `apply_aversive`): per-edge.
- **Predictive coding** (`apply_prediction_pattern`): per-INPUT-channel.

---

## Phase 4 — How Pack ZZ should be designed *with* morphology in mind

Pack ZZ (microglial pruning) is the next pack and ships before Pack M.
It introduces eat-me / don't-eat-me tags on synapses. The user plans
to retry Pack M after Pack ZZ lands; Pack ZZ should be designed so it
extends naturally to morphology-aware pruning.

**Design choices that compose well with Pack M**:

- **Tag *synapse* edges, not body voxels**: Pack ZZ's `eat_me_tag` /
  `dont_eat_me` live on `SynapseEdge`, which is morphology-agnostic.
  Body-voxel pruning (e.g. retracting a dendritic spine that hasn't
  formed any synapse in N steps) is a *different* phase that Pack M
  can add later — call it "spine retraction phase". Pack ZZ doesn't
  need to anticipate it.

- **Permanent flag on labelled-line synapses**: Pack ZZ already plans
  `dont_eat_me = ∞` on `permanent == true`. After Pack M, when
  `install_synapse` may grow an axon trunk to deliver the connection,
  the trunk's voxels (BLOCKED state) are not pruned by Pack ZZ —
  Pack ZZ only looks at SYNAPSE-state edges.

- **Per-region cap on removals**: matches microglia's local territory.
  Independent of morphology.

So Pack ZZ's design is morphology-clean: it operates on the synapse
graph, not on the structural grid voxels. Pack M can then add a sister
phase ("dendritic spine retraction") that prunes dendrite voxels with
no incoming synapses, again per-voxel and per-cell, without conflicting
with Pack ZZ's edge-level cleanup.

---

## Estimated effort for the inversions

When Pack M lands, the inversions listed in **Phase 1** (1–5) take
~2–3 days total. Phase 2 (6–9) is another ~1 day. Phase 3 needs no
work. Phase 4 (Pack ZZ design) costs zero — just design awareness now.

The roadmap effort estimate already accounts for Pack M (1.5–2 days);
add ~3 days for Phase 1 inversions. Total post-Pack-ZZ work to reach
real neuron shapes: ~5 days.
