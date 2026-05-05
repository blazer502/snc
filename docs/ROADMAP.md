# SNC Roadmap

The remaining work, end-to-end, toward the project's north star: **a
pre-adolescent cortical capability — minimal mathematical and communication
skill grown from biologically-grounded sensory organs and learning rules**.
This document is the single source of truth for what comes next, why, and in
what order. Each pack lists motivation, mechanism, API, implementation steps,
success criteria, citations, and dependencies.

The pack history (what's already shipped) lives in [the README](../README.md);
the per-pack failure record lives in
`~/.claude/projects/-home-chanyoung-snc/memory/snc_lessons.md`.

## Live baseline

**Pack 26-C.tune.lite — motor speech** (commit `d4b2c36`) —
**100% accuracy across 25 sessions, 20/20 perfect recall in pure-review**,
with auditory (Pack 26-A) + visual (Pack 26-B) + motor speech (Pack 26-C)
all active. The brain hears words spoken (Peterson-Barney 1952 formants
over Greenwood 1990 cochlea), sees them (Hubel & Wiesel 1962 V1
orientation-tuned receptive fields), and produces distinct articulator
trajectories per word (DIVA-template; 5 articulators × 1-2 firings per
word's dominant phonemes). 20 words, 8 semantic groups.

## Dependency graph

```
                              Pack P-lite v2             (LANDED)
                              (parallel work queue)
                                │
                                ▼
                              Pack ZZ                    (LANDED)
                              (microglial pruning)
                                │
                                ▼
                              Pack M v2                  (LANDED)
                              (real neuron shapes,
                               BLOCKED-stamped)
                                │
                                ▼
                              Phase 1 morphology refactor (HARD prereq
                              (AXON × DENDRITE                  for organs)
                               synaptogenesis,
                               role-aware sprouting,
                               docs/MORPHOLOGY_REFACTOR.md)
                                │
                ┌───────────────┼─────────────────┐
                ▼               ▼                 ▼
        Pack 26-A.tune     Pack 26-B          Pack 26-C
        (auditory          (visual organ      (motor speech
         pathway retry)     retina+LGN+V1)     vocal-tract decoder)
                │               │                 │
                └────────┬──────┴────────┬────────┘
                         ▼               ▼
                   Pack 27          Pack 28
                   (network         (hierarchical
                    diagnostics)     predictive coding)
                         └────────┬───────┘
                                  ▼
                              Pack 29
                              (counting + two-word
                               combinations)
                                  │
                                  ▼
                              Pack 30+
                              (extended vocabulary,
                               semantic relations,
                               simple syntax)
```

---

## Phase 0a — Optimisation (LANDED)

### Pack P-lite v2 — OpenMP parallel workers over the delivery queue

**Status**: v1 LANDED (`3069085`). v2 LANDED (`f4eec6a`).

**Why**: v1 made spike dispatch event-driven and single-threaded (workers
are sequential). v2 adds the parallel piece — workers pull from per-region
sub-queues so each post-synaptic neuron has only one writer. The user's
4-step description (track structure → apply model → activate synapse →
worker accepts work) is fully realised when v2 ships.

**Mechanism**: partition the ring slot's `vector<DeliveryEvent>` into N
sub-buckets keyed by `target_neuron % N`. Each thread processes its own
bucket — no shared writes to `branch_potential` or `incoming_queue` except
on the same post, which only one thread sees. STDP-LTD writes to `syn`
state belonging to multiple pres but each `(pre, syn_idx)` is unique
within the slot, so no contention there either. Determinism preserved
because partition is by post id, not by thread id.

**Implementation steps**:

1. Group `DeliveryEvent`s by `target_neuron % cfg.event_dispatch_threads`
   at fire-dispatch time.
2. `event_dispatch_phase` runs the per-bucket loops under
   `#pragma omp parallel for schedule(static)`.
3. Verify lifetime sweep ≥ 83% s15 (Pack P-lite v1 baseline).
4. Profile: at 12-word scale parallelism may not matter; tested at the
   pre-adolescent 128×128×96 stage it should.

**Estimated effort**: 0.5 day. Low risk.

---

## Phase 1' — Pack M (real neuron shapes) [LANDED]

**Status**: v1 reverted at 67% s15 (pre-Pack-ZZ). v2 LANDED at 91.7% s15
(commit `34eb845`).

**What shipped**: 1-voxel per-polarity templates stamped as BLOCKED
(role=2 axon-trunk semantics) at neuron birth. `stamp_morphology` is
idempotent and triggered automatically from `add_neuron_at`,
`set_role`, `set_polarity`. INPUT / OUTPUT cells skipped (their
labelled-line synapses are tuned against the single-voxel baseline).
2-voxel templates regress to 83% so 1-voxel is the current fit.

**The user's original encoding now realises its intent**:
- "1" NEURON = part of a real neuron's morphology (soma + processes)
- "2" SYNAPSE = contact between two neurons' "1" voxels
- "3" BLOCKED = tissue that exists but isn't synapse-eligible
                (now used for morphology stamps; will be specialised
                further when Phase 1 refactor lands AXON × DENDRITE
                synaptogenesis)

**Phase 1 morphology-refactor work** (`docs/MORPHOLOGY_REFACTOR.md`)
remains pending — when sprouting / synaptogenesis becomes
morphology-aware, the templates can switch from role=2 BLOCKED back
to role=0 DENDRITE / role=1 AXON and expand toward multi-voxel
Markram-Blue-Brain shapes.

## Phase 0b was Pack M but reordered (historical)

**Pack M v1 attempted 2026-05-04 and reverted.** Even minimal templates
(1 voxel per INTERNAL cell, INPUT/OUTPUT skipped) regressed the lifetime
sweep to 67% s15 (vs 75% baseline). Two-voxel templates per cell regressed
to 42% s15. The substrate-overhead pattern is identical to Pack 26-A:
adding *any* extra structural state overruns the brain's sprout/prune
balance. Pack M is now ordered AFTER Pack ZZ in the dependency graph
because microglial pruning is the same prerequisite for both.

The Pack M design (templates, stamping API, role/polarity-driven re-stamp)
is preserved below for the post-Pack-ZZ retry.

### Pack M — Real neuron shapes (morphology templates) [BLOCKED on Pack ZZ]

**Status**: planned. Foundational. Reframes how every subsequent pack thinks
about a "neuron" — from a point soma to an actual 3D shape.

**Why** (project vision): the simulator's name is *Structural* Neuromorphic
Computing — its claim to biological fidelity rests on the structural matrix
representing real neuronal morphology. Currently `add_neuron_at` places a
single soma voxel and `sprouting_phase` extends it as a random isotropic blob
(uniform over 6 neighbours). No cell-type-specific apical-vs-basal direction,
no axonal projection. The "3 BLOCKED" voxel state — designed to mark tissue
that "may exist but cannot become a synapse" — is currently used only for
radial-glia scaffolds, not to construct actual neuron shapes. Pack M fixes
this: each neuron is born with a morphologically realistic dendritic tree
and axonal arbor stamped into the grid, and `Neuron::body` becomes the *set
of "1" voxels* forming the cell's shape. Synapses (state "2") emerge where
shapes physically contact — exactly as in real cortex.

**Concept**:

```
   ENABLED BY THE 2-BIT GRID:
   ───────────────────────────────────────────────
   1  NEURON  = part of a real neuron's morphology
                (soma, dendrites, axon — all "1")
   2  SYNAPSE = contact between two neurons' "1" voxels
   3  BLOCKED = tissue that exists but isn't part of a
                synapse-eligible neurite (radial glia,
                vasculature, axon hillock, myelinated
                axon segments — Pack M may use this for
                axonal-trunk segments)
```

**Mechanism**:

- A `MorphologyTemplate` is a list of relative voxel offsets and roles
  (soma / apical / basal / axon / axon-trunk-blocked).
- Cell-type → template lookup (excitatory pyramidal, PV basket, SST
  Martinotti, VIP, INPUT, OUTPUT).
- At neuron birth (`add_neuron_at`, `birth_neurons`, `seed_fetal`'s VZ /
  brainstem / thalamic / aversive / cortical-plate placements), the template
  is stamped centered on the soma. Each non-soma voxel is placed only if the
  target voxel is EMPTY and within bounds — collisions are silently skipped
  (the cell loses that branch in that direction; a structural fault, like
  in real development).
- Synapses still form via `synaptogenesis_phase` at NEURON↔NEURON contacts
  — the change is that NEURON voxels are now placed deterministically by
  morphology, not stochastically by sprouting.
- Sprouting becomes *secondary*: it grows the existing shape further into
  free space (analogous to dendritic spine elongation), still
  activity-driven but biased toward the cell's preferred axis (pyramidals
  +z, inhibitories laterally).

**Templates** (cell-type → relative voxels around soma at origin):

| cell type | dendrites (basal/apical) | axon | total non-soma voxels |
| --------- | ------------------------ | ---- | --------------------- |
| Pyramidal (EXC) | apical: (0,0,+1)..(0,0,+3); basal: (±1,0,0), (0,±1,0) | (0,0,-1), (0,0,-2), (±1,0,-2) | 9 |
| PV basket | (±1,0,0), (0,±1,0), (0,0,±1) | local axon: 6 neighbours of (0,0,-1) | ~10 |
| SST Martinotti | (0,0,+1), (0,0,+2) | ascending: (0,0,+3), (0,0,+4), (0,0,+5) | 5 |
| VIP | (±1,0,0), (0,±1,0) | (0,0,-1), (0,0,-2) | 6 |
| INPUT | (0,0,+1) (1-voxel "axon hillock" toward cortex) | — | 1 |
| OUTPUT | (0,0,-1) (1-voxel "axon" toward white matter) | — | 1 |

These are deliberately small in v1 — the 64×64×48 toddler grid only has
196 608 voxels and ~390 starting neurons; even at ~10 voxels per neuron
total occupancy stays under 2%.

**API additions**:

```cpp
struct MorphologyVoxel {
  int8_t dx, dy, dz;
  uint8_t role;   // 0 dendrite, 1 axon, 2 axon-trunk (BLOCKED)
};
struct MorphologyTemplate {
  const MorphologyVoxel* voxels;
  int n;
};
// Look up the right template for a given polarity / role.
MorphologyTemplate morphology_for(NeuronPolarity pol, NeuronRole role);
// Stamp the template into the grid centered on the neuron's soma.
// Skips voxels out of bounds or already occupied.
void Simulator::stamp_morphology(Neuron& nu, MorphologyTemplate t);
```

**Implementation steps**:

1. Add `MorphologyVoxel` / `MorphologyTemplate` headers and the static
   per-cell-type tables.
2. Add `Simulator::stamp_morphology` that places NEURON / BLOCKED voxels
   relative to the soma and appends them to `nu.body` (NEURON voxels only).
3. Modify `add_neuron_at` to call `stamp_morphology` after the soma is set.
   Default template = pyramidal; callers that want a different shape call
   `stamp_morphology` again after `set_polarity` / `set_role` (or the
   simulator re-stamps on polarity change).
4. Modify `seed_fetal` to stamp morphologies for VZ / brainstem / thalamic
   / aversive / cortical-plate cells. VZ neurons get a minimal "newborn"
   template (soma + 1 leading process), upgraded after migration.
5. Verify build + smoke test (build_anatomy must still place all
   labelled-line cells without collisions; the `place_or_nearby` collision
   helper added in earlier Pack 26 attempts is still useful here).
6. Run lifetime sweep — must still hit ≥ 75% s15. If regressed: shrink
   templates by half, retry. Goal is to reach a template size that lands
   without regression, then expand toward biological fidelity.

**Success criteria**:

- ✅ Lifetime sweep ≥ 75% s15 with default morphologies stamped.
- ✅ `Neuron::body` is non-trivial (size > 1) for every neuron immediately
  after `add_neuron_at` — verified via a `status` printout.
- ✅ Synaptogenesis between adjacent INPUT and motor cells still produces
  the expected labelled-line connectivity (i.e., morphology overlap
  produces synapses where it should).
- ✅ Bonus: an anatomical visualisation script (`scripts/render_anatomy.py`)
  shows recognisable pyramidal / interneuron shapes.

**Risk + mitigations**:

- **Substrate occupancy jumps.** With stamps at ~10 voxels per neuron,
  occupancy at session 1 goes from <1% to ~2%. Demand-driven growth
  threshold (10%) and pruning still apply. If pre-stamping fills regions
  too densely, sprouting is gated naturally by the 4-neighbour
  volume-exclusion threshold.
- **Collision with build_anatomy's deterministic placements.** Existing
  ext_in / image_in / motor / self / inhibitor cells assume free voxels at
  hand-chosen positions. After Pack M, those positions must accommodate
  the cell's stamp. Mitigation: in `build_anatomy`, after `set_polarity`
  on a labelled-line cell, re-run `stamp_morphology` with the chosen
  polarity-specific template; collisions silently skip (acceptable for
  v1).
- **The sprout/prune balance shifts.** Pack ZZ becomes more important
  because the substrate is denser at start. Pack M ships first to test
  whether the new initial occupancy alone regresses, then Pack ZZ adds
  active shedding.

**Citations**:

- Markram et al. 2015 *Cell* 163:456 — *Reconstruction and simulation of
  neocortical microcircuitry* (Blue Brain Project, the canonical
  morphology library).
- Ascoli et al. 2007 *J. Neurosci.* 27:9247 — NeuroMorpho.org morphology
  database.
- DeFelipe et al. 2013 *Nat. Rev. Neurosci.* 14:202 — pyramidal-cell
  diversity across cortical layers and species.
- Tremblay, Lee & Rudy 2016 *Neuron* 91:260 — GABAergic interneuron
  morphology by subtype (PV/SST/VIP).
- Kandel Ch. 2 (cell types in nervous system).

**Estimated effort**: 1.5–2 days. Risk-managed by starting small and
expanding if baseline holds.

---

## Phase 1 — Hard prerequisite (LANDED)

### Pack ZZ — Microglial pruning [LANDED at 6de27d1]

**Status**: v1 + v2 reverted; v3 LANDED (commit `6de27d1`). Lifetime sweep
91.7% s15.

**What shipped**: silence-age criterion (synapse silent for >900 steps AND
weight <0.10 AND consolidation_tag <0.05 AND not permanent AND post.role !=
OUTPUT) with per-region cap 1, warm-up 2500 steps. Eat-me / don't-eat-me
fields are wired and persisted but not currently the trigger; available
for a future tag-based variant.

The v1 / v2 attempts used eat_me_tag accumulation as the criterion and
both regressed (50% / 75% s15). The simpler silence-age path proved more
stable in tuning — the tag accumulator over-counts on synapses that
deliver but rarely cause fires (which is most of them in a sparse
network), while raw silence captures the "genuinely unused" signal more
directly.

### Pack ZZ - reference (original design preserved below)

**Status**: planned. Hard prerequisite for *every* organ-adding pack
(Pack 26-A, 26-B, 26-C). Pack 26-A.tune was reverted four times because
the substrate has no headroom; Pack ZZ creates the headroom.

**Why**: the live baseline depends on a tight sprouting / weight-decay /
spine-retraction balance. Adding even 16 extra cells regresses by 1 word at
session 15. Real cortex sheds synapses actively via microglial phagocytosis
(complement-tagged synapses get eaten); the simulator currently only sheds
through passive weight-decay falling below the spine-retraction floor. Active
shedding lets the substrate absorb new sensory pathways.

**Mechanism** (Xing, Wang, Yang, Tang 2026 *NRR* 21:1698; Schafer & Stevens
2013 *Curr. Opin. Neurobiol.* 23:1034):

- **Eat-me tag (C1q analogue)**: each synapse accumulates a complement-like
  tag that grows when post-synaptic firing rarely follows delivery (low
  `caused_fire_count / delivered_count`) and decays when use rate is high.
- **Don't-eat-me protection (CD47/SIRPα analogue)**: each synapse carries a
  protection level. Permanent / engram synapses always get
  `dont_eat_me = ∞`. Recently-used synapses build a small protection level
  that decays slowly.
- **Microglial phagocytosis**: a per-region eligibility-driven pass removes
  synapses where `eat_me_tag - dont_eat_me > threshold`, capped per region
  per step. Microglia are localised, so removals can't outpace structural
  growth globally.

**API additions**:

```cpp
struct SynapseEdge {
  // ... existing fields ...
  float eat_me_tag = 0.0f;      // C1q analogue, climbs with disuse
  float dont_eat_me = 0.0f;     // CD47/SIRPα analogue, set on permanent
};

struct SimConfig {
  // ... existing fields ...
  float microglia_tag_growth = 0.001f;       // per low-use step
  float microglia_tag_decay  = 0.99f;        // per use
  float microglia_eat_threshold = 1.0f;      // tag - protect > thresh
  int   microglia_max_remove_per_region = 1; // per pass
  int   microglia_pass_period = 50;          // run every N steps
};
```

**Implementation steps**:

1. Add the two fields to `SynapseEdge`. Persist in save/load (SNC10 bump).
2. Update `pruning_phase` (or a new `microglia_phase`):
   - Each step, update `eat_me_tag` and `dont_eat_me` based on the local
     delivery / firing statistics.
   - Every `microglia_pass_period` steps, walk synapses and remove those with
     `eat_me_tag - dont_eat_me > threshold` AND `permanent == false`,
     respecting the per-region cap.
3. Permanent synapses (engram members, labelled-line priors) get
   `dont_eat_me = std::numeric_limits<float>::max()` at install.
4. Verify the lifetime sweep — must still hit ≥75% s15. Pack ZZ must NOT
   regress on its own.
5. Bonus criterion: total synapses at s15 should be lower than baseline
   (active shedding visibly reducing the connectome).

**Success criteria**:

- ✅ Lifetime sweep ≥ 75% s15 with no other changes.
- ✅ Total synapse count at s15 lower than the Pack 25.1 baseline.
- ✅ Permanent synapses (engram members, priors) never removed.

**Risk + mitigations**:

- A prior microglial pruning attempt (Pack 26 v1) regressed because it lacked
  the don't-eat-me path and removed engram-adjacent synapses. The new
  mechanism explicitly protects them.
- Start with very gentle removal cap (1 per region per 50 steps).
- If baseline regresses: tighten the eat-me threshold, or lengthen
  `microglia_pass_period`, or revert per-piece.

**Citations**:

- Xing et al. 2026 *Neural Regen. Res.* 21:1698 — `resource/NRR-21-1698.pdf`.
- TREM2 / lipid metabolism papers — `resource/TREM2-*.pdf` (read in detail
  for v2 of this pack).
- Schafer & Stevens 2013 (review).
- Stevens et al. 2007 *Cell* 131:1164 (C1q-tagged synapse pruning).

**Estimated effort**: 1–2 days.

---

## Phase A — Sensory organs

### Pack 26-A.tune (retry) — Cochlear pathway

**Status**: BLOCKED — 5 reverts. Pack ZZ + Pack M provide substrate
headroom but the cochlear pathway introduces dynamic perturbations
that destabilise the curriculum equilibrium. Real fix is Phase 1
morphology refactor (AXON × DENDRITE synaptogenesis); see
`docs/MORPHOLOGY_REFACTOR.md`.

5th attempt (post-Pack-ZZ + Pack-M, 2026-05-04): with pre-wired
A1→motor 75% s15, anatomy-only 67%, no-prewire-with-acoustic 50%.
Pack M v2 baseline (91.7%) restored.

The architecture playbook lives in
`~/.claude/projects/-home-chanyoung-snc/memory/snc_pack26a_status.md`.

**Why** (User Directive 1): the brain must receive sound through a simulated
cochlea, not via labelled feature vectors. Words enter as formants on a
log-frequency basilar membrane, ascend through CN/IC/MGN/A1, and bind to
motor outputs through STDP. Currently the network only sees label-INPUTs that
are essentially symbolic.

**Architecture** (full version, recreate after Pack ZZ):

```
   cochlea (32 INPUT bins, log-frequency 200..4000 Hz, Greenwood 1990)
       │ 2:1 convergence, weight 0.45, delay 3
       ▼
   cochlear nucleus (16, z=4)
       │ 1:1, weight 0.55, delay 5
       ▼
   inferior colliculus (16, z=5)
       │ 1:1, weight 0.55, delay 5
       ▼
   medial geniculate (16, z=6)
       │ 1:2 divergence, weight 0.55, delay 2 (Bruno-Sakmann)
       ▼
   primary auditory cortex A1 (32 INTERNAL, z=8)
       │ plastic, weight 0.10, delay 4, branch 1
       ▼
   motor cortex (12 OUTPUTs)
```

Per-word formants from Peterson-Barney 1952; Gaussian-skirted formant bins
with σ=2 bins. Temporal envelope: 4 onset + 18 vowel + 4 offset = 26 steps.

**Implementation steps** (after Pack ZZ lands):

1. Recreate `kCochleaBins`, formant table, channel layout. Bump `kAllFeatures`
   to 108. Restore Brain struct fields and `rebuild_index`.
2. Place neurons via `place_or_nearby` helper at the documented coordinates.
   Keep z-shift (z=0/4/5/6/8) — original z=0..5 collides with seed_fetal at
   seed=1234.
3. Wire the labelled-line ascent with documented weights/delays.
4. Add `freq_to_bin`, `cochlea_pattern_at`, `run_acoustic_present`.
5. Bias 3.0 on tonotopic A1 cells matching word's formants in `cmd_teach`.
   Pack 25.1's niche-floor lets them survive promote_engram.
6. Verify lifetime sweep ≥ 75% s15.

**Success criteria**:

- ✅ ≥ 75% s15 lifetime sweep with full pathway active.
- ✅ A1 cells appear in the per-class engram membership lists (verify via
  `imagine` recall).
- ✅ Bonus: removing label-INPUT priors for one trained word still produces
  recall via the acoustic path alone.

**Citations**:

- Schreiner & Winer 2007 *Annu. Rev. Neurosci.* 30:151 — auditory cortex tonotopy.
- Bruno & Sakmann 2006 *Science* 312:1622 — thalamocortical synapse strength.
- Peterson & Barney 1952 *JASA* 24:175 — vowel formants.
- Greenwood 1990 *JASA* 87:2592 — cochlear log-frequency map.
- Kandel Ch. 30 (auditory system).

**Estimated effort**: 1 day after Pack ZZ.

---

### Pack 26-B — Visual organ (retina + LGN + V1)

**Status**: planned, blocked on Pack ZZ.

**Why**: Saccadic vision (Pack 21) currently feeds a 4×4 retinotopic image
through hand-designed pixel patterns. Pack 26-B replaces this with a real
retina that does center-surround filtering and an LGN→V1 ascent with
orientation-tuned simple cells.

**Architecture**:

```
   retina (64 photoreceptors, e.g. 8×8 grayscale image)
       │ ON-center / OFF-center receptive fields (2 channels)
       │   weight 0.6, delay 2 (bipolar→ganglion)
       ▼
   ganglion cells (32 ON + 32 OFF, parvocellular pathway)
       │ 1:1, weight 0.55, delay 3
       ▼
   LGN (32 relay cells)
       │ Hubel-Wiesel: pooled responses form orientation-tuned receptive
       │   fields on V1 simple cells. Gabor-like RFs at 4 orientations.
       │ weight 0.4, delay 3
       ▼
   V1 simple cells (16 — 4 orientations × 4 spatial positions)
       │ plastic, branch 1, weight 0.10
       ▼
   motor cortex (12 OUTPUTs)
```

Per-word visual templates (similar to current `kImageBits` but on an 8×8
grayscale grid). Saccadic attention from Pack 21 stays — saccades drive
sequential V1 activations that V1→motor STDP binds to the concept.

**Implementation steps**:

1. Replace 4×4 binary `kImageBits` with 8×8 grayscale templates per word.
2. Add retinal photoreceptors (64 INPUT cells, channels 76..139 if Pack 26-A
   active else 76..139 — adjust kAllFeatures accordingly).
3. Center-surround: each ganglion cell pools 3×3 photoreceptors with
   `+1 center, -0.5 surround` for ON, opposite for OFF.
4. LGN as 1:1 relay (mirrors Pack 26-A's MGN role).
5. V1 simple cells: install pre-wired Gabor-like RFs at 4 orientations
   (0°, 45°, 90°, 135°) using `install_synapse` with hand-crafted weights
   (Hubel-Wiesel labelled-line priors).
6. V1 → motor plastic connections shaped by STDP.
7. `cmd_see` updated to drive retina from the word's grayscale template.
8. Verify lifetime sweep ≥ 75% s15.

**Success criteria**:

- ✅ Saccade-driven visual presentation produces correct motor recall on
  trained words.
- ✅ V1 cells show orientation tuning (test with stripe-like images).
- ✅ Lifetime sweep matches or exceeds baseline.

**Citations**:

- Hubel & Wiesel 1962 *J. Physiol.* 160:106 — V1 simple cells.
- Kuffler 1953 *J. Neurophysiol.* 16:37 — center-surround receptive fields.
- DiCarlo, Zoccolan & Rust 2012 *Neuron* 73:415 — ventral visual stream.
- Kandel Ch. 27, 28 (visual system).

**Estimated effort**: 1.5 days after Pack ZZ.

---

### Pack 26-C — Motor speech (vocal tract decoder)

**Status**: planned, blocked on Pack ZZ.

**Why** (User Directive 1, output side): "Output should eventually be a
temporal motor program decoded by an inverse model, not classification
rate-readout." Currently the simulator's "speech" is `argmax(motor rates)`
over 12 motor cells. Pack 26-C replaces this with a temporal motor program
sent to a simulated vocal tract.

**Architecture**:

```
   motor cortex (12 OUTPUTs, current)
       │
       ▼
   premotor sequencer (12 INTERNAL cells)
       │ recurrent, encodes temporal pattern of articulator commands
       │   over time
       ▼
   articulator cells (5: jaw, tongue-tip, tongue-body, lips, glottis)
       │ continuous time-series output
       ▼
   vocal-tract inverse model (offline, in a Python sidecar)
       │ takes articulator trajectories → synthesized formants
       ▼
   audio waveform (or formant trace) — fed BACK to the cochlea
```

Closed loop: the brain "speaks", produces formants, hears them via efference
copy + the cochlear pathway. This realises Levelt 1989's articulatory loop
and matches User Directive 1's biological-output requirement.

**Implementation steps**:

1. Add 5 articulator cells (INTERNAL with sentinel channels).
2. Add 12 premotor sequencer cells with recurrent connections (decay-based
   temporal integrator).
3. Wire `motor → premotor → articulator` with STDP on the
   premotor→articulator connections so each motor learns its articulator
   pattern through reward.
4. Sidecar Python tool reads articulator trajectories from the simulator
   (via dump_csv hook) and synthesises formants using a simple Klatt-style
   vocal-tract model.
5. The synthesised formants feed back via the cochlea (Pack 26-A) — the
   network HEARS itself speaking.
6. Reward signal: when the synthesised formants match the target word's
   formants (Peterson-Barney distance), apply positive reward.

**Success criteria**:

- ✅ Each trained motor produces a distinct articulator trajectory.
- ✅ Synthesised audio is recognisable to the brain itself (cochlear input
  → motor recall round-trips).
- ✅ Babble produces random articulator sequences; teaching shapes them.

**Citations**:

- Levelt 1989 *Speaking: From Intention to Articulation* (MIT Press) —
  articulatory loop.
- Klatt 1980 *JASA* 67:971 — formant synthesis.
- Tourville & Guenther 2011 *Lang. Cogn. Neurosci.* 26:952 — DIVA model.
- Kandel Ch. 39 (motor program decoding, BMI link).

**Estimated effort**: 2–3 days after Pack ZZ + Pack 26-A.

---

## Phase B — Cortical infrastructure

### Pack 27 — Network diagnostics

**Status**: planned. Read-only. Can be done in parallel with Pack 26-x.

**Why**: the network's small-world / hub / modularity structure is currently
invisible. Pack 27 adds instrumentation so we can SEE whether learning
produces the canonical brain network properties — a precondition for
non-trivial cognitive function (Sporns 2010, Bullmore & Sporns 2012).

**Metrics to compute**:

- **Clustering coefficient** (per node + global average).
- **Characteristic path length** (BFS over the synapse graph).
- **Small-world index** σ = (C/C_random) / (L/L_random).
- **Hub identification**: degree-based + betweenness-centrality.
  Distinguish provincial (within-module) vs connector (between-module) hubs.
- **Modularity Q** via Louvain or spectral clustering on the synapse graph.
- **Φ-proxy** (IIT): compute integrated information on a small subnetwork
  (full Φ is infeasible at full scale; a small 8-cell subnetwork around an
  engram is tractable per Albantakis 2023).

**Implementation steps**:

1. New `Simulator::network_stats()` API returns a `NetworkStats` struct.
2. Python tool in `scripts/network_diagnostics.py` walks the dumped
   synapses CSV and computes the metrics.
3. Plots: degree distribution (lognormal expected), small-world index over
   developmental time, hub locations overlaid on the 3D anatomy.
4. Add `Φ-proxy` for an 8-cell subnetwork around a chosen engram.

**Success criteria**:

- ✅ Small-world index σ > 1 on the trained brain (vs σ ≈ 1 on a random graph).
- ✅ Lognormal degree distribution (Buzsáki & Mizuseki 2014).
- ✅ Connector hubs found at posterior-medial cortex analogues.
- ✅ Φ-proxy increases with training.

**Citations**:

- Sporns *Networks of the Brain* — `resource/Sporns_Book.pdf`.
- Bullmore & Sporns 2012 *Nat. Rev. Neurosci.* 13:336.
- Albantakis et al. 2023 *PLoS Comput. Biol.* 19:e1011465 —
  `resource/journal.pcbi.1011465.pdf`.
- Cogitate Consortium 2025 *Nature* 642 — `resource/s41586-025-08888-1.pdf`.

**Estimated effort**: 1 day. Read-only, no risk to baseline.

---

### Pack 28 — Hierarchical predictive coding

**Status**: planned, after Pack 26-A so there's something to predict.

**Why**: real cortex doesn't process bottom-up sensory input alone — it
constantly subtracts top-down predictions, so only the *prediction error*
drives learning. The simulator already has `apply_prediction_pattern` for
INPUT neurons, but nothing currently *generates* predictions from higher
layers. Pack 28 closes that loop.

**Architecture**:

```
   Layer L+1 (e.g. motor) maintains a running expectation of L's activity.
       │ top-down feedback weight via a prediction-projection synapse
       │ subtracts from L's input_acc:
       ▼
   Layer L's effective_input = input_acc - predicted_input
   So a perfectly-predicted stimulus produces no surprise (no LTP).
```

The simulator already supports this for INPUT neurons; Pack 28 extends it to
INTERNAL cells too.

**Implementation steps**:

1. Add `Neuron::predicted_internal` field for INTERNAL neurons.
2. Add a `prediction_phase` between integrate and chemistry that applies
   top-down feedback weights on prediction-projection synapses.
3. Mark certain synapses as `prediction = true` (no STDP, but their
   delivery subtracts from `predicted_internal`).
4. Wire `motor → A1` (Pack 26-A) and `motor → V1` (Pack 26-B) prediction
   projections so the brain anticipates its own sensory consequences.
5. Add a `surprise` readout = sum of |effective_input| across cortex.
   Surprise should drop as learning progresses (Friston 2010 free-energy
   principle).

**Success criteria**:

- ✅ Surprise on trained words is lower than on novel words.
- ✅ Lifetime sweep ≥ baseline (predictive subtraction shouldn't hurt;
  ideally slightly improves consolidation since redundant firing drops).

**Citations**:

- Rao & Ballard 1999 *Nat. Neurosci.* 2:79 — predictive coding in V1.
- Bastos et al. 2012 *Neuron* 76:695 — canonical microcircuits for PC.
- Friston 2010 *Nat. Rev. Neurosci.* 11:127 — free-energy principle.
- Mashour, Roelfsema, Changeux & Dehaene 2020 (GNW + PC) —
  `resource/Conscious-Processing-and-the-Global-Neuronal-Works.pdf`.

**Estimated effort**: 1–2 days after Pack 26-A.

---

## Phase C — Capability

### Pack 29 — Counting + two-word combinations

**Status**: the *original* Pack C goal in user directive 4. Blocked on
Phase A (sensory organs). Marks the transition from "12-word naming" to
"minimal communication".

**Why** (User Directive 4): "Vocabulary already at 12 words; goal is
'minimal mathematical and communication skills.' Counting, two-word
combinations, basic comprehension."

**Sub-pack 29-a: Counting (1..5)**

Add five new concepts: "one", "two", "three", "four", "five". Each driven
by both a verbal label (cochlear input) and a quantity stimulus (e.g.
N pixels lit on the visual grid). Engrams must bind quantity ↔ label.

**Sub-pack 29-b: Two-word combinations**

Curriculum extends to two-word episodes:
- "more ball" (request)
- "no dog" (negation)
- "yes mom" (affirmation)
- "bye dad" (farewell)
- "hi baby" (greeting)

Mechanism: present two label patterns sequentially within one teach episode;
reward the corresponding two-motor sequence. The brain learns sequence-of-
motors, not just argmax. Pack 26-C's premotor sequencer carries the
temporal binding.

**Sub-pack 29-c: Subject-verb-object (toddler-grade)**

Extend to three-word episodes: "more dog ball", "no dog bye". Six-word
combinatorial space → simple syntax emerges from sequential STDP on premotor.

**Implementation steps**:

1. Extend `kClasses` from 12 to 17 (add five number words).
2. Add quantity stimulus channels (1..5 dots on visual grid).
3. Curriculum: count by demonstration (visual N-dots + spoken "N").
4. Sequential teaching episodes: present word A, brief silence, present
   word B, reward the AB motor sequence.
5. Decoding: during pure review, the brain emits motor sequences; check
   that "more ball" produces motor[more] then motor[ball] in temporal
   order.

**Success criteria**:

- ✅ Counting: shown N dots, brain says the right number word ≥ 50% of the
  time (guessing baseline 20%).
- ✅ Two-word: shown a familiar AB pair, brain produces motor sequence A
  then B in correct order ≥ 50% of the time.
- ✅ Lifetime sweep extended to 50 sessions still maintains ≥ 70% on the
  original 12 words.

**Citations**:

- Carey 2009 *The Origin of Concepts* (number words and the
  successor function in toddlers).
- Tomasello 2003 *Constructing a Language* — usage-based syntax acquisition.
- MacArthur-Bates CDI norms for 18–24 month olds.

**Estimated effort**: 3–5 days. This is the user-directive-4 goal post.

---

## Phase T — Pack TREE: branching dendritic-tree topology

### Pack TREE — neurons as actual neuron-shaped morphology

(Internal data structure is a tree of `BranchData` per voxel — that's
where the "TREE" name comes from. The end product is neuron-shaped
cells, not abstract trees: real dendritic arbours with proximal
trunks and distal tapering, plus axonal projections, just like
cortical pyramidal / interneuron cells.)


**Status**: planned. The user observed:

> Currently, neurons are represented by 1 bit, so they are depicted at
> the lowest resolution compared to their actual structure. To
> represent more complex and non-linear structures, neurons must
> extend outward like tree roots, rather than using 1 bit, just like
> the actual structure.

**Why**: today's `Neuron::body` is `std::vector<Voxel>` — a flat list
of (x, y, z) positions. Each voxel is binary "this is part of the
neuron / it isn't"; there's no parent-child branching topology, no
proximal-vs-distal distinction, no thickness gradient. Real
dendritic and axonal arbours are **trees**: a primary trunk
branching into secondary branches, each branching further, with
thickness tapering toward the leaves. Pack M v2 + Phase 1' stamp
these voxels at birth, but the *graph* of which voxel is the parent
of which doesn't exist yet — the tree is implicit in the relative
positions of stamped voxels and any branching that emerges via
sprouting is undirected.

Pack TREE replaces flat `body` with an explicit branch graph.

### Mechanism

Replace `std::vector<Voxel> body` with `std::vector<BranchNode> body`:

```cpp
struct BranchNode {
  Voxel    pos;                    // (x, y, z) in the grid
  uint8_t  role;                   // DENDRITE / AXON / AXON_TRUNK
  uint16_t parent_idx;             // index into Neuron::body, 0 = soma
                                    // (UINT16_MAX = no parent / root)
  uint8_t  depth;                  // tree-edges from soma
  uint8_t  thickness;              // 255 at trunk, decreasing distally
};
```

Construction:

1. **Stamp**: morphology templates (Pack M v2 / Phase 1') gain explicit
   `parent_idx` and `depth` per voxel. Soma is `body[0]` with
   `parent_idx = UINT16_MAX, depth = 0, thickness = 255`. Each
   stamped voxel records its parent (typically the soma for the
   first morphology layer; subsequent layers point to nearer voxels).
2. **Sprout**: `sprouting_phase` chooses a parent voxel from the
   current tree's leaves (highest depth, lowest thickness) and
   extends a new BranchNode as that voxel's child. Depth increments;
   thickness decrements (e.g., `parent.thickness * 0.85`). Real
   biology: distal dendrites are thinner because actin / microtubule
   transport falls off with distance.
3. **Prune**: deterministic spine retraction prefers high-depth,
   low-thickness leaves first — modelling the fact that distal
   spines are the most frequently lost during developmental pruning
   and the easiest for microglia to engulf.

### Behavioural / scientific value

- **Synaptogenesis routing**: `synaptogenesis_phase` can derive the
  post's dendritic branch index from the contact voxel's tree
  position (apical apical-tuft → branch 0, basal → branch 1, oblique
  → branch 2). The current `synaptogenesis_default_branch` config
  becomes obsolete because the branch index is anatomically
  determined.
- **Conduction-delay accuracy**: today's delay = Manhattan distance
  from soma. With a real tree, delay = sum of edge lengths along the
  branch path (longer for distal contacts because the path winds
  through the tree). Matches real cortical conduction.
- **Microglial pruning v4**: Pack ZZ's eat-me tag growth multiplied
  by `(thickness_max - thickness)/thickness_max` — distal thin
  branches accumulate eat-me faster, matching Stiles & Jernigan
  2010 adolescent-pruning observations.
- **Visualisation**: `dump_csv` emits parent_idx so visualisation
  scripts render actual neuron shapes (not just voxel blobs). The
  brain's morphology becomes inspectable at a glance.

### Implementation steps

1. Add `BranchNode` struct in `neuron.hpp`, replacing the
   `body` field. Soma is always `body[0]`.
2. `stamp_morphology` populates the tree from template — each
   morphology voxel carries its parent_idx (soma for first ring,
   nearest first-ring voxel for second ring, etc.).
3. `sprouting_phase` picks parent from leaves (current: random
   body voxel) and decrements thickness on extension.
4. `pruning_phase` and `microglia_phase` use depth + thickness as
   priority scores.
5. `synaptogenesis_phase` derives post's branch index from the
   contact voxel's depth / role.
6. `dump_csv` emits parent_idx so scripts/render_anatomy.py can
   draw actual trees.
7. Save format bump to SNC12 (BranchNode is bigger than Voxel).
8. Verify lifetime sweep ≥ 100% s25 with all current packs active.

### Risks

- **Save-format break**: SNC11 brains can't load. Mitigate with a
  one-shot migration that synthesises depth=0 / parent_idx=0 for
  legacy bodies.
- **Memory footprint**: BranchNode is ~10 bytes vs Voxel's 6 bytes.
  At 500 cells × 6 voxels each = 3000 nodes × 10 bytes = 30 KB.
  Negligible.
- **Sprouting / pruning behaviour shift**: current uniform-random
  sprouting becomes leaf-biased. May change baseline dynamics.
  Tunable via a config flag (uniform vs leaf-biased) for safe
  rollout.
- **Synaptogenesis branch routing**: changes which dendritic branch
  receives newly-formed synapses. May affect engram dynamics.
  Mitigation: keep existing `synaptogenesis_default_branch` as a
  fallback when the contact voxel has no clear branch identity
  (depth = 0, etc.).

### Estimated effort

3–5 days. The data-structure change touches ~20 call sites; the
behavioural changes (sprouting, pruning, synaptogenesis routing)
each need their own A/B verification. Risk-managed by feature-flag
gating each behavioural change.

### Slot in roadmap

After Phase A organs (LANDED) and Phase B diagnostics + predictive
coding. Pack TREE deepens the **structural** side of the simulator
by giving every neuron a real branching shape, complementing Phase 1
+ Phase 1' which gave them axon-vs-dendrite roles. With Pack TREE
in place, a future visualisation pass can show the brain as a
forest of neuron-shaped cells rather than the current voxel blobs —
matching the user's vision: "neurons must extend outward like tree
roots, just like the actual structure."

---

## Phase Φ — Consciousness deliberation loop (USER-TRIGGERED)

### Pack Φ — continuous deliberation loop

**Status**: planned, USER-TRIGGERED. The user has indicated this should
sit in the remaining plan and be ordered when they ask. Discussion
document at `docs/DISCUSSION_CONSCIOUSNESS.md`.

**Why**: the simulator is currently mostly reactive (input → cascade →
output). Real cortex has continuous internal dynamics — a deliberation
loop that runs even without external input, sustained by a designated
workspace population, attention selection, and predict-then-compare
cycles. Pack Φ implements that loop using the simulator's existing
biological constraints (energy, engram protection, AXON × DENDRITE
chemistry, microglia, spatial niches).

**Mechanism** (sketch — full design in
`docs/DISCUSSION_CONSCIOUSNESS.md`):

1. **Workspace population (GNW)**: ~12–20 high-degree connector hub
   neurons identified via Pack 27 diagnostics. Tagged
   `is_workspace = true`, given a small intrinsic-bias drive each step,
   exempt from microglial pruning.
2. **Continuous deliberation phase**: a new `deliberation_phase()`
   runs every step regardless of external input. Picks the dominant
   engram (highest mean `fire_rate_ema`), amplifies it, predicts its
   sensory consequences (Pack 28 hook), compares to actual input,
   updates.
3. **Φ-proxy gate**: only allow workspace cells to broadcast when
   local Φ exceeds threshold (Pack 27 measures it). Sub-threshold
   deliberation is "subliminal" — no behavioural effect.

**Prerequisites**: Pack 27 (network diagnostics), Pack 28 (hierarchical
predictive coding). MVP is achievable without Pack 27/28 but with
reduced fidelity.

**What you'd observe**: the brain shows non-zero workspace activity
even with no external input; `imagine` becomes self-sustaining;
`status` between probes reveals the dominant engram the brain is
currently "thinking about". Sleep replay automatically includes
deliberation trajectories from the day.

**Caveats**: this is a functional / mechanistic model, not a
metaphysical one. We do not claim hard-problem consciousness; we
claim functional signatures (workspace ignition, integrated
information, predictive coding loops) that the literature treats as
correlates of consciousness in real brains.

**Estimated effort**: 2 days for MVP (workspace + bias + deliberation
period, no Φ-gate, no prediction). 3–5 days with full Pack 27 + 28
integration.

**Slot**: after Pack 26-C-full closes the I/O loop, after Pack 27 +
28 land, before or alongside Pack 29. Triggered by user request.

---

## Pack V — multimodal external validation (LANDED)

User directive 2026-05-04: *"we need more validation method. For the
easy access, we can use MNIST and CIFRA dataset with voice for
multi-modal dataset"*. The lifetime sweep validates the brain on its
own canonical 4-pixel patterns; Pack V validates against an external,
real-world dataset (MNIST handwritten digits) paired with the spoken
digit name (cochlea voice modality).

**Deliverables (landed):**
- `image_teach <word> p0..p15` — present arbitrary 4×4 pixel pattern
  jointly with the cochlea voice for `<word>` and the label features.
  Mirrors `cmd_teach` but binds an externally-supplied image rather
  than the canonical `kImageBits[c]`.
- `image_test p0..p15` — visual-only readout (no label, no voice). The
  motor argmax is the brain's classification.
- `scripts/prep_mnist.py` — fetches MNIST, mean-pools 28×28 → 4×4,
  binarises at intensity 64/255, writes `data/mnist_{train,test}.csv`.
  Subset to digits 1–4 to match existing `one`/`two`/`three`/`four`
  vocabulary.
- `scripts/run_mnist.py` — drives `snc_chat`: optional bootstrap +
  train + visual-only test. Reports per-class accuracy in two modes:
  `open` (argmax over 20 vocabulary) and `forced` (argmax restricted
  to the 4 digit words).

**First empirical run (warm-started from `lifetime_brain.snc`,
30 train + 20 test per digit):**

| mode   | overall | one | two | three | four |
| :----- | :-----: | :-: | :-: | :---: | :--: |
| open   |  2.5%   | 5%  | 5%  | 0%    | 0%   |
| forced | 21.2%   | 55% | 15% | 5%    | 10%  |

**Interpretation (honest).** Forced-argmax is at chance (25%) on a
4-class problem; open-argmax is *below* the 5% per-class chance baseline
on 20 classes because the brain's previously-consolidated engrams for
`ball` / `baby` / `mom` / `dad` partially overlap MNIST stroke pixels
and dominate the readout. This exposes two architectural limits:

1. **The 4×4 retina is too coarse for stroke recognition.** Even after
   mean-pooling, MNIST digits collapse to 1–4 lit pixels and most
   different digits become mutually indistinguishable (e.g. `1` and `4`
   both centre-column-dominated).
2. **Voice/label engrams dominate visual binding.** With only 30
   training samples per class against pre-consolidated phonemic
   engrams, image-to-motor weight changes can't override the existing
   pixel-pattern → motor associations.

**Pack V's value is the framework, not the accuracy.** It provides a
reproducible external benchmark that future packs (retina expansion,
extended training) can score against, replacing the prior "lifetime
sweep on canonical 4-pixel patterns" which the brain had already
saturated at 100%.

### Pack V-tune — visual-only training mode (LANDED)

Pack V's first run hinted that pre-consolidated phonemic engrams may be
dominating multimodal training. Pack V-tune isolates the visual binding
pathway by adding `cmd_image_teach_visual` (motor prime + image only;
no label features, no cochlea voice, no A1 tonotopic bias) and a
`--mode {multimodal,visual,curriculum}` flag in `run_mnist.py`.

**Three-mode comparison** (warm-start from `lifetime_brain.snc`,
30 train + 20 test per digit, forced argmax over 4 digit words):

| mode       | overall | one | two | three | four |
| :--------- | :-----: | :-: | :-: | :---: | :--: |
| multimodal | 21.2%   | 55% | 15% | 5%    | 10%  |
| visual     | **25.0%** | 75% | 10% | 5%    | 10%  |
| curriculum | 22.5%   | 65% | 5%  | 10%   | 10%  |

**Findings:**
1. Visual-only training does help (+3.8 pp absolute), confirming that
   label-engram dominance was a real but secondary factor.
2. The 4×4 retina remains the dominant bottleneck — even pure visual
   training cannot separate 2/3/4 above chance because all three
   collapse to similar central/curved 4-pixel patterns. Only `one` (a
   distinctive vertical stroke) reaches >50% in any mode.
3. Curriculum (multimodal first, visual second) doesn't beat pure
   visual; the multimodal first half partially "anchors" the engrams
   in a way the visual half can't easily reshape.

**Conclusion:** Pack VR (retina expansion) is the right next step.
Without higher pixel resolution, the substrate cannot represent
stroke-level differences between similar digits.

### Pack V-cross — graded pixels + cross-modal dropout (LANDED)

User feedback 2026-05-05: *"multi-modal data makes people remind the
meaning easily in reality"*. Pack V-tune's visual > multimodal result
was backwards from biology (Damasio 1989 convergence zones; Smith & Yu
2008 cross-situational learning predict cross-modal helps). Two fixes
together restored the expected ordering:

1. **Graded pixels** (`prep_mnist.py` writes float intensities in
   [0, 1] instead of binary). Binary 4×4 collapses 2/3/4 into nearly
   identical masks; graded preserves stroke-density differences.
2. **Cross-modal dropout** (`--mode {cross,dropout}` in run_mnist.py).
   Per-trial picks between `image_teach` (multimodal) and
   `image_teach_visual` (image-only) so the visual pathway gets
   exclusive learning episodes — same principle as modality dropout
   in deep learning, biologically grounded in Damasio convergence
   zones requiring each modality to retain an independent path.

**Cumulative improvement** (forced argmax, 60 train + 30 test/digit,
warm-start from `lifetime_brain.snc`):

| stage                          | overall | one | two | three | four |
| :----------------------------- | :-----: | :-: | :-: | :---: | :--: |
| Pack V binary multimodal       | 21.2%   | 55% | 15% |  5%   | 10%  |
| Pack V binary visual           | 25.0%   | 75% | 10% |  5%   | 10%  |
| Pack V-cross graded multimodal | 25.8%   | 67% | 20% |  3%   | 13%  |
| Pack V-cross graded visual     | 25.0%   | 73% |  7% |  7%   | 13%  |
| Pack V-cross graded **cross**  | **29.2%** | 77% | 13% |  7%   | 20%  |
| Pack V-cross graded **dropout**| **29.2%** | 77% | 10% |  7%   | 23%  |

Pack V-cross **doubles the lift above chance** (4.2 pp → 8 pp above the
4-class 25% baseline). Multimodal also flips back above visual — the
user's "multi-modal helps recall" intuition was correct once the visual
input carried enough information for binding to grow synergistically
rather than redundantly. The remaining headroom (29% → 100%) is bounded
by the 4×4 retina; that's Pack VR's job.

### Pack V-quiz — elementary-school number test (LANDED)

User request 2026-05-05: *"it would be fun to try taking a test
consisting entirely of text-based numbers after training in
multi-model like testing in elementary school"*. Built
`scripts/quiz_numbers.py` -- a structured 3-section quiz on digits
1..4, mirroring a primary-school worksheet:

- **Section A** -- *read the word*: ``show <digit>`` (label features only)
- **Section B** -- *read the printed digit*: ``image_test`` on
  idealised 4×4 printed forms (intentionally different stroke
  geometry from MNIST mean-pool, so this is a real transfer test)
- **Section C** -- *read the handwriting*: ``image_test`` on held-out
  MNIST samples

Scoring: closed-set 4-class argmax (multiple-choice quiz format).

**First scorecard** (Pack V-cross dropout-trained brain):

| Section | Score |
| :------ | :---: |
| A. read the word        | **100%**  |
| B. read the printed digit |  25.0% |
| C. read the handwriting |  18.8%   |
| Overall (avg)            | 47.9%   |

Reads exactly like a real first-grader's report: the symbolic pathway
is rock-solid (the brain *recites* numbers from the word) but visual
transfer is fragile, and the brain over-defaults to ``one`` -- the
shape it has the cleanest engram for. Section B is the most diagnostic
result: a true transfer test the brain has never been trained on, and
it's at chance.

This quiz is the new "is the brain learning generalisable concepts or
memorising MNIST mean-pools?" benchmark.

**Pack VR (next, sketch):** retina expansion 4×4 → 8×8 or 16×16, with
V1 receptive fields tiled accordingly (more orientation/location
columns), and a CIFAR-10 prep pipeline using the existing 4-class
vocabulary (`cat`, `dog`, `ball` already in `kWords`; needs `bird`,
`car`, ...). Expected to also require label-engram regularisation so
that visual binding can compete during multi-modal teaching.

---

## Phase D — Beyond pre-adolescent

### Pack 30+ (sketch)

**Pack 30**: extended vocabulary (100 words from CDI top-100), basic
semantic categorisation (animate / inanimate, object / action).

**Pack 31**: simple grammar — agent-action-patient, articles, verb tense
markers. Statistical learning over sequence patterns.

**Pack 32**: working-memory dynamics. Persistent activity in a designated
prefrontal patch (Wang 2001 *Nat. Rev. Neurosci.* 2:485) carries stimulus
identity across delays, enabling delay-match-to-sample tasks.

**Pack 33**: arithmetic. Counting + ordinal recognition + simple addition
on small N (1+1, 2+1, etc.) using number-line representation in IPS
analogue (Dehaene 2011 *The Number Sense*).

**Pack 34**: theory-of-mind / social cognition (long horizon).

These are sketches — each will get its own detailed plan when the
preceding pack lands and we know what the substrate looks like.

---

## Open questions / research gaps

These are research questions the project keeps bumping into. Each may
warrant a focused investigation pack rather than feature work.

1. **Why does the substrate have no headroom?** Even Pack 26-A.tune.lite
   (16 cells) regresses by 1 word. Is the issue energy budget, owner-map
   crowding, or something subtler about the sprouting/pruning balance?
   Pack ZZ should answer this empirically.

2. **What's the right curriculum granularity?** Pack 24-curriculum's
   spaced rehearsal helps; multi-trial teach hurt. Is there a principled
   way to derive curriculum density from the simulator's consolidation
   timescale (synaptic tag decay)?

3. **Engram drift over many sessions.** The sweep stops at session 15.
   What happens at session 100? 1000? Real cortex remembers childhood
   words for decades; can the simulator?

4. **Cross-pack interference.** Each new pack risks regressing the
   baseline. Is there an automated A/B harness we should build (Pack 27.5)
   so every PR runs the lifetime sweep automatically?

5. **The Φ-proxy and consciousness.** Albantakis 2023 IIT 4.0 is
   formidable to compute at full scale. Does the simulator's Φ-proxy
   correlate with behavioural competence (more taught → higher Φ)? Could
   become a foundation for User Directive 4's "minimal awareness" if the
   project ever grows that scope.

---

## Effort estimate

| Phase | Pack | Days | Cumulative |
| ----- | ---- | :--: | :--------: |
| 0a | Pack P-lite v1 (event-driven dispatch) | LANDED | — |
| 0a | Pack P-lite v2 (parallel workers)      | LANDED | — |
| 1  | Pack ZZ (microglial pruning)           | LANDED | —       |
| 1' | Pack M v2 (morphology, BLOCKED stamps) | LANDED | —       |
| 2  | Phase 1 morphology refactor (AXON×DEND)| LANDED | —       |
| 2' | Phase 1' (multi-voxel arborisations)   | LANDED | —       |
| 2'' | Vocab expansion 12 -> 16 (numbers)    | LANDED | —       |
| A | Pack 26-A.tune.lite (cochlear)          | LANDED | —       |
| A | Pack 26-B.tune.lite (visual V1)         | LANDED | —       |
| A | Pack 26-C.tune.lite (motor speech)      | LANDED | —       |
| A' | Pack 26-C-full (closed-loop articulator → cochlea)| 1.5 | 1.5 |
| B | Pack 27 (network diagnostics)           | LANDED | —       |
| B | Pack 28 (predictive coding for INTERNAL)| LANDED | —       |
| C | Pack 29 v1 (cmd_pair_teach API)         | LANDED | —       |
| C | Pack 29 v2 (sequence-prediction behaviour) | 2–3 | 2–3     |
| C | Pack 29 v3 (counting / quantity binding) | 2–3 | 4–6     |
| V | Pack V (MNIST + voice multimodal validation) | LANDED | — |
| V | Pack V-tune (visual-only training mode)  | LANDED | — |
| V | Pack V-cross (graded pixels + cross-modal dropout) | LANDED | — |
| V | Pack V-quiz (elementary-school number test)      | LANDED | — |
| V'| Pack VR (retina expansion + CIFAR + label-engram regularisation) | 3–5 | — |
| T | Pack TREE MVP — branch data structure       | LANDED | — |
| T' | Pack TREE behavioural (leaf-biased + directional sprout) | LANDED | — |
| Φ | Pack Φ — consciousness deliberation loop (user-triggered) | 2–5 | — |

**Total to user-directive-4 goal**: ~2.5–3.5 weeks of focused work,
assuming no compounding regressions. Pack ZZ comes first because the
v1 attempt at Pack M proved the substrate has no headroom for *any*
extra structural state — same constraint as Pack 26-A. After Pack ZZ
adds active synapse shedding, both Pack M (real neuron shapes) and
Pack 26-A (cochlear pathway) become viable.

---

## Standing rules (carried from user directives + lessons)

1. **Real human-like input/output, not ML-style.** Sensory in via simulated
   organs, motor out via temporal articulator programs.
2. **Connectivity referenced from primary literature.** Cite the source for
   every new pathway's neuron count, weight, delay.
3. **Don't ship regressions.** Lifetime sweep must stay ≥ 75% s15.
4. **Tuning iterations stay local.** Revert per-piece if they don't land.
5. **Memory is the source of truth for project state.** Code doesn't
   narrate history; the auto-memory under
   `~/.claude/projects/-home-chanyoung-snc/memory/` does.
