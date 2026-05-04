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

**Pack 25.1** (commit `84009ec`) — 75% accuracy at session 15 on the 12-word
lifetime sweep. CREB-style engram allocation, memory linking, silent-engram
parameter, bias-overrides-niche.

## Dependency graph

```
                             Pack ZZ                    (HARD prereq)
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

## Phase 0 — Hard prerequisite

### Pack ZZ — Microglial pruning

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

**Status**: blocked on Pack ZZ. Architecture playbook lives in
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
| 0 | Pack ZZ                     | 1–2 | 1–2 |
| A | Pack 26-A.tune retry        | 1   | 2–3 |
| A | Pack 26-B (visual)          | 1.5 | 3.5–4.5 |
| A | Pack 26-C (motor speech)    | 2–3 | 5.5–7.5 |
| B | Pack 27 (diagnostics)       | 1   | 6.5–8.5 |
| B | Pack 28 (predictive coding) | 1–2 | 7.5–10.5 |
| C | Pack 29 (counting + 2-word) | 3–5 | 10.5–15.5 |

**Total to user-directive-4 goal**: ~2–3 weeks of focused work, assuming
no compounding regressions. Each pack is independently testable; the
no-regressions directive applies at every step.

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
