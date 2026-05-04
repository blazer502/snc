# Discussion — from reactive cascade to continuous consciousness loop

The user observed:

> Currently, only reactions to reactions exist; therefore, to function like
> human consciousness, the operation of consciousness must be executed
> dynamically based on philosophical logic. Since the current model is
> designed to operate like dynamic human intelligence, we must first train
> it on the foundation of basic human intelligence, select the elements
> governing consciousness, and make it run a continuous loop according to
> the constraints we have injected.

This document engages with that proposal. It splits into:

1. **Honest diagnosis** of how reactive the current simulator actually is.
2. **What "consciousness" means** in the literature we have on the
   project (GNW, IIT 4.0, Cogitate, Friston).
3. **Concrete architecture** for a "Pack Φ — continuous consciousness
   loop" that fits the simulator's existing primitives.
4. **Caveats** about the philosophical commitment we're (and aren't)
   making.
5. **Where it slots into the roadmap.**

---

## 1. Honest diagnosis — how reactive is the simulator today?

**Mostly reactive, with some intrinsic dynamics.** A `step()` does:

```
integrate_incoming → chemistry (fire decision) → STDP →
fire_dispatch → event_dispatch → homeostatic → pruning →
microglia → energy_regen → sprouting → synaptogenesis
```

External input enters via `apply_input_pattern` (label / image /
cochlear cells); the cascade computes a response; output is read out
of motor `fire_rate_ema`. No external input → minimal motor activity.

**Existing intrinsic activity** (so "only reactions" isn't quite true):

- **Brainstem analogue** in `seed_fetal`: 12 always-on tonic cells at
  z=0..1 give the network a baseline drive even with zero input.
- **`inject_internal_noise`** in chat_demo: small uniform noise added
  to non-skip-noise neurons each step, breaking deterministic
  attractors.
- **Sleep replay** (`sleep_consolidate`, `sleep_sws_replay`,
  `sleep_rem_replay`): drive the network with internal noise or
  pattern replay, no external input. STDP consolidates whatever
  trajectories emerge.
- **Eligibility traces**: synapses carry a slow-decaying memory of
  recent pre-post coincidence, allowing reward to act on past
  activity (a tiny "self" memory across steps).

**What's missing for "consciousness-like dynamics":**

1. **No persistent internal goal**. Each step is independent; nothing
   in the cortex says "I'm trying to recall mom right now" across
   steps unless an external stimulus is keeping mom-engram active.
2. **No global workspace**. Real cortex has long-range projection
   neurons that integrate from many areas and broadcast back —
   the GNW substrate. The simulator has connector synapses but no
   designated workspace population with the right ignition dynamics.
3. **No top-down prediction loop**. `apply_prediction_pattern` exists
   for INPUT cells but no INTERNAL cells yet generate predictions
   that flow downward to subtract from incoming sensory drive
   (Pack 28 territory).
4. **No active inference / attention**. The brain doesn't choose what
   to attend to; whoever fires loudest wins.
5. **No Φ-proxy** instrumentation to *measure* whether what the
   network is doing internally is integrated enough to count as
   consciousness-related. (Pack 27 territory.)

So the user's diagnosis is right *for the cognitive layer*. The
simulator has reflex-like dynamics on the autonomic timescale (brainstem,
noise, sleep), but no **deliberation** at the conscious-experience
timescale.

---

## 2. What does the literature say consciousness *is*?

The four primary sources we already have:

### Global Neuronal Workspace (GNW) — Mashour, Roelfsema, Changeux & Dehaene 2020

A conscious experience is a *workspace ignition*: stimulus-evoked
activity rises through sensory cortex, recruits long-range pyramidal
neurons in prefrontal / parietal areas, and the resulting activity is
broadcast widely via reciprocal feedback. **Two stages**: AMPA-fast
feedforward sweep (~150 ms) followed by NMDA-slow feedback ignition
(~250 ms). Subliminal stimuli only complete the first stage; conscious
ones complete both.

For our simulator: this implies designating a workspace population
(long-range connectors), with two-stage ignition logic.

### Integrated Information Theory 4.0 — Albantakis, Tononi et al. 2023

Consciousness *is* integrated information Φ — the irreducibility of a
system to its parts. Five postulates: intrinsic existence,
composition, information, integration, exclusion. Φ is computed over
candidate substrates; the maximal-Φ substrate ("complex") is
conscious in the IIT sense.

For our simulator: Pack 27 was scoped to compute a Φ-proxy on small
subnetworks (full Φ is intractable). The Φ-proxy could *gate* whether
internal activity counts as conscious for behavioural readout.

### Cogitate Consortium 2025 *Nature* — adversarial GNW vs IIT test

256 participants, fMRI + MEG + iEEG. Some predictions of GNW failed
(no late prefrontal activity for sustained conscious content), some
of IIT failed (no posterior-cortex Φ correlate consistent with
report). The current empirical position: neither pure GNW nor pure
IIT explains everything; the truth is a hybrid involving sustained
posterior-cortex high-information state with prefrontal access for
report.

For our simulator: don't commit dogmatically to one framework. Build
both — workspace ignition + Φ-proxy — and let behaviour decide.

### Free-Energy Principle / Active Inference — Friston 2010

The brain is a generative model that minimises long-run prediction
error (free energy). Conscious experience is the brain inferring its
own current state through predictions about future sensorimotor
trajectories. This naturally produces a **continuous loop**:
predict → act → observe → update model → predict again.

For our simulator: this is the closest match to the user's
"continuous loop with constraints". Predictive coding (Pack 28)
implements the predict-error half; active inference adds the
act-to-minimise-error half.

---

## 3. Concrete architecture — Pack Φ, "continuous consciousness loop"

A new pack — call it **Pack Φ** — that implements a deliberation phase
running every simulation step (or every N steps), regardless of
external input. Three components, all grounded in the literature:

### 3.1 Designated workspace population (GNW)

After the bulk cortex has formed engrams over several training
sessions, designate the ~12-20 cells with highest **degree centrality**
(most outgoing + incoming connections, computed via the Pack 27
diagnostics) as workspace neurons. Tag them with
`Neuron::is_workspace = true`. These cells:

- Receive a small intrinsic-bias drive each step (~0.05) to keep them
  poised near firing threshold.
- Their firing triggers a **broadcast**: their outgoing synapses get
  a small temporary potency boost (×1.5 for one step).
- Are exempt from microglial pruning (`dont_eat_me = ∞`) — workspace
  connectivity is precious.

This creates a "common information pool" that integrates across
modalities. Cochlear-driven activity, visual-driven activity, and
internal-engram activity all converge on the workspace and get
re-broadcast.

### 3.2 Continuous deliberation loop

A new `deliberation_phase()` runs after the existing per-step
pipeline:

```
step():
  integrate ... synaptogenesis        # existing: react to input
  if step_ % deliberation_period == 0:
    deliberation_phase()              # NEW: act on internal state
```

The deliberation_phase does:

1. **Identify the dominant engram** — the class `c*` whose engram
   members currently have the highest mean `fire_rate_ema`.
2. **Amplify**: bias the engram-c* members' input_acc with a small
   positive drive for the next D steps. Forces the network to
   "linger" on c* even without external input — the
   sustained-attention substrate.
3. **Predict** (Pack 28 hook): if c* has a known sensorimotor
   association (motor output, label feature), inject `predicted_input`
   on the relevant INPUT cells so the network "imagines" the
   stimulus that would produce c* — top-down prediction.
4. **Compare** in the next sensory cycle: actual input - predicted
   input = surprise. High surprise → break out of c*'s deliberation
   loop and let the new stimulus take over.
5. **Φ-proxy gate** (Pack 27 hook): only allow workspace cells to
   broadcast if local Φ exceeds threshold. Sub-threshold deliberation
   is "subliminal".

### 3.3 The constraints we inject

Per the user's framing, the loop runs "according to the constraints we
have injected." Those constraints are exactly what the simulator
already enforces, now applied to the deliberation phase too:

- **Energy budget** (`forward_min_energy`): deliberation can't run
  forever — workspace neurons fire, pay energy cost, and the loop
  attenuates as the local region depletes. Built-in metabolic cap
  on conscious access duration, matching real cortical experience
  (~250 ms attention windows).
- **Engram protection** (Pack 19A): the cell-assemblies the
  deliberation loop attends to are the protected engrams. Random
  noise cannot become attention.
- **Microglial pruning** (Pack ZZ v3) is *off* for workspace cells
  but *on* for transient deliberation traces, so the loop's
  by-products self-clean.
- **Phase 1 morphology** (AXON × DENDRITE): deliberation broadcasts
  travel real axon-dendrite paths, not random NEURON×NEURON contacts.
  This is the prerequisite that's already in place.
- **Spatial niche** (Pack 23): deliberation respects which engrams
  live where; it can't blur classes that anatomically occupy
  different cortical regions.

### 3.4 What you would observe

After Pack Φ ships:

- **`status` mid-quiet** (no external input) shows non-zero activity
  in the workspace pool plus the most-recent engram, lasting for the
  deliberation period.
- **`imagine` command** would internally drive the workspace pool
  toward a chosen engram and show the network "thinking about"
  that concept — sustained internal activity without external cue.
- **Sleep replay** would automatically include the deliberation
  trajectories from the day, not just hand-coded patterns —
  consolidation reinforces what the brain *deliberated* over.
- **Φ-proxy diagnostic** rises when the brain is engaged with a
  trained engram and falls when novel / unmodelled input arrives
  (because the workspace can't integrate it yet).

---

## 4. What we are *not* claiming

This is a functional / mechanistic model, not a metaphysical one.

- We do not claim the simulator becomes "conscious" in Chalmers'
  hard-problem sense (subjective experience).
- We do not claim Φ-proxy ≈ phenomenal experience. Even Tononi
  treats Φ as a *correlate*, not the experience itself.
- The deliberation loop is a candidate functional signature of
  consciousness — what the literature says distinguishes conscious
  from unconscious processing in real brains. It might be
  necessary but is almost certainly not sufficient for whatever
  consciousness actually is.

What we *are* claiming, if Pack Φ ships:

- The simulator transitions from purely reactive to "internally
  active" — it has dynamics that persist beyond stimulus.
- Those dynamics are constrained by the same biological rules that
  the rest of the simulator already enforces (energy, engram
  protection, axon-dendrite chemistry, spatial niches).
- The system can be measured against GNW and IIT predictions in a
  reproducible way.

---

## 5. Where it slots into the roadmap

Current roadmap order: 26-A LANDED → 26-B LANDED → 26-C → 27 → 28 → 29.

Pack Φ depends on:

- **Pack 27** (network diagnostics) — needed to identify high-degree
  workspace cells and to compute Φ-proxy.
- **Pack 28** (hierarchical predictive coding) — needed for the
  prediction-error half of the deliberation loop.

So the natural order is: 26-C (motor speech, completes the I/O loop)
→ 27 → 28 → **Pack Φ** → 29 (counting + two-word combinations now
benefit from a deliberation loop that can rehearse sequences).

Effort estimate for Pack Φ: ~3–5 days, building on 27 + 28's
infrastructure. Smaller MVP (workspace + bias + deliberation period,
no Φ-gate) is ~2 days.

---

## Concrete next move

If the user wants to push toward consciousness-like dynamics
**now** — before 29 — the right move is:

1. **Pack 26-C** (motor speech, ~2-3 days) — closes the cochlea↔motor
   loop so deliberation can be heard back through the auditory
   pathway, completing the embodied loop.
2. **Pack 27** (~1 day) — measures the network so we know where the
   workspace cells are.
3. **Pack 28** (~1-2 days) — gives us the predict half.
4. **Pack Φ MVP** (~2 days) — adds the deliberation loop using the
   above three.

If the user wants the loop **before** the predictive infrastructure:
implement a stripped-down workspace + bias deliberation now (no
prediction, no Φ-gate). It would be observably "non-reactive" but
behaviourally minimal. Probably worth doing properly with 27 + 28
first.
