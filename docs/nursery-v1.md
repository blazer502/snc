# Developmental Multicenter SNC — v1 nursery prototype

The first runnable slice of the [developmental multicenter research
plan](developmental-multicenter-snc.md). It builds the plan's *minimal
publishable prototype* (§13): a small multicenter agent in a digital nursery that
learns visual↔language associations through **local learning on structurally
plastic inter-center pathways**, and measures where that modular, adaptive
connectivity helps — cross-modal grounding and continual learning — and where it
does not.

This is a developmental-layer experiment in Python (`python/dev_snc/`), kept
separate from the SNC substrate (`src/`, `python/snc/`) per the plan's Phase 0.
It reuses the substrate's ideas — sparse CSR connectivity, a slow structural
clock vs. a fast execution path, local plasticity — rather than its CUDA runtime,
which is a later scaling concern (plan Phase 5).

## What it is

Two specialized centers joined by two separately-learnable inter-center pathways:

```
              W_vl  (naming: local delta rule, per-group softmax)
  visual center ───────────────────────────────────▶ language center
   (fixed sparse    ◀───────────────────────────────   (word neurons)
    spiking code)   W_lv  (recall: local Hebbian co-activation)
```

- **Visual center** — a fixed random projection + k-winners-take-all gives a
  sparse spiking code of an object's features. It is *not* trained: the plan's
  thesis is that intelligence lives in the connectivity among centers, so the
  visual code is a reservoir and all learning happens on the pathways.
- **Language center** — one neuron per word (compositional: a color word block
  and a shape word block).
- **Pathways** — sparse, masked `(post × pre)` weight matrices updated by a
  local rule (pre-activity × post-signal). Each update also carries a `reward`
  factor for future neuromodulation, but v1 holds it at 1.0, so the rules are
  **two-factor** in practice (the reward/three-factor path is scaffolded, not yet
  exercised). On a slow clock, **structural plasticity** (a) grows synapses toward
  co-active neuron pairs, (b) prunes the weakest (protecting young synapses so they
  can mature first), and (c) **consolidates** matured synapses — freezing them
  against pruning and further change so later tasks cannot erode them (plan
  §6.2–6.3; SI/EWC-like). Growth is capped by how many co-active pairs exist, so
  the budget is not fixed: the pathway settles to a sparse, task-sustained floor.
  Connectivity is itself a learning result (plan RQ4).

Both pathways are trained on **every** teaching episode from the same
co-observation (an object is seen and named at once). They are independent
matrices — naming is discriminative, recall is an unsupervised associative memory
— so the agent grounds a word↔object association *bidirectionally* from shared
developmental experience with no task-level (retrieval-labeled) supervision. Recall
is genuine associative memory, **not** free transfer from the naming pathway to an
untrained direction: both directions are trained.

**Center metadata in the graph representation (plan step 3).** Each neuron carries
a `center` id parallel to the substrate's `role`/`sign`/`channel` arrays. This is
threaded into the C++ `SNNGraph` and the `snc_export` binary (a trailing,
backward-compatible `center[N]` block) and read back by `python/snc/graph.py`, so a
multicenter graph round-trips through the substrate's own format
(`MulticenterGraph.to_graph_dict` → `write_graph_bin` → `load_graph`).

**Digital nursery + body API (plan step 2).** `python/dev_snc/nursery.py` is a
deterministic 2D grid of attribute-bearing objects with a thin body API — `eyes`
(observe / look / focus / scene), `feet` (move / turn), `hands` (push), `speech`
(answer / ask), `memory`, and a `teacher` that names objects and gives reward
(every action returns True on success). Perception is object-centric for v1
(one-hot color + shape), which keeps the environment clean while still forcing a
*learned* code→word mapping.

## How to run

```bash
python3 python/train_nursery.py --selftest      # deterministic assertions (fast)
python3 python/train_nursery.py --demo --seeds 12 # narrate one agent + full suite
python3 python/train_nursery.py                  # the comparison tables below
```

## Results

Four conditions express the proposed system and its ablations (plan §9). Synapse
counts are shown next to accuracy so the *connectivity budget* is visible: the
proposed system is never given more synapses than the baselines it beats, so a win
cannot be attributed to extra parameters (plan §12.6). Means over 12 seeds.

**Cross-modal grounding** — the naming task: learn color+shape names from a subset
of combinations, then test naming (vision→word), generalization to held-out
combinations, and word→object recall (say a word, pick the matching object).

| condition | naming | generalization | recall | synapses |
|---|--:|--:|--:|--:|
| **proposed** (modular + structural) | **1.00** | 0.42 | **1.00** | **352** |
| weight-only (same start budget, no topology) | 0.81 | 0.19 | 0.88 | 400 |
| weight-only (matched final budget ≈310) | 0.76 | 0.10 | 0.71 | 310 |
| dense pathways (capacity ceiling) | 1.00 | 0.50 | 1.00 | 3584 |
| pathway-off (cross-modal lesion) | 0.08 | 0.06 | 0.33 | 400 |

**Continual learning** — class-incremental: objects are introduced a few classes
at a time; we re-test the earlier classes after all later ones. `retention` is
end-of-training accuracy on every class introduced before the final phase (many
objects, so low variance); `all_acc` is accuracy over all classes.

| condition | retention | all-class acc | synapses |
|---|--:|--:|--:|
| **proposed** (modular + structural) | **0.90 ± 0.07** | **0.91** | **314** |
| weight-only (same start budget, no topology) | 0.44 | 0.48 | 400 |
| weight-only (matched final budget ≈310) | 0.40 | 0.38 | 310 |
| dense pathways (capacity ceiling) | 1.00 | 1.00 | 6144 |
| pathway-off (cross-modal lesion) | 0.07 | 0.05 | 400 |

### What the numbers say

- **Cross-modal grounding is robust (RQ2/RQ5).** The proposed system names objects
  (1.00) and recalls the object for a word (1.00). Recall is associative memory in
  the reverse pathway, trained by co-activation on every episode — not transfer
  from the naming path (zeroing the reverse weights drops recall to chance). Both
  directions come from the same shared co-observation with no retrieval-labeled
  supervision. Lesioning the pathway floors naming to 0.08 (≈ joint chance 1/12)
  and recall to 0.33 (≈ analytic chance 2/7 ≈ 0.29): the competence lives in the
  inter-center connectivity.
- **Adaptive connectivity beats fixed sparsity at a lower budget (RQ4).** Structural
  growth/pruning routes synapses onto the *discriminative* visual neurons, so at
  **352/314 synapses** the proposed system reaches perfect naming/recall and 0.91
  continual accuracy — while a fixed **random**-sparse pathway of the *same or
  larger* budget (400) manages only 0.81 naming and 0.48 continual accuracy, and a
  matched-final random pathway (310) does worse still. Because structure wins with
  *fewer* synapses, the gain is from adaptive connectivity, not parameter count.
- **Consolidation is what buys retention.** Freezing matured synapses lifts
  continual accuracy from 0.73 (structure without consolidation) to 0.91 — the
  plan's slow-clock consolidation (§6.2–6.3) doing real work.
- **The dense pathway is the capacity ceiling, not a free win.** It reaches 1.00 on
  everything, but with ~10× (naming) to ~20× (continual) the synapses. Under a
  matched budget the advantage inverts. Note this baseline only densifies the two
  pathway masks; the centers and the fixed sparse visual reservoir are unchanged,
  so it is a dense-*pathway* ceiling, not a single homogeneous population.
- **Compositional generalization is weak — an honest limitation.** Naming *unseen*
  color×shape combinations sits at 0.42 (proposed) / 0.50 (dense), above the 0.06
  lesion floor and joint chance ≈0.08 but far from solved. v1 demonstrates
  associative grounding and efficient continual learning, not yet strong
  compositional generalization (plan §11 Risk 3). Closing it — a factored visual
  code / factored pathways — is future work.

## Mapping to the research questions

| RQ | v1 evidence |
|---|---|
| RQ2 — connectivity replaces forced unification | pathway lesion floors all cross-modal behavior; competence is in the pathways |
| RQ4 — structural plasticity as a first-class mechanism | grow/prune/consolidate beats fixed random sparsity on both tasks at fewer synapses; approaches the dense ceiling at ~1/10–1/20 the budget |
| RQ5 — one modality's learning grounds another | co-observation trains both directions; word→object recall works with no retrieval-labeled supervision |
| RQ1, RQ3 | partially exercised (single agent, object-centric nursery); left for later phases |

## Framing recommendation (plan step 8)

**Position v1 as an AI-architecture / developmental-learning contribution, with
the digital nursery as its benchmark — not (yet) as a systems-substrate paper.**

The differentiated, reproducible signals here are *learning* results: bidirectional
cross-modal grounding from shared experience, and structural plasticity (with
consolidation) as a connectivity-efficient continual-learning mechanism that beats
fixed sparsity at a lower budget. Those speak to the continual-learning /
neuro-inspired-architecture audience. The systems angle (sparse inter-center event
routing on CUDA) is genuinely interesting but is Phase 5 and not yet distinguished
from the existing substrate work, so it makes a stronger *second* paper once the
developmental workload is larger and irregular enough to stress the runtime.
Concretely: lead with the architecture + nursery benchmark; keep the
substrate/runtime paper as the follow-on.

## Limitations & next steps

- Two centers (visual, language) only; motor/spatial, auditory, memory, and reward
  centers are scaffolded in the metadata schema but not yet learning.
- Object-centric perception, not pixels; scripted teacher, not interactive.
- Reward/neuromodulation is scaffolded (held at 1.0); the rules are two-factor in v1.
- Weak compositional generalization (see above).
- Next: exercise reward in a reinforcement task; add a factored visual code; add the
  motor+spatial centers to reach the plan's navigation and object-permanence
  milestones (§7.3 stages 1–2); scale the nursery and move hot pathways onto the
  CUDA runtime (Phase 5).

Code: `python/dev_snc/` (`nursery`, `centers`, `plasticity`, `agent`, `tasks`,
`experiment`, `selftest`); CLI `python/train_nursery.py`.
