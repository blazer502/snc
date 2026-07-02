# Developmental Multicenter SNC — structural consolidation ("consolidate into the wiring")

A follow-on to [sleep/replay consolidation](nursery-consolidation.md). That phase
replayed episodic traces into a *dense* semantic readout. This one folds the same
replay into the substrate's **slow structural clock**: replay accumulates evidence,
then **developmental pruning** keeps only the most-evidenced synapses (plan §6.3
"prune unused connections during consolidation"), so the consolidated rule lives in
durable **sparse structure**, not a dense weight blob. It closes the loop with the
project's core finding — *structure is connectivity-efficient* (see
[structural-advantage.md](structural-advantage.md) and
[nursery-v1.md](nursery-v1.md)) — now for memory consolidation.

## What it does

Same property-learning setup as the previous phase (a colour-determined property,
30% label noise, held-out combos). Replay builds a Hebbian evidence vector over the
visual code (`SemanticMemory.consolidate`), then a single structural step
(`SemanticMemory.prune(budget)`) keeps the `budget` synapses with the strongest
accumulated evidence and drops the rest. Evidence magnitude is the keep/grow signal
— the sleep-time analogue of the substrate's co-activation-driven synaptogenesis.

Three readouts of the *same* consolidated evidence are compared:

- **dense** — one synapse per visual neuron. The physical readout allocates 256,
  but because the 12-active code spans only 12 object types, only ~58 neurons ever
  carry nonzero weight; those ~58 are its *effective* wiring (the zero-weight rest
  contribute nothing), so we compare against that count.
- **structural** — pruned to the 16 most-evidenced synapses;
- **random-sparse** — 16 synapses kept at random (same evidence weights), averaged
  over several masks.

## How to run

```bash
python3 python/train_nursery.py --selftest         # includes structural-consolidation assertions
python3 python/train_nursery.py --consol-seeds 12  # prints this table under the consolidation one
```

## Results

Means over 24 seeds.

| condition | seen types | unseen combos | synapses |
|---|--:|--:|--:|
| dense readout (effective wires) | 0.94 | 0.81 | ~58 |
| **structural (evidence-pruned)** | **0.91** | **0.76** | **16** |
| random-sparse (same budget) | 0.75 | 0.64 | 16 |

### What the numbers say

- **Structure beats random sparsity at the same budget — the robust result.** At 16
  synapses, evidence-based pruning keeps the discriminative wires and holds 0.91 seen
  / 0.76 unseen, while the *same 16* chosen at random miss them and collapse to
  0.75 / 0.64. Keeping the most-evidenced synapses is what makes compression work.
  This is the plan's RQ4 (structural plasticity as a first-class mechanism) and the
  same result the v1 continual-learning study found — evidence/co-activation-guided
  sparse structure beats fixed random sparsity of equal budget.
- **Consolidation into structure ≈ the dense readout at ~3.6× fewer wires.** The
  dense readout only ever uses ~58 wires (the rest are zero-weight); pruning to 16
  keeps most of its accuracy — a small, consistent drop of ~0.04 (seen 0.94→0.91,
  unseen 0.81→0.76), never a gain. The durable rule compresses into a handful of
  synapses, though not for free.
- **Consolidation now lives on the slow structural clock.** The fast episodic store,
  the sleep replay, and the slow structural pruning are the plan's three timescales
  (§6.2–6.3) end to end: experience → replay → durable sparse wiring.

## Mapping to the plan

| plan element | evidence |
|---|---|
| §6.3 prune unused connections during consolidation | ~58 → 16 effective synapses at a ~0.04 accuracy cost |
| RQ4 — structural plasticity / connectivity as the learned object | evidence-pruned 16 beats random 16 (0.91 vs 0.75 seen) |
| §6.2 three timescales | episodic (fast) → replay (sleep) → structural pruning (slow) |

## Limitations & next steps

- Consolidation here is prune-after-dense: evidence is accumulated densely, then
  pruned once. Growing synapses *from empty* during replay (true replay-driven
  synaptogenesis on the live `SparseConn`) is the faithful next step — a naive
  delta-rule-over-growth version was unstable, so it needs the same young-synapse
  protection / consolidation the structural pathways already use.
- The property is a single colour-determined bit; multi-attribute and
  multi-class rules are the honest harder cases.
- The consolidated structure is a standalone pathway; folding it into the live
  agent's inter-center pathways (so consolidated semantic knowledge feeds naming and
  recall) would connect this back to the multicenter agent.

Code: `python/dev_snc/memory.py` (`SemanticMemory.prune`),
`tasks.run_structural_consolidation`,
`experiment.run_structural_consolidation_suite`; CLI
`python/train_nursery.py --consol-seeds N`.
