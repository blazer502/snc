# Developmental Multicenter SNC — sleep/replay consolidation ("learn the rule while sleeping")

The phase after [object permanence](nursery-permanence.md). It adds a sixth
center — **semantic memory** — and the mechanism that fills it: **sleep/replay
consolidation** (plan §6.2). The agent gathers *noisy* observations during the
day, keeps them as episodic traces, and during a sleep phase **replays** them into
a durable semantic store. Two things emerge that fast episodic memory cannot give:
the noise averages out (denoising), and the consolidated rule generalizes to
instances the agent never saw (abstraction) — the plan's "abstraction from repeated
episodes" (§5.2).

This completes the memory story and the two-timescale split (plan §6.2): the
[episodic memory](nursery-permanence.md) is the *fast* event store; semantic memory
is the *slow* consolidated store built from it during sleep.

## What it adds

- **Semantic memory center** (`memory.SemanticMemory`) — a Hebbian evidence vector
  over the visual code that predicts a binary object property. It is *not* written
  during experience; it is built by `consolidate(traces)` during sleep, replaying
  the episodic traces so evidence accumulates. Because the readout is linear over
  the attribute-structured visual code, the rule it forms generalizes across shape.
- **Sleep as a phase.** Interaction produces episodic traces; a separate replay
  pass consolidates them. Nothing about the day's raw labels is kept — only the
  consolidated evidence — so semantic memory is both durable and compact.

## How to run

```bash
python3 python/train_nursery.py --selftest        # includes consolidation assertions
python3 python/train_nursery.py --consol-seeds 12 # prints the table below
```

## Results

An object's property is determined by its colour (so it *can* be generalized
across shape). The agent sees the training types over several episodes with 30%
label noise, keeps the observations, then sleeps and replays. We test property
prediction on the seen types (denoising) and on held-out colour×shape combos
(abstraction). Means over 12 seeds.

| condition | seen types (denoise) | unseen combos (abstract) |
|---|--:|--:|
| **sleep-consolidated semantic** | **0.97** | **0.79** |
| episodic, aggregate at query (unbounded storage) | 0.91 | 0.54 |
| episodic, single transient trace | 0.66 | 0.54 |

Baselines: with 30% label noise a single episodic trace tops out at ≈0.70 on seen
types; aggregating *every* stored trace at query time also denoises (0.91) but needs
unbounded storage and still cannot help for a never-seen combo; for such a combo no
episodic scheme has any trace, so both sit at the 0.50 chance level.

### What the numbers say

- **Denoising comes from aggregation (0.97 / 0.91 vs 0.66).** A single noisy
  observation is right ~70% of the time; pooling repeated observations out-votes the
  noise. Both the consolidated store (0.97) and a query-time majority over every kept
  trace (0.91) recover the property — but semantic memory does it in a compact,
  durable weight vector rather than a growing log of raw episodes (plan §6.2
  "consolidate repeated patterns").
- **Abstraction is the signature of consolidation (0.79 vs 0.54).** The consolidated
  rule predicts the property of colour×shape combinations the agent *never observed*
  — because replay built a colour-determined rule over the shared visual code, not a
  lookup table of instances. *No* episodic scheme can do this: with no trace for an
  unseen combo, every episodic baseline is at chance (0.54). Generalizing to novel
  instances is what semantic memory uniquely buys (plan §5.2 "reusable abstractions").
- **Six centers, two timescales.** Perception, language, space, motor, episodic and
  now semantic memory each do one job; and the agent now has both a fast event
  memory (episodic) and a slow consolidated memory (semantic) built from it during
  sleep — the plan's medium/slow timescale separation (§6.2).

## Mapping to the plan

| plan element | v-phase evidence |
|---|---|
| §6.2 slow timescale: replay past episodes during sleep-like phases | replay lifts seen-type accuracy 0.66 → 0.97 |
| §5.2 semantic memory: consolidated concepts, abstraction from repeated episodes | held-out-combo accuracy 0.79 vs 0.54 episodic |
| RQ4 — structural/consolidation as a first-class learning mechanism | consolidation, not per-step supervision, produces the durable rule |

## Limitations & next steps

- The property is a single colour-determined bit; multi-attribute rules,
  multi-class properties, and rules that mix attributes are the honest harder cases.
- Semantic memory is a separate readout, not yet consolidated into the *structural*
  pathways themselves; folding replay into structural growth (grow synapses for
  repeatedly-replayed associations) would unify consolidation with the substrate's
  slow structural clock. **Update:** this is now built — see
  [nursery-structural-consolidation.md](nursery-structural-consolidation.md)
  (replay-then-prune keeps the discriminative synapses — beating random sparsity at
  equal budget and approximating the dense readout's effective wiring).
- Sleep is a single post-hoc phase; interleaving short replay phases during
  interaction (and measuring forgetting across many "days") is the natural
  continual-learning extension.

Code: `python/dev_snc/memory.py` (`SemanticMemory`), `tasks.run_consolidation`,
`experiment.run_consolidation_suite`; CLI `python/train_nursery.py --consol-seeds N`.
