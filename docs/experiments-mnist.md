# Full-MNIST Multi-Seed Study

Rigorous version of the two headline questions from `new-plan.md` (Experiments 1
and 2), run on the **full MNIST set** with error bars over seeds. Reproduce with:

```bash
./scripts/fetch_mnist.sh
DEVICE=cuda SEEDS="1 2 3 4 5" ./scripts/run_mnist_study.sh   # 35 runs, GPU; ~4 min
python3 scripts/aggregate_study.py                           # mean +/- std table
```

## Setup

| | |
|---|---|
| Data | full MNIST — 60,000 train / 10,000 t10k test |
| Network | 784 inputs → 256 hidden → 10 outputs (one hidden layer) |
| Encoding | Poisson rate, 20 timesteps |
| Learning | e-prop (local, no global backprop); spike-count readout, cross-entropy |
| Schedule | 12 epochs (Exp 1) / 12 outer rounds × 1 inner epoch (Exp 2) |
| Seeds | 5 (1–5); reported as best-test mean ± population std |
| Hardware | GPU minibatch e-prop (`--device cuda --batch 32 --lr 1.0`), 4× RTX A6000 |

The whole 35-run matrix completes in **~4 min on GPU** (vs ~11 min CPU for fewer
seeds); GPU minibatch SGD reaches the same accuracy as the per-sample CPU path
(verified within ~0.5%). Accuracies are below conv-SNN / BPTT state of the art —
expected for a single fully-connected hidden layer with rate coding and local
learning. The contribution is the **controlled comparison under a matched
synapse budget**, not an absolute leaderboard number.

Accuracies are below conv-SNN / BPTT state of the art — that is expected for a
single fully-connected hidden layer with rate coding and local learning. The
contribution is the **controlled comparison under a matched synapse budget**,
not an absolute leaderboard number.

## Experiment 1 — does brain-like structure beat random sparsity?

Frozen topology, weights trained by e-prop. `random-sparse` and `static-snc`
share a 40k-synapse budget; `dense` is a fully-connected higher-budget reference.

| structure | synapses | best test acc | energy¹ (rel.) |
|---|---:|---|---:|
| dense (reference) | 203,264 | 0.9581 ± 0.0016 | 1.00 |
| random-sparse | 39,914 | 0.8979 ± 0.0025 | 0.23 |
| **static-snc** | **32,352** | **0.9403 ± 0.0017** | **0.21** |

**Result.** The morphology-constrained `static-snc` topology beats `random-sparse`
by **+4.2 points (0.940 vs 0.898)** — while using *fewer* synapses (32k vs 40k)
and slightly less energy. Error bars are tight and non-overlapping across all 5
seeds. It recovers most of the dense accuracy (0.940 vs 0.958) at **~6× fewer
synapses and ~4.7× lower energy**. Brain-inspired local connectivity is a
genuinely better sparse prior than random sparsity at equal cost.

¹ Energy proxy = spikes + 0.2·synaptic-events accumulated over a training epoch,
normalised to dense (representative seed). Lower is better.

## Experiment 2 — does dynamic structure beat static structure?

Two-timescale co-training (`snc_cotrain`): inner-loop e-prop weight training,
outer-loop budget-constant grow/prune/rewire. `static` = `--grow 0`. Both modes
see the same budget and the same number of epochs.

| synapse budget | static best test | dynamic best test | Δ |
|---:|---|---|---:|
| 2,080 (very tight) | 0.5286 ± 0.0097 | **0.6588 ± 0.0092** | **+0.130** |
| 5,200 (looser)     | 0.8167 ± 0.0055 | 0.8201 ± 0.0069 | +0.003 |

**Result.** At a very tight budget, dynamic rewiring beats static structure by
**+13.0 points (0.659 vs 0.529)**, with non-overlapping error bars across 5
seeds. At the looser budget the two are tied. Structural plasticity pays off
precisely when synapses are scarce and must be allocated well; once the budget
is comfortable, a fixed local topology already suffices and rewiring churn adds
nothing.

## Experiment 3 — depth: does it help, is the structural gap robust?

Frozen structure, e-prop with direct feedback alignment (every hidden neuron
gets its own random feedback row from the output error, so the rule works for
any number of hidden layers). Hidden width 256 per layer; the **synapse budget
is fixed at 40k across all depths**, so deeper = budget spread thinner.
Reproduce: `./scripts/run_depth_sweep.sh && python3 scripts/aggregate_depth.py`.

| hidden layers | static-snc | random-sparse | gap |
|---|---|---|---:|
| 1 (256)             | 0.9409 ± 0.0017 | 0.8980 ± 0.0031 | **+0.043** |
| 2 (256,256)         | 0.9257 ± 0.0012 | 0.8868 ± 0.0053 | **+0.039** |
| 3 (256,256,256)     | 0.9079 ± 0.0011 | 0.8673 ± 0.0008 | **+0.041** |

**Result.** Two things. (1) **Depth does not help in this regime** — accuracy
declines with depth for both topologies (static-snc 0.941 → 0.926 → 0.908). That
is expected: at a fixed budget each added layer thins the per-layer fan-out, and
direct feedback alignment is known to degrade with depth. (2) **The structure-
aware advantage is robust to depth** — static-snc beats random-sparse by ~+4
points at *every* depth, with non-overlapping error bars across 3 seeds. The
morphology-constrained prior helps regardless of how deep the graph is.

Pushing depth to actually *help* would need a stronger credit-assignment rule
(surrogate-gradient BPTT, Phase 5) and/or a budget that grows with depth — both
orthogonal to the matched-budget structure comparison.

## Experiment 4 — surrogate-gradient BPTT (PyTorch bridge, Phase 5)

The C++ trainers use local learning (e-prop + direct feedback alignment). The
PyTorch bridge (`python/`) instead trains the **exact same SNC topology** with
true surrogate-gradient backprop-through-time: the LIF recurrence runs in torch
ops over the exported edge list, the spike is a custom autograd Function
(Heaviside forward, fast-sigmoid surrogate backward), and PyTorch unrolls the
T-step loop. Full MNIST, Adam, 15 epochs, single seed (`python3 python/train.py`).

Best test acc (max over epochs), single seed:

| structure | synapses | 1 hidden | 2 hidden | 3 hidden |
|---|---:|---|---|---|
| dense (ref)   | 203,264 | 0.9776 | — | — |
| static-snc    | 32,352 | **0.9767** | 0.9737 | 0.9707 |
| random-sparse | 39,914 | 0.9689 | 0.9693 | 0.9651 |

Four results, compared against the local-learning runs above:

1. **BPTT ≫ e-prop.** At 1 hidden layer, surrogate-BPTT lifts static-snc 0.940 →
   **0.977** and random-sparse 0.898 → **0.969** — the sparse SNC net reaches
   ~97.7% on MNIST.
2. **Structure-aware sparsity ≈ dense, at 6× fewer synapses.** static-snc (32k
   syn) hits 0.9767 vs dense (203k syn) 0.9776 — within 0.1 pt at ~6× lower
   synapse cost. The morphology prior is a near-free sparsification under BPTT.
3. **Depth no longer hurts.** Under e-prop/DFA accuracy fell with depth
   (static-snc 0.941 → 0.908, Exp 3); under BPTT it is nearly flat
   (0.977 → 0.971). BPTT removes the depth penalty — though depth still does not
   *help* on MNIST, which a one-hidden-layer net already solves at ~97.7%.
4. **The structural advantage shrinks as the optimizer strengthens.** static-snc
   beats random-sparse by ~+4 pts under e-prop but only ~+0.8 pts under BPTT: a
   powerful learner partly compensates for a weaker topology. The morphology
   prior matters most when learning is local/cheap — a genuinely interesting
   interaction, not a pure win for either.

(Single seed. The PyTorch path supports delay-1 graphs; see `python/README.md`.)

## Takeaways

1. **Structure-aware sparsity > random sparsity** at equal (indeed lower) cost:
   +4.2 pts on full MNIST under local learning, robust across depth (~+4 pts at
   1–3 hidden layers). The gap narrows to ~+0.8 pts under surrogate-BPTT.
2. **Dynamic structure > static** in the scarce-synapse regime: +13.0 pts at
   ~2k synapses; neutral when synapses are plentiful.
3. **Learning rule matters more than depth here:** surrogate-BPTT reaches ~0.977
   (vs ~0.940 for e-prop) and removes the depth penalty, but extra depth does not
   help on MNIST — and static-snc nearly matches dense at ~6× fewer synapses.

Both are *controlled, multi-seed, full-dataset* results, consistent with the
earlier sub-sampled runs.

## Limitations / next

Rate coding, 12 epochs, one hyper-parameter setting per config (no per-config
tuning), and local learning (e-prop + direct feedback alignment) — which is why
depth does not yet pay off (Exp 3). Training runs on the GPU (minibatch e-prop),
so epochs/depth are not the bottleneck. Natural extensions: surrogate-gradient
BPTT (to make depth productive), latency and event-based encodings, and a
depth-scaled synapse budget. These do not change the matched-budget comparisons
above.
