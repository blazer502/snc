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

## Takeaways

1. **Structure-aware sparsity > random sparsity** at equal (indeed lower) cost:
   +4.2 pts on full MNIST, tight error bars.
2. **Dynamic structure > static** in the scarce-synapse regime: +13.0 pts at
   ~2k synapses; neutral when synapses are plentiful.

Both are *controlled, multi-seed, full-dataset* results, consistent with the
earlier sub-sampled runs.

## Limitations / next

Single hidden layer, rate coding, 12 epochs, one hyper-parameter setting per
config (no per-config tuning). Training now runs on the GPU (minibatch e-prop),
so epochs/depth are no longer the bottleneck. Natural extensions: deeper /
multi-layer graphs, latency and event-based encodings, and longer schedules.
These do not change the matched-budget comparisons above.
