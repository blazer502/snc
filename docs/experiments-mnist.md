# Full-MNIST Multi-Seed Study

Rigorous version of the two headline questions from `new-plan.md` (Experiments 1
and 2), run on the **full MNIST set** with error bars over seeds. Reproduce with:

```bash
./scripts/fetch_mnist.sh
./scripts/run_mnist_study.sh            # 21 runs, parallel; ~11 min on 64 cores
python3 scripts/aggregate_study.py      # mean +/- std table
```

## Setup

| | |
|---|---|
| Data | full MNIST — 60,000 train / 10,000 t10k test |
| Network | 784 inputs → 256 hidden → 10 outputs (one hidden layer) |
| Encoding | Poisson rate, 20 timesteps |
| Learning | e-prop (local, no global backprop); spike-count readout, cross-entropy |
| Schedule | 10 epochs (Exp 1) / 10 outer rounds × 1 inner epoch (Exp 2) |
| Seeds | 3 (1, 2, 3); reported as best-test mean ± population std |
| Hardware | CPU only (single-threaded per run, 21 runs in parallel) |

Accuracies are below conv-SNN / BPTT state of the art — that is expected for a
single fully-connected hidden layer with rate coding and local learning. The
contribution is the **controlled comparison under a matched synapse budget**,
not an absolute leaderboard number.

## Experiment 1 — does brain-like structure beat random sparsity?

Frozen topology, weights trained by e-prop. `random-sparse` and `static-snc`
share a 40k-synapse budget; `dense` is a fully-connected higher-budget reference.

| structure | synapses | best test acc | energy¹ (rel.) |
|---|---:|---|---:|
| dense (reference) | 203,264 | 0.9604 ± 0.0011 | 1.00 |
| random-sparse | 39,914 | 0.9022 ± 0.0029 | 0.24 |
| **static-snc** | **32,352** | **0.9427 ± 0.0021** | **0.22** |

**Result.** The morphology-constrained `static-snc` topology beats `random-sparse`
by **+4.0 points (0.943 vs 0.902)** — while using *fewer* synapses (32k vs 40k)
and slightly less energy. Error bars are tight and non-overlapping across all 3
seeds. It recovers most of the dense accuracy (0.943 vs 0.960) at **~6× fewer
synapses and ~4.6× lower energy**. Brain-inspired local connectivity is a
genuinely better sparse prior than random sparsity at equal cost.

¹ Energy proxy = spikes + 0.2·synaptic-events accumulated over a training epoch,
normalised to dense (representative seed). Lower is better.

## Experiment 2 — does dynamic structure beat static structure?

Two-timescale co-training (`snc_cotrain`): inner-loop e-prop weight training,
outer-loop budget-constant grow/prune/rewire. `static` = `--grow 0`. Both modes
see the same budget and the same number of epochs.

| synapse budget | static best test | dynamic best test | Δ |
|---:|---|---|---:|
| 2,080 (very tight) | 0.5563 ± 0.0089 | **0.6649 ± 0.0047** | **+0.109** |
| 5,200 (looser)     | 0.8201 ± 0.0017 | 0.8186 ± 0.0038 | −0.002 |

**Result.** At a very tight budget, dynamic rewiring beats static structure by
**+10.9 points (0.665 vs 0.556)**, with non-overlapping error bars across 3
seeds. At the looser budget the two are tied. Structural plasticity pays off
precisely when synapses are scarce and must be allocated well; once the budget
is comfortable, a fixed local topology already suffices and rewiring churn adds
nothing.

## Takeaways

1. **Structure-aware sparsity > random sparsity** at equal (indeed lower) cost:
   +4.0 pts on full MNIST, tight error bars.
2. **Dynamic structure > static** in the scarce-synapse regime: +10.9 pts at
   ~2k synapses; neutral when synapses are plentiful.

Both are *controlled, multi-seed, full-dataset* results, consistent with the
earlier sub-sampled runs.

## Limitations / next

Single hidden layer, rate coding, 10 epochs, one hyper-parameter setting per
config (no per-config tuning), CPU e-prop (so depth/epochs are compute-bound).
Natural extensions: deeper / multi-layer graphs, latency and event-based
encodings, longer schedules, and CUDA-accelerated training to push beyond the
current epoch budget. These do not change the matched-budget comparisons above.
