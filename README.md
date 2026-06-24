# SNC — Structure-Aware Spiking Neural Computing

SNC is a **structure-aware spiking neural computing substrate**. Instead of
treating SNN connectivity as a fixed adjacency matrix, SNC represents a compact
structural substrate where neuron bodies, axons, dendrites, and synaptic
contacts define **sparse, local connectivity**. The resulting graph is executed
by event-driven **CPU / OpenMP / CUDA** runtimes, and trained either by local
learning (**e-prop**) or by surrogate-gradient **BPTT** through a PyTorch bridge.
SNC separates slow, brain-inspired *structural plasticity* from the fast spiking
*execution* path, so sparse topology can evolve while inference and training stay
efficient.

The research question:

> Can brain-inspired structural constraints (locality, morphology, conduction
> delays, structural plasticity) produce sparse SNN topologies that improve
> **efficiency** without losing **accuracy** on real learning tasks?

SNC grew out of a brain-development simulator; that model is now the substrate's
slow structural layer and is documented under **Background** below.

## Key results

Controlled, matched-budget, multi-seed studies (full details in [`docs/`](docs/README.md)):

| finding | evidence |
|---|---|
| **Structure-aware sparsity > random** at equal budget; gap widens as budget tightens (+4 to +38 pts) and as the learner weakens | MNIST, [structural-advantage](docs/structural-advantage.md) |
| **Dynamic plasticity > static** in the scarce regime (+13–18 pts at ~2k synapses); neutral when budget is ample | MNIST co-training |
| **Structure ≈ dense at ~6× fewer synapses** under BPTT (0.975 vs 0.976); sparsity *enables* widths dense can't fit | MNIST BPTT, [LM scaling](docs/llm-direction.md) |
| **Match the prior to the task's axis:** spatial *locality* helps vision but *hurts* audio; *conduction delays* help audio (+3.4 pts) where locality fails | SHD, [experiments-shd](docs/experiments-shd.md) |
| **3 CUDA delivery backends** (atomic / bucket / sort), parity-exact vs CPU; bucket ~2.2× under heavy contention; GPU minibatch e-prop ~15× faster than CPU | [architecture](docs/architecture.md) |
| **Spiking char-LM** trains on the substrate (next-token BPTT), well below the unigram baseline | [llm-direction](docs/llm-direction.md) |

The honest one-line thesis: *brain-inspired structure is a real inductive bias,
not a free lunch — it buys accuracy-per-synapse and feasibility-at-scale in the
budget-constrained, locally-learned, structurally-matched regime, and is roughly
neutral as a drop-in for a dense BPTT-trained layer.*

## Quick start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j                       # add -DSNC_ENABLE_CUDA=ON for GPU backends

# Benchmark: build a sparse graph, encode, run, report (CPU/OpenMP/CUDA)
./build/snc_bench --self-test
./build/snc_bench --structure static-snc --encoder poisson --num-steps 50 --num-samples 64
./build/snc_bench --compare-backends         # (CUDA build) parity + speedup across 5 backends

# Train frozen-structure weights with local e-prop (CPU or GPU)
./build/snc_train --structure static-snc --epochs 12          # synthetic
./scripts/fetch_mnist.sh
./build/snc_train --dataset mnist --data-dir data/mnist --device cuda --batch 32 --lr 1.0

# Two-timescale co-training (grow / prune / rewire on a slow clock)
./build/snc_cotrain --dataset mnist --data-dir data/mnist --grow 200 --structural-budget 3000

# PyTorch bridge: surrogate-gradient BPTT on the SNC topology (see python/README.md)
python3 python/train.py     --structure static-snc --hidden 256 --device cuda   # MNIST
python3 python/train_lm.py  --layers 512 --structure static-snc                 # char-LM
./scripts/fetch_shd.sh && python3 python/train_shd.py --rec-structure random-sparse --delay-max 30
```

## Repository layout

```
include/snc/     substrate headers — snn_graph, encoders, runtime, dataset,
                 trainer, cuda_trainer, connectome
include/         original engine — brain_grid, neuron, simulator, energy_field
src/graph/       SNNGraph (CSR) + dense/random/static-snc generators + compiler
src/runtime/     cpu/openmp backend + CUDA backends (atomic/bucket/sort)
src/structure/   mutable connectome — grow / prune / rewire
src/training/    encoders, datasets, e-prop trainer (CPU + CUDA), train/cotrain CLIs
src/bench/       snc_bench (benchmark), snc_export (graph -> PyTorch bridge)
src/*.cpp        original brain engine + demos (chat, vocab, ...)
python/          PyTorch bridge — graph loader + SNN / LM / SHD models (BPTT)
scripts/         data fetch (mnist, shd) + experiment sweeps + aggregators
docs/            architecture, experiments, analysis, plan, brain-model
```

## Documentation

Start at the **[documentation index](docs/README.md)**. The map:

- [`docs/architecture.md`](docs/architecture.md) — substrate design: CSR graph, the
  three-layer separation, execution model, all backends, the training paths.
- [`docs/new-plan.md`](docs/new-plan.md) — the repositioning plan and roadmap.
- [`docs/experiments-mnist.md`](docs/experiments-mnist.md) — MNIST study (structure, depth, e-prop vs BPTT), error-barred.
- [`docs/experiments-shd.md`](docs/experiments-shd.md) — SHD spiking-audio (locality fails, delays win).
- [`docs/structural-advantage.md`](docs/structural-advantage.md) — *when* structure helps + the accuracy-vs-synapse frontier.
- [`docs/llm-direction.md`](docs/llm-direction.md) — recurrent spiking language model + scaling.
- [`docs/related-work.md`](docs/related-work.md) — positioning vs DeepR/RigL, e-prop, sparse/plastic SNNs.
- [`python/README.md`](python/README.md) — the PyTorch bridge.

## Background: the brain-development model

The structural layer is generated by a biologically-grounded brain-development
engine (`BrainGrid` / `Simulator`): a 2-bit voxel grid, per-cell-type 3D
morphology, axon-dendrite synaptogenesis, three-factor learning, microglial
pruning, engram allocation, and sleep consolidation — grounded in primary
literature. That model, the original computing engine, and the pack-by-pack
history are documented in **[`docs/brain-model.md`](docs/brain-model.md)**, with
the long-form roadmap in [`docs/ROADMAP.md`](docs/ROADMAP.md) and the morphology
and consciousness notes in [`docs/MORPHOLOGY_REFACTOR.md`](docs/MORPHOLOGY_REFACTOR.md)
and [`docs/DISCUSSION_CONSCIOUSNESS.md`](docs/DISCUSSION_CONSCIOUSNESS.md). The
original demos still build (`snc_demo`, `snc_chat`, ...).

## About the author

**Chanyoung Park** — author of SNC, currently **open to work**.

- GitHub: [@blazer502](https://github.com/blazer502)
- Email: [ppoo1220@gmail.com](mailto:ppoo1220@gmail.com)
- Google Scholar: [Chanyoung Park](https://scholar.google.com/citations?user=zajmzf4AAAAJ)

If you're hiring for computational / theoretical neuroscience, biologically
grounded AI architectures, or neuromorphic / systems engineering, please reach out.

## Notes

- The CUDA backends and the GPU trainer require `-DSNC_ENABLE_CUDA=ON` (nvcc + a
  device); everything falls back to CPU/OpenMP cleanly when CUDA is absent.
- The PyTorch bridge needs `torch` + `numpy` (+ `h5py` for SHD). Datasets are
  fetched by `scripts/fetch_*.sh` into the gitignored `data/`.
- Structural-layer engine note: `(X * Y) % 32 == 0` is required so z-slice
  parallelism never updates bits in the same packed word; save format is `SNC11`.
