# SNC Architecture: Structure-Aware SNN Substrate

This document describes the runtime substrate added on top of the original
brain-development simulator. It implements the first slice of
[`new-plan.md`](new-plan.md): reposition SNC as a **structure-aware spiking
neural computing substrate** that separates slow structural evolution from fast
spike execution.

```
[Structural Layer]   BrainGrid + Simulator + Neuron   (existing)
  3D voxel grid, morphology, growth/pruning/synaptogenesis, energy field
        |
        |  compile  (src/graph/snn_graph.cpp : compile_from_simulator)
        v
[Spiking Graph Layer]   SNNGraph                       (new)
  immutable CSR over outgoing synapses + per-neuron role/sign/channel
        |
        |  execute  (src/runtime/*)
        v
[Runtime Layer]   cpu / openmp / cuda backends         (new)
  LIF integrate-fire, delayed sign-aware delivery, spike-count decode
```

Guiding principle (new-plan.md §2): **the structural layer is not on the
critical path of every spike timestep.** Structure mutates on a slow epoch
clock; between epochs the compiled CSR graph is executed many times by a
backend.

## Modules

| Path | Role |
|------|------|
| `include/snc/snn_graph.hpp`, `src/graph/snn_graph.cpp` | `SNNGraph` (CSR), generators (`make_dense`, `make_random_sparse`, `make_static_snc`), `compile_from_simulator`, `compute_stats`, `validate` |
| `include/snc/encoders.hpp`, `src/training/encoders.cpp` | `InputEvent` + `direct` / `poisson` / `latency` encoders |
| `include/snc/runtime.hpp`, `src/runtime/cpu_backend.cpp` | `Backend` enum, LIF `forward()` over CSR, CPU + OpenMP |
| `src/runtime/cuda_backend.cu` | Stage-1 atomic CUDA forward (built with `-DSNC_ENABLE_CUDA=ON`) |
| `src/runtime/cuda_stub.cpp` | no-op `cuda::*` symbols for non-CUDA builds |
| `include/snc/dataset.hpp`, `src/training/datasets.cpp` | `Dataset` + synthetic generator + MNIST IDX loader |
| `include/snc/trainer.hpp`, `src/training/trainer.cpp` | frozen-structure **e-prop** weight training |
| `include/snc/connectome.hpp`, `src/structure/connectome.cpp` | mutable position-aware connectome + grow/prune/rewire |
| `src/bench/snc_bench.cpp` | `snc_bench` CLI: build graph, encode, run, report |
| `src/training/train_main.cpp` | `snc_train` CLI: train frozen-topology weights, log per epoch |
| `src/training/cotrain_main.cpp` | `snc_cotrain` CLI: two-timescale structure+weight co-training |

## The compiled graph (CSR)

`SNNGraph` stores outgoing synapses in Compressed-Sparse-Row form. Synapse `s`
in `[row_ptr[i], row_ptr[i+1])` is an outgoing edge of neuron `i`:

```
post_ids[s]   weights[s]   delays[s]   branch_ids[s]
```

Per-neuron metadata is parallel to `[0, num_neurons)`: `role` (INTERNAL /
INPUT / OUTPUT), `sign` (+1 / -1, Dale's principle carried on the pre neuron),
and `channel` (input feature or output class). This is the "compiled artefact"
of new-plan.md §3.2 — runtimes consume it read-only and own no plasticity.

## Execution model

One forward pass simulates `num_steps` of leaky integrate-and-fire:

1. **Inject** the step's external drive (from the encoder's `InputEvent`s).
2. **Integrate + fire**: `v = v*decay + ext + ring[t]`; fire when `v >=
   threshold`; reset + refractory.
3. **Decode**: output-neuron firings increment per-class logits.
4. **Expand**: each fired neuron's outgoing synapses deposit `sign*weight` into
   a **delay ring buffer** at slot `(t + delay) % ring_len`, where
   `ring_len = max_delay + 1` (so a delivery never lands on the slot being read
   and cleared this step).

Backends share this model:

- **cpu** — single-thread, deterministic; the correctness reference.
- **openmp** — parallel integrate/fire; atomic ring delivery. Spike counts
  match cpu. *Note:* for small graphs the per-step fork/join + atomics make it
  slower than single-thread cpu; the parallel win is the CUDA path.
- **cuda-atomic** — Stage 1 of new-plan.md §4.2: every fired neuron `atomicAdd`s
  its weighted spikes into the device ring.
- **cuda-bucket** — Stage 2: delivery is sharded across `kShards` copies of the
  ring by source neuron, so atomic contention on any one address drops ~`kShards`-
  fold; the shards are reduced into the ring each step.
- **cuda-sort** — Stage 3: fired neurons' synapses are expanded into
  (ring-index key, signed value) events, sorted by key and segment-reduced
  (Thrust `sort_by_key` + `reduce_by_key`), then scattered into the ring —
  fully atomic-free.

All three are validated to produce **the same spike/event counts as cpu**. When
a `cuda-*` backend is requested but unavailable (built without CUDA, or no
device), `forward()` transparently falls back to cpu and reports the actual
backend in `ForwardResult::used_backend`.

### Backend throughput (RTX A6000, parity exact across all)

The reduction strategies trade overhead for contention relief, so which wins
depends on the regime:

| regime | cpu | cuda-atomic | cuda-bucket | cuda-sort |
|---|---|---|---|---|
| moderate fan-in (static-snc, 120k syn) | 1.0× | **1.9×** | 1.6× | 0.3× |
| heavy contention (dense, 1024→32, full drive) | 1.0× | 1.6× | **2.2×** | 0.3× |

Bucketing wins when many sources hammer few high-fan-in targets (atomic
contention dominates); at moderate contention the shard-reduction overhead makes
plain atomic faster. Sort is contention-free but sort overhead dominates at
these graph sizes — it is the strategy for extreme fan-in / very large event
counts, not the default.

**Locality (Exp. 5).** At an equal 120k-synapse budget, the morphology-
constrained `static-snc` graph sustains ~7.0e8 synaptic-events/s vs ~6.6e8 for
`random-sparse` (+6%) on cuda-atomic, and emits fewer events overall — a modest
memory-locality benefit from spatially-clustered targets.

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target snc_bench -j

# With the CUDA backend (requires nvcc + a device):
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DSNC_ENABLE_CUDA=ON
cmake --build build-cuda --target snc_bench -j
```

## Running

```bash
# Validate the structural -> CSR compiler:
./build/snc_bench --self-test

# Forward inference on a structure-aware sparse graph:
./build/snc_bench --structure static-snc --encoder poisson --backend cpu \
                  --num-steps 50 --num-samples 64

# Parity + speedup across all five backends (cpu/openmp/cuda-atomic/bucket/sort):
./build-cuda/snc_bench --structure static-snc --num-steps 80 --num-samples 64 \
                       --synapse-budget 200000 --hidden 512 --compare-backends
```

See `snc_bench --help` for the full option list (datasets, encoders, backends,
structures, LIF params, JSON/CSV logging).

## Training (Track A): frozen structure, e-prop weights

`snc_train` freezes a graph's topology and trains only its weights with
**e-prop** (Bellec et al. 2020), a local, online approximation to BPTT. This
isolates the value of the structural prior (new-plan.md §8.1).

The trainer owns its own **signed** weight vector — topology and delays come
from the frozen `SNNGraph`, but learning is decoupled from the graph's
Dale-sign layout. Per sample, over the `num_steps` window:

```
eligibility  E_s   = sum_t  psi_post(s)[t] * tr_pre(s)[t]     # local, per synapse
learn signal L_j   = output: softmax_j - onehot_j             # readout error
                     hidden: sum_k B[j,k] * (softmax_k-onehot_k)   # random feedback
update       w_s  -= (lr / T) * L_post(s) * E_s
```

`psi` is a triangular surrogate spike derivative and `tr` a low-passed
pre-spike trace. Output spike counts are the class logits (softmax +
cross-entropy). Hidden credit assignment uses a fixed random feedback matrix
`B` (broadcast alignment), so there is **no global backprop**.

Stability note: weights must start near threshold (`--w-init` small) so the
surrogate is non-zero — too-strong init saturates every neuron, ψ collapses to
0, and nothing learns. The shipped default keeps the network in regime.

```bash
cmake --build build --target snc_train -j

# Train a frozen structure-aware sparse graph on the synthetic task:
./build/snc_train --structure static-snc --epochs 12

# Does hidden-layer e-prop actually help vs a trained readout only?
./build/snc_train --train-mode readout   # stalls near chance
./build/snc_train --train-mode all       # learns (default)
```

Verified: on the separable synthetic task (10 classes, chance 0.10) full e-prop
goes 0.10 → ~1.0 test accuracy in a few epochs with a clean loss curve; a
trained-readout-only baseline stays near chance, showing the local hidden
learning carries the task. Runs are deterministic for a fixed seed; static-snc
reaches the same accuracy at ~half the synaptic events of the dense baseline.

### GPU-accelerated training

`snc_train --device cuda` runs the same e-prop rule on the GPU, but processes a
whole **minibatch of samples in parallel** (`--batch`, default 64) — each sample
is independent under a frozen topology, so per-batch state/eligibility are sized
`[batch × N]` / `[batch × S]` and the kernels fan out over `batch·N` / `batch·S`
threads. Poisson encoding runs on-device via cuRAND; weights stay resident
across the epoch. It is **minibatch SGD** (not bit-identical to the per-sample
CPU path), so it does `batch×` fewer weight updates per epoch and wants a larger
learning rate (`--batch 32 --lr 1.0` matches the CPU `--lr 0.08` here). Latency
encoding is unsupported on the GPU path and falls back to the CPU trainer.

Full MNIST (60k/10k, static-snc, hidden 256), RTX A6000:

| path | epoch time | test acc |
|---|---|---|
| CPU (per-sample e-prop) | 23.3 s | 0.939 |
| **GPU (minibatch e-prop)** | **1.5 s** | **0.941** |

≈ **15× per-epoch speedup at accuracy parity** — full training drops from ~4 min
to ~18 s. When CUDA is unavailable the flag falls back to the CPU trainer.

## Two-timescale co-training (Track B-1): structure + weights

`snc_cotrain` separates the two clocks of new-plan.md §2/§8.2:

```
for each outer (structural) round:
    inner loop:  train weights for K epochs with e-prop      # fast path
    collect activity stats (per-synapse deliveries, per-neuron firing)
    structural_update: prune weakest + grow local replacements # slow path
    recompile graph; carry learned weights across
```

The slow path lives in `Connectome`, a mutable, position-aware edge list (one
1-D position per neuron per layer). `structural_update` is **budget-constant
rewiring**: prune the `K` weakest synapses by |weight| (skipping ones too young
to have trained), then grow `K` new ones, each connecting spatially-near
neurons in adjacent layers — targeted at the posts that just lost a synapse and
biased toward the most-active local pre. Because positions are monotonic in
index, the locality window is a contiguous index range, so growth is cheap and
exact. `--grow 0` freezes structure (static baseline) so dynamic vs static can
be compared at an equal synapse budget and equal epoch count.

Both timescales can run on the GPU: `--device cuda --batch N` trains each inner
loop with the batched GPU e-prop trainer (which also returns the per-synapse /
per-neuron activity the structural update consumes), falling back to the CPU
trainer when CUDA is unavailable.

```bash
cmake --build build --target snc_cotrain -j
./snc_cotrain --outer 16 --inner 2 --grow 80  --structural-budget 800   # dynamic
./snc_cotrain --outer 16 --inner 2 --grow 0   --structural-budget 800   # static
./snc_cotrain --dataset mnist --data-dir data/mnist --device cuda --batch 32 \
              --outer 12 --inner 1 --grow 200 --structural-budget 3000     # GPU
```

**Finding (synthetic, 10 classes, noise 0.5, 3 seeds).** Rewiring helps most
when the budget is *tight* — every synapse must count, so reallocating from
useless to useful local connections matters:

| synapse budget | static test-acc | dynamic test-acc |
|---|---|---|
| ~800  (very tight) | 0.48 | **0.57** |
| ~1200 (tight)      | 0.62 | 0.62 |
| ~2800 (ample)      | **0.95** | 0.84 |

At an ample budget the static topology is already sufficient and rewiring churn
(fresh synapses need re-training) *hurts*. Budget is held constant across rounds
(grown == pruned); runs are deterministic for a fixed seed.

### Real data: MNIST

```bash
./scripts/fetch_mnist.sh                       # -> data/mnist/*-ubyte (gitignored)

# Frozen-structure e-prop baseline (no co-training):
./snc_train  --dataset mnist --data-dir data/mnist --hidden 256 \
             --num-train 2000 --num-test 1000 --num-steps 20 --epochs 8
# -> ~0.85 test accuracy, 32k-synapse local SNN, no backprop, ~8 s.

# Two-timescale co-training (genuine t10k test split):
./snc_cotrain --dataset mnist --data-dir data/mnist --hidden 256 \
              --num-train 1500 --num-test 1000 --num-steps 20 \
              --outer 12 --inner 2 --structural-budget 3000 --grow 200
```

The tight-budget finding **holds on real MNIST** (1500 train / 1000 t10k test,
3 seeds): dynamic rewiring beats static by ~6 points at the tightest budget, and
all three seeds agree.

| MNIST synapse budget | static test-acc | dynamic test-acc |
|---|---|---|
| ~5200 (loose) | 0.708 | 0.716 |
| ~2080 (tight) | 0.461 | **0.523** |

The **rigorous full-MNIST, multi-seed version** of both this and Experiment 1
is in [`experiments-mnist.md`](experiments-mnist.md): with error bars over 3
seeds on the full dataset, `static-snc` beats `random-sparse` by +4.0 pts at a
matched budget, and dynamic co-training beats static by +10.9 pts at a very
tight budget.

## Scope so far

Implemented: graph abstraction + generators, structural→CSR compiler, three
encoders, cpu/openmp runtimes, **three CUDA delivery backends** (atomic /
bucket / sort, parity-checked), MNIST + synthetic datasets, the benchmark CLI,
**frozen-structure e-prop training** (CPU and **GPU minibatch**, ~15× faster),
and **two-timescale structure+weight co-training** (grow/prune/rewire on the
slow clock).

Evaluated: the rigorous full-MNIST multi-seed study (Phase 7) — see
[`experiments-mnist.md`](experiments-mnist.md).

Multi-layer graphs are supported (`--hidden W1,W2,...`): all generators loop
over consecutive layer pairs and e-prop uses direct feedback alignment, so depth
"just works" — see the depth sweep in [`experiments-mnist.md`](experiments-mnist.md).

**PyTorch bridge (Phase 5).** `snc_export` dumps a graph's CSR to a flat binary;
`python/` loads it and trains the exact topology with true surrogate-gradient
**BPTT** (LIF in torch ops, custom Heaviside/fast-sigmoid autograd Function,
autograd unrolls the timestep loop). PyTorch owns data/optimizer/loss; SNC owns
the structure + spiking model. See [`python/README.md`](../python/README.md).
On MNIST, BPTT reaches ~0.977 (vs ~0.940 for e-prop), removes the depth
penalty, and lets static-snc match dense at ~6× fewer synapses (Exp 4).

Not yet (later phases of new-plan.md): event-based / audio datasets, and a
PyTorch path for delayed (delay>1) graphs.
