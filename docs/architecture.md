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
| `src/bench/snc_bench.cpp` | `snc_bench` CLI: build graph, encode, run, report |

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
  its weighted spikes into the device ring. Validated to match cpu spike/event
  counts exactly. `cuda-bucket` / `cuda-sort` currently alias this kernel; the
  target-sharded and segmented-reduction backends are the next PR.

When a `cuda-*` backend is requested but unavailable (built without CUDA, or no
device), `forward()` transparently falls back to cpu and reports the actual
backend in `ForwardResult::used_backend`.

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

# Parity + speedup across backends (cpu / openmp / cuda-atomic):
./build-cuda/snc_bench --structure static-snc --num-steps 80 --num-samples 64 \
                       --synapse-budget 200000 --hidden 512 --compare-backends
```

See `snc_bench --help` for the full option list (datasets, encoders, backends,
structures, LIF params, JSON/CSV logging).

## Scope of this slice

Implemented: graph abstraction + generators, structural→CSR compiler, three
encoders, cpu/openmp runtimes, a verified Stage-1 CUDA atomic backend, and the
benchmark CLI.

Not yet (later phases of new-plan.md): supervised surrogate-gradient training
and the PyTorch bridge (Phase 5), structural epochs driven by activity
(Phase 6), the CUDA bucket/sort reduction backends (Phase 4), and event-based /
audio datasets (Phase 7).
