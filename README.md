# Structural Neuromorphic Computing

Voxel-based rendering + connectomics + structural plasticity. The structural
matrix (where the tissue is) is intentionally separated from neuronal dynamics
(what the cells are doing). Growth and pruning rules give the network a
Game-of-Life-like spatial flavour, while a regional energy field acts as a
metabolic throttle on structural change.

## Layout

```
include/
  brain_grid.hpp     2-bit packed 3D structural matrix
  energy_field.hpp   coarse regional energy budget
  neuron.hpp         per-neuron state (chemistry, body, synapses)
  simulator.hpp      orchestrates one simulation step
src/
  brain_grid.cpp
  energy_field.cpp
  simulator.cpp
  main.cpp           demo driver
```

## Cell encoding (2 bits)

| value | meaning                                            |
| ----- | -------------------------------------------------- |
| `0`   | empty space                                         |
| `1`   | neuron part (soma / dendrite / axon)                |
| `2`   | synaptic contact between two neurons                |
| `3`   | tissue that cannot become a synapse (no-synapse)    |

32 cells are packed into one `uint64_t`. The grid keeps a front and a back
buffer so synchronous updates can be done with `swap_buffers()`. The current
algorithm uses asynchronous updates that are activity-driven; the back buffer
is exposed for callers that want a strict cellular-automaton sweep.

## Simulation step

```
chemistry            -> per-neuron LIF, parallel across neurons
synaptic transmission-> deliver spikes to post.input_acc, parallel
energy regeneration  -> regions regenerate; firing somas pay
sprouting            -> active neurons extend NEURON voxels under
                        volume exclusion + energy gates
synaptogenesis       -> contacts between distinct neurons may convert
                        a NEURON voxel to SYNAPSE; per-region capped
pruning              -> weak / silent synapses revert to NEURON
```

Chemistry and synaptic transmission run in parallel (OpenMP) using `omp atomic`
for cross-neuron accumulation. Structural mutation phases (sprouting,
synaptogenesis, pruning) run serially because their work is bounded by
recently-active neurons rather than total grid size.

## Build

```
cmake -S . -B build
cmake --build build -j
./build/snc_demo [steps=200] [neurons=80]
```

OpenMP is detected automatically; without it the code still builds and runs
single-threaded.

## Energy throttle

The volume is partitioned into cubic regions (default `8^3` voxels). Every
region holds a single energy budget. Firing, sprouting, synapse formation and
synapse use all draw from the local region; energy regenerates by a fixed
amount per step. Hot regions therefore cannot keep growing; this matches the
"computation density limits structural growth" intuition.

## Pruning

Synapses are pruned when their weight falls below `prune_weight_floor` or they
have not transmitted in `prune_inactive_steps` steps. The contact voxel
demotes back to `NEURON` (still owned by the post-synaptic neuron); only the
edge is removed.

## Notes

- `(X * Y) % 32 == 0` is required so that z-slice parallelism never causes
  two threads to update bits inside the same packed word. The default
  `64x64x64` grid satisfies this.
- A small auxiliary `owner_` map (one `uint32_t` per voxel) records neuron
  ownership for synaptogenesis. It is *not* part of the 2-bit structural
  matrix; it exists purely so the simulator can tell whether two adjacent
  `NEURON` voxels belong to two different cells.
