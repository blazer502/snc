# SNC New Plan: Structure-Aware SNN + CUDA Runtime + Training Evaluation

## 0. Context

Repository:

```bash
git clone git@github.com:blazer502/snc.git
cd snc
```

SNC is currently implemented as a brain-like structural neuromorphic model. The new direction is to reposition it as a **structure-aware spiking neural computing substrate** rather than a literal brain simulator.

The goal is:

> Use brain-inspired structural features—locality, morphology-constrained sparse connectivity, structural plasticity, event-driven spikes, and parallelism—to accelerate SNN-style computation and evaluate the model through real training workloads.

In other words, SNC should become:

> A structure-aware SNN runtime and training system that separates slow structural evolution from fast spike execution, supports CPU/OpenMP and CUDA backends, and evaluates structural connectivity on standard neuromorphic/SNN tasks.

---

## 1. High-Level Direction

### 1.1 Previous framing

Avoid making the primary claim that SNC behaves like a real brain.

Terms to de-emphasize:

- real brain simulation
- pre-adolescent brain capability
- human-like sensory/motor system
- artificial toddler
- biological realism as the main goal

These claims are difficult to validate scientifically and make the project look speculative.

### 1.2 New framing

Use this framing instead:

> SNC is a structure-aware spiking neural computing substrate that uses brain-inspired morphology and structural plasticity to build sparse SNN graphs, then executes them efficiently through event-driven parallel runtimes.

Core idea:

```text
brain-inspired structure
  -> sparse local spiking graph
  -> event-driven execution
  -> parallel CPU/GPU runtime
  -> real model training and evaluation
```

### 1.3 Main research question

```text
Can brain-inspired structural constraints produce sparse SNN topologies
that improve efficiency, locality, and parallel execution without losing
accuracy on real learning tasks?
```

### 1.4 Expected contributions

1. **Structure-aware SNN substrate**
   - Represent neuron bodies, axons, dendrites, and synaptic contacts in a compact structural grid.
   - Use local morphology rules to generate sparse connectivity.

2. **Two-timescale execution model**
   - Fast path: spike execution every timestep.
   - Slow path: structural growth, pruning, rewiring, and synaptogenesis every structural epoch.

3. **CUDA backend for sparse event-driven execution**
   - Move the SNN fast path from OpenMP to CUDA.
   - Use target-sharded event buffers and reduction to reduce atomic contention.

4. **Training and evaluation pipeline**
   - Train SNC-generated sparse SNNs on real datasets.
   - Compare against dense SNNs, random sparse SNNs, and static graph baselines.

5. **Input encoding layer**
   - Support multiple SNN input encodings: direct current, rate/Poisson, latency, event passthrough, and cochlear/audio encoding.

---

## 2. Target System Architecture

The new SNC architecture should be organized into three layers.

```text
[Structural Layer]
  - 3D voxel grid
  - neuron body / axon / dendrite representation
  - morphology templates
  - axon-dendrite contact rule
  - structural growth / pruning / rewiring
  - local energy/resource model

        ↓ compile / export

[Spiking Graph Layer]
  - neurons
  - synapses
  - delays
  - branches
  - weights
  - eligibility traces
  - spike queues

        ↓ execute

[Runtime Layer]
  - CPU baseline
  - OpenMP backend
  - CUDA backend
  - deterministic event dispatch
  - profiling
  - benchmark harness
```

Important principle:

> The structural layer should not be on the critical path of every spike timestep.

Instead:

```text
for each training epoch or simulation window:
    run many fast SNN timesteps
    periodically run structural update
    recompile / compact graph for runtime backend
```

---

## 3. Refactoring Plan

### 3.1 Separate structural model from SNN execution

Create clear boundaries between:

```text
src/structure/
  structural_grid.*
  morphology.*
  growth.*
  pruning.*
  synaptogenesis.*

src/graph/
  snn_graph.*
  neuron_state.*
  synapse_csr.*
  graph_compiler.*

src/runtime/
  cpu_backend.*
  openmp_backend.*
  cuda_backend.*

src/training/
  encoders.*
  losses.*
  train_loop.*
  datasets.*
```

The structural layer should generate or update a sparse graph. The runtime layer should execute that graph.

### 3.2 Add graph export/compile step

Add a compiler from structural grid to runtime graph:

```cpp
SNNGraph compile_structure_to_graph(const StructuralGrid& grid);
```

The compiled graph should support at least two formats:

```text
CSR format:
  row_ptr[neuron_id] -> outgoing synapse range
  post_ids[synapse_id]
  weights[synapse_id]
  delays[synapse_id]
  branch_ids[synapse_id]

COO/event format:
  pre_ids[synapse_id]
  post_ids[synapse_id]
  weights[synapse_id]
  delays[synapse_id]
```

CSR is useful for expanding spikes from pre-synaptic neurons.
COO or sorted event buffers are useful for target-side reduction.

### 3.3 Introduce structural epochs

Do not mutate structure every timestep by default.

Add config:

```yaml
structural_period: 100
enable_growth: true
enable_pruning: true
enable_rewiring: true
```

Main loop:

```cpp
for (int t = 0; t < total_steps; t++) {
    runtime.step();

    if (t % structural_period == 0) {
        structure.update(activity_stats);
        graph = compile_structure_to_graph(structure);
        runtime.reload_graph(graph);
    }
}
```

---

## 4. CUDA Backend Plan

### 4.1 Why CUDA is needed

OpenMP is useful for an initial prototype, but SNC's target workloads are sparse, event-driven, and highly parallel. CUDA can accelerate:

- membrane integration
- spike generation
- synaptic event expansion
- target-side accumulation
- STDP / eligibility updates
- batched inference
- training over minibatches

The key challenge is irregular sparse event delivery.

### 4.2 CUDA development stages

#### Stage 1: GPU inference fast path

Implement forward simulation only.

Kernels:

```text
1. integrate_kernel
2. fire_kernel
3. expand_spikes_kernel
4. deliver_events_atomic_kernel
5. reset_or_decay_kernel
```

Initial event delivery:

```cpp
for each fired neuron:
    for each outgoing synapse:
        atomicAdd(input_acc[target], weight);
```

This is simple and good for correctness testing.

#### Stage 2: Target-sharded event delivery

Replace naive atomic delivery with target-sharded buckets.

Pipeline:

```text
fired neurons
  -> expand outgoing synapses
  -> produce event list: (target, branch, value, delay)
  -> assign bucket = target % num_buckets
  -> reduce events per target
  -> apply accumulated input
```

This is the CUDA version of the existing OpenMP per-target bucketing idea.

Expected benefit:

- reduce global atomic contention
- improve determinism
- expose more regular parallel work
- improve performance under high fan-in

#### Stage 3: Sorted / segmented reduction backend

Use sort-by-target or bucket-by-target, then segmented reduction.

Possible implementation options:

- custom CUDA kernels
- CUB radix sort + segmented reduce
- Thrust sort/reduce as an initial implementation
- later replace with custom kernels

Pseudo-pipeline:

```cpp
expand_spikes<<<...>>>(fired, row_ptr, post_ids, weights, events);

sort_by_key(events.target);

segmented_reduce_by_target(events.target, events.value, input_acc);
```

#### Stage 4: STDP / eligibility on GPU

Move local learning updates to GPU.

Data needed:

```text
pre_trace[neuron]
post_trace[neuron]
eligibility[synapse]
weight[synapse]
last_spike_time[neuron]
```

Initial rule:

```text
if pre fires before post:
    weight += lr_plus * pre_trace * post_trace

if post fires before pre:
    weight -= lr_minus * pre_trace * post_trace
```

Keep rules modular so we can compare:

- fixed weights
- STDP
- reward-modulated STDP
- surrogate-gradient-trained weights

#### Stage 5: GPU structural kernels

Do this only after the spike fast path works.

Candidate structural kernels:

- active voxel frontier update
- local growth candidate generation
- axon-dendrite contact detection
- pruning by activity
- local energy field update
- region-level compaction

Important: structural mutation is a slow path. It does not need to run every timestep.

---

## 5. CUDA Data Layout

### 5.1 Device neuron state

Use Struct-of-Arrays.

```cpp
struct DeviceNeurons {
    float* v;
    float* input_acc;
    float* threshold;
    float* refractory;
    uint8_t* fired;
    uint8_t* role;
};
```

Avoid Array-of-Structs for the hot path.

### 5.2 Device synapse graph

Use CSR for outgoing synapses.

```cpp
struct DeviceSynapses {
    int* row_ptr;       // [num_neurons + 1]
    int* post_ids;      // [num_synapses]
    float* weights;     // [num_synapses]
    int* delays;        // [num_synapses]
    int* branch_ids;    // [num_synapses]
    float* eligibility; // [num_synapses], optional
};
```

### 5.3 Device event buffers

```cpp
struct DeviceEvents {
    int* target_ids;
    int* branch_ids;
    float* values;
    int* delivery_times;
    int* count;
};
```

For delayed spikes, maintain either:

```text
ring buffer by time slot
```

or:

```text
event buffer sorted by delivery_time
```

Start with a fixed-size ring buffer because it matches SNN timestep simulation.

---

## 6. Input Encoding Research Summary

SNC needs a clean input API. Existing brain-like and SNN systems usually do not feed raw tensors directly into spiking neurons. They convert data into spike trains, event streams, or injected currents.

Implement all dataset inputs through a unified event abstraction:

```cpp
struct InputEvent {
    int sample_id;
    int channel_id;
    int time;
    float amplitude;
};
```

Every encoder should produce `InputEvent` streams.

### 6.1 Direct current encoding

Used for debugging and stable training.

```text
pixel value -> input current
audio feature -> input current
```

Example:

```cpp
input_acc[channel] += normalized_value;
```

Pros:

- easiest to implement
- stable for early training
- useful for debugging

Cons:

- less spike-native
- weaker neuromorphic claim

Use this first.

### 6.2 Rate / Poisson encoding

Common SNN input method.

```text
larger input value -> higher spike probability / firing rate
```

For image pixels:

```text
bright pixel -> many spikes
dark pixel   -> few or no spikes
```

Pseudo-code:

```cpp
p = pixel_intensity * max_rate * dt;
if random_uniform() < p:
    emit_spike(channel, time);
```

Pros:

- widely used
- simple baseline
- easy to compare with existing SNN frameworks

Cons:

- produces many spikes
- can be inefficient for latency/energy

### 6.3 Latency / time-to-first-spike encoding

Encode value as spike timing.

```text
larger input value -> earlier spike
smaller input value -> later spike or no spike
```

Example:

```cpp
time = max_time * (1.0 - normalized_value);
emit_spike(channel, time);
```

Pros:

- sparse
- efficient
- fits event-driven execution

Cons:

- sometimes harder to train
- timing precision matters

### 6.4 Delta / temporal-difference encoding

Encode changes rather than absolute values.

```text
if value[t] - value[t-1] > threshold:
    emit positive spike

if value[t] - value[t-1] < -threshold:
    emit negative spike
```

Useful for:

- video
- event camera
- streaming sensor data
- audio changes

### 6.5 Event passthrough

For neuromorphic datasets that already provide events.

Input format:

```text
event = (x, y, polarity, time)
```

Map to channel:

```cpp
channel_id = polarity * width * height + y * width + x;
emit_spike(channel_id, time);
```

Use for:

- N-MNIST
- DVS Gesture
- CIFAR10-DVS
- other event camera datasets

### 6.6 Cochlear / audio encoding

For audio or speech tasks, use a cochlear frontend.

Pipeline:

```text
audio waveform
  -> filterbank / cochlear model
  -> channel energy over time
  -> spike encoding
  -> input event stream
```

Possible encodings:

- rate coding per frequency band
- latency coding per time-frequency bin
- threshold-crossing spikes
- delta spikes

Use for:

- SHD
- SSC
- spoken digit tasks
- SNC's existing cochlear pathway

---

## 7. Training Evaluation Plan

### 7.1 Evaluation philosophy

Do not evaluate only with custom demos.

Add standard training tasks and compare against clear baselines.

Primary question:

```text
Does SNC's structure-aware sparse topology improve efficiency or accuracy
under the same synapse budget?
```

Secondary question:

```text
Can CUDA event dispatch make SNC faster than OpenMP for sparse SNN workloads?
```

### 7.2 Datasets

Start with simple datasets, then move to neuromorphic data.

#### Tier 1: Debug / baseline

- MNIST
- Fashion-MNIST

Purpose:

- easy debugging
- compare encoding methods
- verify training loop

Encoders:

- direct current
- Poisson
- latency

#### Tier 2: Event-based vision

- N-MNIST
- DVS Gesture
- CIFAR10-DVS

Purpose:

- evaluate event-driven runtime
- avoid artificial spike conversion
- measure spike/event throughput

Encoder:

- event passthrough

#### Tier 3: Audio / speech

- SHD
- SSC
- spoken digits

Purpose:

- evaluate cochlear-style input
- align with SNC's existing audio pathway

Encoder:

- cochlear frontend
- rate/latency/delta spikes

### 7.3 Model configurations

Compare these model variants.

```text
A. Dense SNN
B. Random sparse SNN
C. Static SNC structure
D. Dynamic SNC structure
E. Dynamic SNC structure + STDP
F. Dynamic SNC structure + surrogate-gradient training
```

### 7.4 Runtime configurations

Compare these execution backends.

```text
1. CPU single-thread
2. OpenMP
3. CUDA atomic
4. CUDA target-bucketed
5. CUDA sorted/reduced
```

### 7.5 Metrics

Report both ML and systems metrics.

ML metrics:

```text
accuracy
loss
convergence speed
epochs to target accuracy
robustness to sparsity
```

SNN metrics:

```text
spike count per inference
synaptic events per inference
average firing rate
time-to-decision
synapse utilization
```

Systems metrics:

```text
training time per epoch
inference latency
spikes/sec
synaptic events/sec
memory per neuron
memory per synapse
GPU memory usage
GPU occupancy
atomic contention
event buffer overflow rate
speedup over OpenMP
```

Structural metrics:

```text
number of neurons
number of synapses
average fan-in / fan-out
graph diameter
clustering coefficient
locality of connections
growth/pruning rate
active region count
```

Energy proxy:

```text
energy_proxy =
    alpha * num_spikes
  + beta  * num_synaptic_events
  + gamma * memory_traffic
  + delta * structural_updates
```

Use the same proxy consistently across variants.

---

## 8. Training Methods

### 8.1 Frozen-structure training

First training method.

Pipeline:

```text
1. Grow SNC structure
2. Compile structure to SNN graph
3. Freeze topology
4. Train only weights
5. Evaluate accuracy and efficiency
```

This isolates the value of SNC's structural prior.

Compare:

```text
SNC sparse topology vs random sparse topology
under the same number of synapses.
```

### 8.2 Two-timescale training

Second training method.

Pipeline:

```text
inner loop:
  train weights for K steps or K batches

outer loop:
  grow / prune / rewire structure
  recompile graph
  continue training
```

Pseudo-code:

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        events = encoder(batch.x)
        logits = snc.forward(events)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()

    if epoch % structural_period == 0:
        snc.structural_update(activity_stats)
        snc.recompile_graph()
```

### 8.3 STDP / local learning

Use for unsupervised or biologically inspired baseline.

Pipeline:

```text
input spikes
  -> run network
  -> update weights locally with pre/post traces
  -> train readout layer
```

This is useful but should not be the only training method.

### 8.4 Surrogate-gradient training

Use this as the main supervised training method.

Requirements:

- PyTorch bridge or custom CUDA backward
- differentiable surrogate for spike function
- output decoder
- loss function

Spike function:

```text
fired = v > threshold
```

Backward uses surrogate derivative:

```text
d fired / d v ≈ surrogate'(v - threshold)
```

---

## 9. PyTorch Integration Plan

### 9.1 Goal

Do not reimplement the whole training ecosystem in C++.

Use PyTorch for:

- dataset loading
- batching
- optimizer
- loss
- logging
- checkpointing

Use SNC for:

- structural graph generation
- spike simulation
- CUDA event dispatch
- optional custom backward

### 9.2 Initial Python API

```python
import snc

model = snc.SNCModel(
    structure="snc",
    backend="cuda",
    encoder="poisson",
    num_steps=100,
)

logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

### 9.3 C++/CUDA extension

Directory:

```text
python/
  snc/
    __init__.py
    model.py
    encoders.py
    datasets.py

src/cuda/
  snc_cuda.cu
  kernels.cu
  bindings.cpp
```

Build with:

```text
torch.utils.cpp_extension
```

or CMake + pybind11.

### 9.4 Output decoder

Support:

```text
spike count decoder
first-spike decoder
final membrane decoder
temporal readout decoder
```

Start with spike count:

```python
logits[class_id] = spike_count[output_neuron[class_id]]
```

---

## 10. Proposed CLI

Add a new benchmark binary:

```bash
./build/snc_bench \
  --dataset mnist \
  --encoder poisson \
  --backend cuda \
  --structure static-snc \
  --num-steps 100 \
  --epochs 10
```

Options:

```text
--dataset mnist | fashion-mnist | n-mnist | dvs-gesture | shd | ssc
--encoder direct | poisson | latency | delta | event | cochlear
--backend cpu | openmp | cuda-atomic | cuda-bucket | cuda-sort
--structure dense | random-sparse | static-snc | dynamic-snc
--learning fixed | stdp | surrogate
--num-steps N
--structural-period K
--batch-size B
--synapse-budget S
--seed SEED
--log-dir DIR
```

---

## 11. Experimental Matrix

### Experiment 1: Does SNC topology help?

```text
Task: MNIST / N-MNIST
Backend: CPU or CUDA
Training: surrogate gradient
Compare:
  - dense SNN
  - random sparse SNN
  - static SNC sparse SNN
Same:
  - neuron count
  - synapse budget
  - training steps
Metrics:
  - accuracy
  - spike count
  - synaptic events
  - memory
```

### Experiment 2: Does dynamic structure help?

```text
Task: MNIST / SHD
Backend: CUDA
Training: two-timescale
Compare:
  - static SNC
  - dynamic SNC grow/prune
  - dynamic SNC grow/prune/rewire
Metrics:
  - accuracy
  - convergence speed
  - final synapse count
  - structural update overhead
```

### Experiment 3: Does CUDA backend help?

```text
Task: synthetic sparse SNN + N-MNIST
Compare:
  - OpenMP
  - CUDA atomic
  - CUDA bucket
  - CUDA sorted/reduced
Vary:
  - spike rate
  - fan-in
  - fan-out
  - delay distribution
  - synapse count
Metrics:
  - events/sec
  - latency
  - atomic conflicts
  - GPU occupancy
  - memory bandwidth
```

### Experiment 4: Which input encoding works best?

```text
Task: MNIST / SHD
Compare:
  - direct
  - Poisson
  - latency
  - delta
  - event passthrough
Metrics:
  - accuracy
  - spike count
  - time-to-decision
  - training stability
```

### Experiment 5: Structural locality and GPU performance

```text
Task: synthetic + real datasets
Compare:
  - random sparse graph
  - SNC morphology-constrained graph
Metrics:
  - cache locality proxy
  - event bucket locality
  - reduction contention
  - GPU throughput
```

---

## 12. Implementation Checklist

### Phase 0: Cleanup and framing

- [ ] Rename or add README section: "Structure-Aware SNN Computing Substrate"
- [ ] Move speculative brain-like claims to a background/motivation section
- [ ] Document the new architecture
- [ ] Add `docs/new-plan.md`
- [ ] Add `docs/architecture.md`

### Phase 1: Graph abstraction

- [ ] Add `SNNGraph`
- [ ] Add CSR synapse representation
- [ ] Add graph compiler from structural grid
- [ ] Add random sparse graph generator
- [ ] Add dense graph generator
- [ ] Add graph statistics utility

### Phase 2: CPU/OpenMP benchmark path

- [ ] Add `snc_bench`
- [ ] Add fixed-structure inference
- [ ] Add encoder API
- [ ] Add direct encoder
- [ ] Add Poisson encoder
- [ ] Add latency encoder
- [ ] Add CSV/JSON logging
- [ ] Add deterministic seed support

### Phase 3: CUDA atomic backend

- [ ] Add CUDA build option
- [ ] Add device neuron arrays
- [ ] Add device synapse CSR
- [ ] Add integrate kernel
- [ ] Add fire kernel
- [ ] Add spike expansion kernel
- [ ] Add atomic delivery kernel
- [ ] Validate against CPU backend

### Phase 4: CUDA bucket/reduction backend

- [ ] Add event buffer
- [ ] Add target bucket assignment
- [ ] Add segmented reduction path
- [ ] Add overflow handling
- [ ] Add performance counters
- [ ] Compare against CUDA atomic

### Phase 5: Training pipeline

- [ ] Add Python package skeleton
- [ ] Add PyTorch dataset loaders
- [ ] Add Python encoders
- [ ] Add C++/CUDA extension wrapper
- [ ] Add spike count decoder
- [ ] Add surrogate gradient training path
- [ ] Add checkpoints and logging

### Phase 6: Structural training

- [ ] Add frozen-structure training
- [ ] Add activity statistics collection
- [ ] Add structural epoch update
- [ ] Add graph recompilation after structure update
- [ ] Add pruning rule based on activity
- [ ] Add growth rule based on activity/resource field

### Phase 7: Evaluation

- [ ] Run MNIST baseline
- [ ] Run N-MNIST event benchmark
- [ ] Run SHD audio benchmark
- [ ] Compare structure variants
- [ ] Compare runtime backends
- [ ] Generate plots
- [ ] Write experiment report

---

## 13. Acceptance Criteria

The agent should consider the first version successful if the following are true:

### Functional

- [ ] `snc_bench` can run with CPU/OpenMP and CUDA backend.
- [ ] SNC can load or generate a sparse graph.
- [ ] SNC can encode MNIST inputs into spike/input events.
- [ ] SNC can run forward inference and produce class logits.
- [ ] CUDA output matches CPU output within tolerance for fixed seed and deterministic settings.

### Training

- [ ] A PyTorch training script can train a fixed SNC sparse topology.
- [ ] At least one dataset reaches non-random accuracy.
- [ ] Training logs accuracy, loss, spike count, and runtime.

### Systems

- [ ] CUDA backend reports events/sec and speedup over OpenMP.
- [ ] Atomic and bucketed CUDA paths are both implemented or at least scaffolded.
- [ ] Event buffer overflow is detected and reported.
- [ ] Memory usage is reported.

### Research

- [ ] Static SNC topology is compared with random sparse topology under the same synapse budget.
- [ ] At least one input encoding comparison is reported.
- [ ] At least one runtime backend comparison is reported.

---

## 14. Suggested First Pull Request

The first PR should be small and structural.

Title:

```text
Refactor SNC toward structure-aware SNN benchmark substrate
```

Contents:

```text
1. Add docs/new-plan.md
2. Add SNNGraph abstraction
3. Add CSR conversion from current synapse representation
4. Add snc_bench skeleton
5. Add encoder interface
6. Add direct/Poisson encoder stubs
7. Add runtime backend enum: cpu/openmp/cuda
8. Add build flag for CUDA, even if CUDA backend is initially stubbed
```

Do not implement all CUDA kernels in the first PR.

---

## 15. Suggested README Abstract

Use this as the new project description:

```text
SNC is a structure-aware spiking neural computing substrate. Instead of
treating SNN connectivity as a fixed adjacency matrix, SNC represents a
compact structural substrate where neuron bodies, axons, dendrites, and
synaptic contacts define sparse local connectivity. The resulting graph is
executed by event-driven CPU/OpenMP and CUDA runtimes. SNC separates slow
brain-inspired structural plasticity from the fast spiking execution path,
allowing sparse topology to evolve while keeping inference and training
efficient. The system is designed to evaluate whether biological structural
principles can improve the efficiency, locality, and scalability of SNN
workloads.
```

---

## 16. Notes for the Server Agent

Prioritize engineering order over biological completeness.

Recommended order:

```text
1. Make graph abstraction clean.
2. Make benchmark executable.
3. Make CPU baseline deterministic.
4. Add CUDA atomic backend.
5. Add input encoders.
6. Add fixed-structure training.
7. Add target-bucketed CUDA backend.
8. Add structural epochs.
9. Run evaluations.
```

Avoid spending too much time on:

- biologically realistic sensory organs
- complex motor output
- sleep/consciousness analogies
- human-like behavior claims
- full GPU structural mutation before the fast path works

Focus on:

- sparse graph execution
- event-driven runtime
- CUDA acceleration
- standard training tasks
- measurable speed/accuracy/memory tradeoffs
