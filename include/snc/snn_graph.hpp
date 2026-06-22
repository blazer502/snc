// Structure-aware SNN substrate: the runtime-facing spiking graph.
//
// The biological side of SNC (BrainGrid + Simulator + Neuron) is a slow
// structural-plasticity model. For fast, backend-agnostic spike execution we
// compile that structure down to a flat, immutable Compressed-Sparse-Row
// (CSR) graph. This file defines that graph and the ways to build one:
//
//   * compile_from_simulator -- snapshot the live structural model.
//   * make_dense             -- fully-connected layered baseline.
//   * make_random_sparse     -- Erdos-Renyi sparse baseline at a synapse budget.
//
// The CSR graph owns no chemistry and no plasticity state; it is the
// "compiled" artefact described in new-plan.md section 3.2. Runtimes
// (cpu/openmp/cuda) consume it read-only.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace snc {

class Simulator;  // fwd; compile_from_simulator reads neurons() read-only.

// Functional role of a graph neuron. Mirrors NeuronRole in neuron.hpp but is
// kept independent so the graph layer has no compile dependency on the
// structural model.
enum class GraphRole : uint8_t { INTERNAL = 0, INPUT = 1, OUTPUT = 2 };

// Runtime-facing spiking graph in CSR-over-outgoing-synapses form.
//
// Synapse `s` in [row_ptr[i], row_ptr[i+1]) is an outgoing edge of neuron `i`:
//   post_ids[s]   -- post-synaptic neuron id
//   weights[s]    -- non-negative efficacy (Dale's sign lives on the PRE neuron)
//   delays[s]     -- conduction delay in timesteps (>= 1)
//   branch_ids[s] -- post-synaptic dendritic branch (informational for now)
//
// Per-neuron metadata is parallel to [0, num_neurons):
//   role[i]    -- INTERNAL / INPUT / OUTPUT
//   sign[i]    -- +1 excitatory, -1 inhibitory (Dale's principle, on the PRE)
//   channel[i] -- INPUT: driven feature index; OUTPUT: reported class; else -1
struct SNNGraph {
  int num_neurons = 0;

  // CSR arrays.
  std::vector<int> row_ptr;          // size num_neurons + 1
  std::vector<int> post_ids;         // size num_synapses
  std::vector<float> weights;        // size num_synapses
  std::vector<int> delays;           // size num_synapses
  std::vector<uint8_t> branch_ids;   // size num_synapses

  // Per-neuron metadata.
  std::vector<GraphRole> role;       // size num_neurons
  std::vector<int8_t> sign;          // size num_neurons, +1 / -1
  std::vector<int> channel;          // size num_neurons, -1 if not in/out

  int num_synapses() const noexcept {
    return static_cast<int>(post_ids.size());
  }

  // Convenience views resolved once at build time so runtimes/decoders need
  // not rescan role[]. input_neurons[c] / output_neurons may have repeats of
  // the same channel; the decoder groups by channel.
  std::vector<int> input_neurons;    // ids with role == INPUT
  std::vector<int> output_neurons;   // ids with role == OUTPUT
  int num_input_channels = 0;        // 1 + max input channel (0 if none)
  int num_output_channels = 0;       // 1 + max output channel (0 if none)

  // Recompute input/output views and channel counts from role[]/channel[].
  // Called by every generator; call again if you mutate role[]/channel[].
  void rebuild_io_index();

  // Structural integrity check (sizes consistent, row_ptr monotone, post ids
  // in range, delays >= 1). Returns true and leaves `err` empty on success.
  bool validate(std::string& err) const;
};

// Aggregate structural metrics (new-plan.md section 7.5 "Structural metrics").
struct GraphStats {
  int num_neurons = 0;
  int num_synapses = 0;
  int num_inputs = 0;
  int num_outputs = 0;
  double avg_fan_out = 0.0;
  int max_fan_out = 0;
  double avg_fan_in = 0.0;
  int max_fan_in = 0;
  double avg_delay = 0.0;
  double avg_weight = 0.0;
  double density = 0.0;              // synapses / (N*(N-1))
  std::size_t bytes = 0;            // approximate device-relevant footprint
};

GraphStats compute_stats(const SNNGraph& g);
std::string format_stats(const GraphStats& s);

// ---- Generators -----------------------------------------------------------

// Compile the live structural model into a CSR graph. Reads sim.neurons()
// read-only; Dale's sign is taken from each pre neuron's polarity, delays from
// each synapse's conduction_delay. INPUT/OUTPUT roles and channels are carried
// across so encoders/decoders line up with the structural model.
SNNGraph compile_from_simulator(const Simulator& sim);

// Layered fully-connected baseline. `layers` lists neuron counts per layer;
// every neuron in layer L projects to every neuron in layer L+1. Layer 0 is
// INPUT (channel = position in layer), the last layer is OUTPUT (channel =
// position in layer). `delay` is applied to all edges; weights are `weight`.
SNNGraph make_dense(const std::vector<int>& layers, float weight, int delay);

// Erdos-Renyi-style random sparse graph honouring a target synapse budget.
// Same layer/role convention as make_dense, but each potential forward edge is
// kept independently so the expected synapse count equals `synapse_budget`.
// Deterministic for a fixed `seed`. `inhibitory_fraction` of neurons are made
// inhibitory (Dale's principle) so it is a fair baseline against compiled SNC
// graphs.
SNNGraph make_random_sparse(const std::vector<int>& layers, int synapse_budget,
                            float weight, int delay, float inhibitory_fraction,
                            uint64_t seed);

// Morphology-constrained sparse baseline (SNC's "structure-aware" prior in CSR
// form). Each neuron gets a 1-D position in its layer; a pre neuron connects to
// its nearest `synapse_budget / num_forward_pre` neurons in the next layer, so
// connectivity is spatially local rather than uniformly random. Same layer/role
// convention, synapse budget, and fixed `delay` as make_random_sparse, so the
// only difference from the random baseline is *topology* (new-plan.md Exp. 1).
SNNGraph make_static_snc(const std::vector<int>& layers, int synapse_budget,
                         float weight, int delay, float inhibitory_fraction,
                         uint64_t seed);

}  // namespace snc
