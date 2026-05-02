// Per-neuron state, deliberately decoupled from the structural grid.
//
// The structural grid only stores "what kind of cell occupies this voxel"; it
// has no concept of neuron identity, chemistry, or synaptic weight. All of
// those live here, indexed by neuron id. Each neuron is its own independent
// computational unit: chemistry (`potential`, `input_acc`, `fire_rate_ema`)
// and outgoing synapses are updated in parallel without sharing state with
// the structural matrix.

#pragma once

#include <cstdint>
#include <vector>

namespace snc {

struct Voxel {
  int16_t x, y, z;
};

// A spike packet currently traversing a synapse's axon. The scheduler
// advances each packet by one voxel-step per simulation step; when its
// `delay_remaining` reaches zero the packet is deposited into the post
// neuron's incoming queue.
struct SpikePacket {
  float magnitude;
  int delay_remaining;
};

struct SynapseEdge {
  uint32_t target_neuron;  // post-synaptic neuron id (1-based)
  Voxel pos;               // voxel that holds the SYNAPSE state
  float weight;            // [0, weight_max]
  int last_active_step;    // last simulation step that delivered a spike

  // Eligibility trace for three-factor learning. Updated locally on the
  // synapse itself each step (no information beyond its own pre/post is
  // used). When a reward signal is broadcast, every synapse independently
  // applies `weight += lr * reward * eligibility`. This is the analogue of
  // dopamine-modulated STDP in real cortex.
  float eligibility = 0.0f;

  // Conduction delay in simulation steps -- equal to the Manhattan distance
  // from the pre-synaptic soma to the synaptic voxel at formation time.
  // Captures the fact that an action potential needs time to propagate
  // along the axon. The synapse's transit queue plays the role of "spikes
  // currently traveling along the axon".
  int conduction_delay = 1;
  std::vector<SpikePacket> transit;  // queue 4 (synapse waiting queue)

  // Productivity statistics used by the probabilistic per-neuron pruning
  // phase. `delivered_count` counts spike deliveries; `caused_fire_count`
  // counts deliveries that were followed by a post-synaptic firing within
  // one step. Both are local to the synapse (only its own delivery and the
  // post's `fired_this_step` are consulted) so the per-neuron-independent
  // computation property is preserved.
  uint32_t delivered_count = 0;
  uint32_t caused_fire_count = 0;
  int last_delivery_step = -1;
};

// Functional role of a neuron in a data-driven training run.
//   INTERNAL : ordinary cortical neuron (default).
//   INPUT    : driven by an external feature; its `input_acc` is overwritten
//              by `apply_input_pattern` each step rather than coming from
//              synaptic transmission.
//   OUTPUT   : its `fire_rate_ema` is read out as a class logit.
enum class NeuronRole : uint8_t { INTERNAL = 0, INPUT = 1, OUTPUT = 2 };

// Cells in real cortex obey Dale's principle: a single neuron releases
// only excitatory or only inhibitory neurotransmitter at all of its
// synapses. ~80% of cortical neurons are excitatory (glutamatergic), the
// remaining ~20% are inhibitory (GABAergic). Dispatch flips the sign of
// every spike packet emitted by an INHIBITORY neuron, which gives the
// network the lateral-inhibition / winner-take-all dynamics needed for
// sparse coding and clean output decoding.
enum class NeuronPolarity : uint8_t { EXCITATORY = 0, INHIBITORY = 1 };

struct Neuron {
  uint32_t id = 0;          // 1-based; 0 is reserved for "no owner"
  Voxel soma{0, 0, 0};

  NeuronRole role = NeuronRole::INTERNAL;
  NeuronPolarity polarity = NeuronPolarity::EXCITATORY;
  // For INPUT: which feature drives this neuron.
  // For OUTPUT: which class this neuron reports.
  // Ignored for INTERNAL.
  int channel = -1;

  // All voxels currently owned by this neuron with structural state == NEURON.
  // Voxels whose state is SYNAPSE remain owned by their post-synaptic neuron
  // but are not stored here so that sprouting iteration only walks pure
  // neuron tissue.
  std::vector<Voxel> body;

  // Outgoing synapses (this neuron is pre-synaptic).
  std::vector<SynapseEdge> outgoing;

  // Chemistry state -- a deliberately tiny "leaky integrate and fire" model.
  // Real chemical dynamics (Ca2+, neurotransmitter pools, ...) would replace
  // these fields without changing the structural grid in any way.
  float potential = 0.0f;        // membrane potential proxy
  float input_acc = 0.0f;        // accumulated *external* input (not synaptic)
  float fire_rate_ema = 0.0f;    // exponential moving average of firing rate
  int last_fire_step = -1000000;
  bool fired_this_step = false;

  // Queue 1: spike magnitudes deposited by the synapse scheduler since the
  // last integrate phase. The integrate phase sums these (plus any direct
  // external `input_acc`) into the membrane potential and clears the queue.
  std::vector<float> incoming_queue;

  // Sum of all incoming synapse weights, refreshed by the homeostatic
  // phase. Each pre synapse later reads this scalar and rescales its own
  // weight toward a target -- the simulator analogue of the retrograde
  // BDNF / TNF-alpha signal that real cortical neurons release to balance
  // their total drive (synaptic scaling, Turrigiano 2008).
  float incoming_weight_sum = 0.0f;
};

}  // namespace snc
