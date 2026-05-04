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

  // Long-timescale consolidation tag (synaptic-tagging-and-capture, Frey &
  // Morris 1997). When STDP drives a synapse's weight above a threshold a
  // tag is set; the tag decays slowly. Tagged synapses are protected from
  // spine retraction during the next pruning sweep -- their molecular
  // machinery has captured PRPs (plasticity-related proteins) and is
  // structurally stabilised even if activity drops.
  float consolidation_tag = 0.0f;

  // Innate / labelled-line marker. Real cortex protects key reflex-arc
  // and labelled-line connections (retinogeniculate, brainstem reflexes,
  // etc.) from microglial pruning regardless of postnatal experience --
  // these are anatomically-defined wires that the genome wants to
  // preserve. A permanent synapse is exempt from BOTH the silence-timeout
  // and the spine-retraction prune paths; only an explicit demo call to
  // remove it would do so. STDP and reward still freely modulate its
  // weight; permanence affects only physical removal.
  bool permanent = false;

  // Short-term plasticity: fraction of the vesicle pool currently
  // available for release. Each successful release depletes the pool
  // by `cfg.release_depression`; per step, a `cfg.release_recovery`
  // fraction is restored, capped at 1.0. The effective release
  // magnitude is multiplied by `vesicle_state`, producing the
  // characteristic short-term depression observed at most cortical
  // synapses (Markram & Tsodyks). Defaults (0/0) disable the dynamic
  // and keep `vesicle_state == 1.0` indefinitely so legacy demos
  // are unchanged.
  float vesicle_state = 1.0f;

  // Conduction delay in simulation steps -- equal to the Manhattan distance
  // from the pre-synaptic soma to the synaptic voxel at formation time.
  // Captures the fact that an action potential needs time to propagate
  // along the axon. The synapse's transit queue plays the role of "spikes
  // currently traveling along the axon".
  int conduction_delay = 1;
  std::vector<SpikePacket> transit;  // queue 4 (synapse waiting queue)

  // Dendritic-branch index. Real cortical pyramidals have several
  // dendrites that integrate independently before contributing to the
  // soma; a synapse on one dendrite cannot directly compensate for a
  // synapse on another. We model this by assigning each synapse to a
  // branch index. Branch sums are computed independently and only their
  // dendritic-spike or passive contribution is forwarded to the soma.
  uint8_t branch = 0;

  // Productivity statistics used by the probabilistic per-neuron pruning
  // phase. `delivered_count` counts spike deliveries; `caused_fire_count`
  // counts deliveries that were followed by a post-synaptic firing within
  // one step. Both are local to the synapse (only its own delivery and the
  // post's `fired_this_step` are consulted) so the per-neuron-independent
  // computation property is preserved.
  uint32_t delivered_count = 0;
  uint32_t caused_fire_count = 0;
  int last_delivery_step = -1;

  // Pack ZZ v3 -- microglial pruning analogue (Xing et al. 2026 NRR;
  // Schafer & Stevens 2013). The complement-like "eat me" tag grows
  // on every delivery that did not cause the post to fire within the
  // STDP window (tagged-but-useless delivery) and is reset to 0 on
  // STDP-LTP events (delivery caused the post to fire -- demonstrably
  // useful). The "don't eat me" CD47/SIRPalpha analogue is set to a
  // sentinel large value (1e9) on permanent / labelled-line / engram
  // synapses so they are exempt from microglial elimination. The
  // periodic `microglia_phase` removes a synapse only when its tag
  // exceeds threshold AND its weight is also weak AND no
  // consolidation tag protects it -- many simultaneous conditions to
  // ensure only genuinely surplus connections are pruned.
  float eat_me_tag = 0.0f;
  float dont_eat_me = 0.0f;
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
// every spike packet emitted by an inhibitory neuron, which gives the
// network the lateral-inhibition / winner-take-all dynamics needed for
// sparse coding and clean output decoding.
//
// Three GABAergic subtypes are tracked so the simulator can model their
// distinct wiring rules (Tremblay, Lee & Rudy 2016):
//
//   INHIBITORY (=PV) : parvalbumin-expressing basket cells, fast
//                       perisomatic inhibition. The default and most
//                       common inhibitory class. Polarity-sign flip
//                       on dispatch is the only required behaviour.
//   INHIBITORY_SST   : somatostatin-expressing Martinotti cells, slow
//                       dendrite-targeted inhibition. Used for selective
//                       gating of specific dendritic compartments.
//   INHIBITORY_VIP   : vasoactive intestinal peptide cells. Their
//                       canonical role is to inhibit SST cells, opening a
//                       plasticity window in the dendritic compartments
//                       (disinhibition).
//
// Dispatch currently treats all three as sign-flipping the spike packet.
// Demos can use the subtype to route specific synapses (e.g., SST cells
// only target dendritic branches >= 1).
enum class NeuronPolarity : uint8_t {
  EXCITATORY      = 0,
  INHIBITORY      = 1,  // PV+ basket cells (perisomatic, fast)
  INHIBITORY_SST  = 2,  // Somatostatin (dendritic, slow)
  INHIBITORY_VIP  = 3,  // VIP (disinhibits SST)
};

// Pack M: per-cell-type 3D morphology stamped at neuron birth so that
// `Neuron::body` is *the set of "1" voxels forming the cell's actual
// shape* -- not just a single soma voxel that sprouting later expands
// as a random blob. The 2-bit grid was always intended to support real
// neuronal morphology (Markram 2015 *Cell*; Ascoli 2007 *J. Neurosci.*);
// Pack M makes that implicit guarantee explicit. With Pack ZZ active,
// microglial pruning sheds the small surplus that morphology adds, so
// the stamps fit under the substrate budget.
//
// `dx/dy/dz` are integer offsets relative to the soma. `role`:
//   0 DENDRITE   -- NEURON-state voxel, can become SYNAPSE on contact
//   1 AXON       -- NEURON-state voxel, can become SYNAPSE on contact
//   2 AXON_TRUNK -- BLOCKED-state voxel, conducts but does NOT form
//                   synapses (myelinated trunk analogue)
struct MorphologyVoxel {
  int8_t dx, dy, dz;
  uint8_t role;
};

struct MorphologyTemplate {
  const MorphologyVoxel* voxels;
  int n;
};

// Pack TREE: per-body-voxel tree-topology data, parallel to
// `Neuron::body`. The user observed that neurons should "extend
// outward like tree roots" rather than being represented as flat
// occupancy maps. This struct gives every body voxel an explicit
// position in the cell's branching tree:
//   parent_idx  -- index into Neuron::body of the parent voxel,
//                  or UINT16_MAX for the root (soma)
//   depth       -- distance from soma in tree-edges (soma=0,
//                  template voxels stamped from soma=1, sprouted
//                  from a depth-1 voxel=2, etc.)
//   thickness   -- 255 at trunk, decreases distally; matches real
//                  cortex where distal dendrites are thinner due
//                  to actin / microtubule transport falloff.
//                  Used by future leaf-biased sprouting and
//                  distal-preferring microglial pruning passes.
//
// MVP: the field is populated correctly but no phase consults it
// yet for behavioural decisions. Existing flat-body iteration
// continues unchanged. Future packs flip on the directional
// behaviour by reading depth + thickness during sprouting /
// pruning / synaptogenesis branch routing.
struct BranchData {
  uint16_t parent_idx;
  uint8_t  depth;
  uint8_t  thickness;
};
constexpr uint16_t kBranchNoParent = 0xFFFFu;

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

  // Pack TREE: parallel to `body`, one entry per body voxel. Tracks
  // each voxel's position in the cell's branching tree (parent index,
  // depth from soma, thickness). MVP: populated correctly but no
  // phase consults it for behavioural decisions yet -- future packs
  // (leaf-biased sprouting, distal-preferring pruning, branch-index
  // routing in synaptogenesis) flip those on.
  std::vector<BranchData> body_branch;

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
  // Used only when the post neuron has the default single-branch dendrite
  // (n_branches == 1); multi-branch neurons receive directly into
  // `branch_potential` so their compartmental integration is preserved.
  std::vector<float> incoming_queue;

  // Predictive-coding "expected input" for this step. The chemistry
  // phase computes `effective_input = input_acc - predicted_input`
  // before integrating, so a perfectly-predicted stimulus produces
  // zero effective drive (no surprise) while an unpredicted one
  // produces full drive. Cleared each step. The companion API
  // `Simulator::apply_prediction_pattern` writes it from the
  // demo / caller side -- mimics top-down corticothalamic feedback.
  float predicted_input = 0.0f;

  // Number of dendritic branches. Default 1 = legacy single-compartment
  // behaviour. Setting >1 turns this neuron into a multi-compartment cell.
  uint8_t n_branches = 1;

  // Per-branch persistent potential. Sized to `n_branches`. Scheduler
  // delivery for a synapse on branch b adds the spike magnitude to
  // `branch_potential[b]` directly. integrate_incoming_phase converts
  // each branch's accumulated potential into a soma contribution: a
  // dendritic spike if the branch crossed `dendritic_threshold`, an
  // attenuated passive contribution otherwise.
  std::vector<float> branch_potential;

  // Per-branch dendritic-spike threshold. If empty or shorter than
  // `n_branches` the global `cfg.dendritic_threshold` is used for the
  // missing branches. Real cortical pyramidal neurons host different
  // ion-channel densities on different dendritic compartments (apical
  // tuft / basal / proximal) and therefore have distinct local spike
  // thresholds; this lets a demo isolate "innate-prior" branches from
  // "synaptogenesis-default" branches with a single setter call.
  std::vector<float> branch_threshold;

  // Per-branch passive gain (sub-threshold cable leak to the soma). If
  // empty or shorter than `n_branches`, the global
  // `cfg.dendritic_passive_gain` applies. Real biology: a thin distal
  // dendrite leaks much less of a sub-threshold input than a thick
  // proximal one -- electrotonic length matters.
  std::vector<float> branch_passive_gain;

  // Sum of all incoming synapse weights, refreshed by the homeostatic
  // phase. Each pre synapse later reads this scalar and rescales its own
  // weight toward a target -- the simulator analogue of the retrograde
  // BDNF / TNF-alpha signal that real cortical neurons release to balance
  // their total drive (synaptic scaling, Turrigiano 2008).
  float incoming_weight_sum = 0.0f;

  // Long-timescale activity baseline used as the BCM sliding threshold
  // (Bienenstock-Cooper-Munro 1982). LTP only happens when the post
  // fires *above* its own recent average; below that threshold a
  // potentiation event flips sign and becomes LTD. This stabilises the
  // network against runaway potentiation: a chronically over-active
  // neuron raises its own threshold and stops accreting new excitation.
  float activity_baseline = 0.05f;

  // Sum of LTP magnitude received by this neuron's incoming synapses on
  // the current step. Heterosynaptic plasticity then damps every other
  // incoming synapse on the same post in proportion to this -- the
  // post-synaptic density has only so much room for AMPA receptors, and
  // strengthening one synapse forces a cost on its neighbours
  // (heterosynaptic LTD; Royer & Pare 2003).
  float ltp_received_this_step = 0.0f;

  // CREB-mediated intrinsic-excitability bias used during engram
  // allocation (Josselyn & Tonegawa 2020). Real neurons that happen to
  // express elevated CREB at the moment of an experience are
  // preferentially recruited into the engram even if their raw firing
  // rate is not the highest; the molecular state biases candidacy.
  // Multiplies the candidate score in `promote_engram`. Default 1.0
  // = no bias. Demos/curriculum bump this for stimulus-relevant cells
  // (e.g. label-feature INPUTs, target motor column, future A1
  // tonotopic cells) immediately before a teach episode.
  float excitability_bias = 1.0f;
};

}  // namespace snc
