// Frozen-structure trainer (new-plan.md sections 8.1 & 8.4).
//
// Trains the synaptic weights of a FIXED SNNGraph topology with e-prop, a
// local, online approximation to BPTT (Bellec et al. 2020). The structural
// prior (which neurons connect to which, and with what delay) is held constant;
// only weights move -- this isolates the value of the topology itself.
//
// Learning rule (per sample, summed over the T-step window):
//   eligibility   E_s   = sum_t  psi_post(s)[t] * tr_pre(s)[t]
//   learn signal  L_j   = output: (softmax_j - onehot_j)          [delta]
//                         hidden: sum_k B[j,k] * delta_k           [random feedback]
//   update        w_s  -= lr * L_post(s) * E_s
// where psi is the surrogate spike derivative and tr is a low-passed pre-spike
// trace. The trainer owns its own SIGNED weight vector (topology + delays come
// from the graph), so learning is decoupled from the graph's Dale-sign layout.

#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "snc/dataset.hpp"
#include "snc/encoders.hpp"
#include "snc/runtime.hpp"  // LIFParams
#include "snc/snn_graph.hpp"

namespace snc {

struct TrainConfig {
  int num_steps = 30;
  LIFParams lif;                       // threshold / decay / reset / refractory
  Encoder encoder = Encoder::Poisson;
  EncoderParams enc;                   // gain / max_rate / threshold
  float lr = 0.02f;
  float w_init = 0.3f;                 // initial weight scale (uniform [0, w_init])
  float w_max = 4.0f;                  // weight clamp magnitude
  float surrogate_scale = 0.3f;        // gamma in the triangular surrogate
  float feedback_scale = 1.0f;         // random-feedback matrix scale
  bool train_hidden = true;            // false => readout-only (reservoir mode)
  // Three-factor neuromodulated learning: instead of a supervised per-output
  // error, sample an action from the output, receive a single scalar reward
  // (right/wrong), and modulate the eligibility by the reward advantage
  // (REINFORCE). The brain's dopamine-gated rule; needs no target vector.
  bool reward_mode = false;
  float reward_baseline_decay = 0.99f; // EMA decay for the reward baseline
  uint64_t seed = 1;
};

struct EpochStats {
  double loss = 0.0;
  double train_acc = 0.0;
  double test_acc = 0.0;
  long long spikes = 0;
  long long synaptic_events = 0;
  double energy = 0.0;                 // alpha*spikes + beta*events
};

class Trainer {
 public:
  Trainer(const SNNGraph& graph, const TrainConfig& cfg);

  // One pass over `train` (SGD, one update per sample), then evaluate on
  // `test`. Samples are visited in a seeded-shuffled order each epoch.
  EpochStats train_epoch(const Dataset& train, const Dataset& test, int epoch);

  // Argmax-accuracy over a dataset using current weights (no learning).
  double evaluate(const Dataset& d) const;

  // Weight carry-over across structural epochs (two-timescale co-training).
  const std::vector<float>& weights() const { return w_; }
  void set_weights(const std::vector<float>& w);  // size must == num_synapses

  // Activity statistics consumed by the structural update. Accumulated during
  // train_epoch (not evaluate); indexed by current-graph synapse / neuron id.
  const std::vector<long long>& syn_deliveries() const { return syn_deliv_; }
  const std::vector<long long>& neuron_fires() const { return neur_fire_; }
  void reset_stats();

 private:
  const SNNGraph& g_;
  TrainConfig cfg_;

  std::vector<float> w_;               // signed weight per synapse
  std::vector<long long> syn_deliv_;   // per-synapse spike deliveries (training)
  std::vector<long long> neur_fire_;   // per-neuron firings (training)
  std::vector<float> B_;               // [num_internal * classes] random feedback
  std::vector<int> internal_idx_;      // internal neuron id -> row in B_ (or -1)
  int classes_ = 0;
  std::mt19937_64 action_rng_;         // samples actions in reward_mode
  float reward_baseline_ = 0.0f;       // EMA reward (neuromodulator reference)

  // Channel -> input neuron ids, resolved once for fast injection.
  std::vector<std::vector<int>> chan_to_neurons_;

  // Run the LIF window for one sample. Fills `counts` (per output channel).
  // When `elig` is non-null, also accumulates per-synapse eligibility E_s and
  // per-neuron surrogate-weighted activity needed for the weight update.
  void run_sample(const std::vector<float>& x, uint64_t sample_seed,
                  std::vector<float>& counts, std::vector<double>* elig,
                  long long& spikes, long long& events,
                  std::vector<long long>* syn_deliv,
                  std::vector<long long>* neur_fire) const;

  void apply_update(const std::vector<double>& elig,
                    const std::vector<float>& counts, int label);
};

}  // namespace snc
