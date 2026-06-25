// Mutable structural state for two-timescale co-training (new-plan.md 3.3 & 8.2).
//
// The Connectome is the SLOW-path object: a layered, position-aware edge list
// that grows / prunes / rewires between weight-training rounds. It compiles to
// an immutable SNNGraph (CSR) for the FAST inner loop, and accepts trained
// weights back so they survive across structural epochs.
//
// Structural rule (budget-constant rewiring, DeepR-style but locality-bound):
//   * prune  the K weakest synapses by |weight| (skipping ones too young to
//             have been trained), and
//   * grow   K new synapses, each connecting spatially-near neurons in adjacent
//             layers, biased toward under-driven post and active local pre.
// Total synapse count is held ~constant, so dynamic vs static structure can be
// compared at an equal budget.

#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "snc/snn_graph.hpp"

namespace snc {

// Activity collected during the inner training loop, consumed by the structural
// update. Indexed by the CURRENT graph's synapse / neuron ids (to_graph order).
struct ActivityStats {
  std::vector<long long> syn_deliveries;
  std::vector<long long> neuron_fires;
};

struct StructConfig {
  int grow_per_epoch = 200;       // synapses pruned and regrown per structural step
  int delay = 1;                  // delay assigned to new synapses
  float w_grow_init = 0.02f;      // initial weight of a new synapse
  float locality_window = 0.2f;   // normalized position window for local growth
  int protect_rounds = 2;         // structural rounds a new synapse is prune-immune
  int max_attempts_mult = 8;      // growth attempts = mult * grow_per_epoch
  // Reward-modulated rewiring: scale the rewire count by recent competence so
  // the structural clock explores topology when reward is low and consolidates
  // (anneals plasticity, critical-period-like) as reward rises. Same reward that
  // drives the three-factor weight rule -> topology and weights co-learn from it.
  bool reward_modulated = false;
  float reward_chance = 0.1f;     // reward level treated as zero competence
  float reward_floor = 0.1f;      // min fraction of grow_per_epoch kept when mastered
  uint64_t seed = 1;
};

struct StructReport {
  int pruned = 0;
  int grown = 0;
  int synapses = 0;
};

class Connectome {
 public:
  // Layered local seed: same role/position convention as make_static_snc, kept
  // as a mutable, position-aware edge list (weights start at +w_init).
  static Connectome layered_local(const std::vector<int>& layers,
                                  int synapse_budget, int delay, float w_init,
                                  uint64_t seed);

  SNNGraph to_graph() const;                          // CSR snapshot
  std::vector<float> edge_weights() const;            // weights in to_graph order
  void set_edge_weights(const std::vector<float>& w); // write back (same order)

  // Prune + grow one structural step using the inner loop's activity stats.
  // `reward` in [0,1] is the recent performance/reward; when cfg.reward_modulated
  // it scales the rewire count (low reward -> more rewiring).
  StructReport structural_update(const ActivityStats& stats,
                                 const StructConfig& cfg, float reward = 1.0f);

  int num_neurons() const { return n_; }
  int num_synapses() const;

 private:
  struct Edge {
    int post;
    int delay;
    float weight;
    int age;  // structural rounds survived (growth = 0)
  };

  int n_ = 0;
  int num_layers_ = 0;
  std::vector<int> layer_;            // [n_] layer index per neuron
  std::vector<float> pos_;            // [n_] normalized position within layer
  std::vector<GraphRole> role_;       // [n_]
  std::vector<int> channel_;          // [n_]
  std::vector<int> layer_begin_;      // [num_layers_] first neuron id of layer
  std::vector<int> layer_count_;      // [num_layers_]
  std::vector<std::vector<Edge>> adj_;  // [n_] outgoing edges per pre neuron
  std::mt19937_64 rng_;

  bool connected(int pre, int post) const;
};

}  // namespace snc
