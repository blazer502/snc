#include "snc/connectome.hpp"

#include <algorithm>
#include <cmath>

namespace snc {

namespace {
// Normalised position of the k-th neuron within a layer of `cnt`.
float layer_pos(int k, int cnt) {
  return cnt > 1 ? static_cast<float>(k) / (cnt - 1) : 0.5f;
}
}  // namespace

Connectome Connectome::layered_local(const std::vector<int>& layers,
                                     int synapse_budget, int delay, float w_init,
                                     uint64_t seed) {
  Connectome c;
  c.num_layers_ = static_cast<int>(layers.size());
  for (int v : layers) c.n_ += v;
  c.layer_.resize(c.n_);
  c.pos_.resize(c.n_);
  c.role_.assign(c.n_, GraphRole::INTERNAL);
  c.channel_.assign(c.n_, -1);
  c.adj_.assign(c.n_, {});
  c.layer_begin_.resize(c.num_layers_);
  c.layer_count_.resize(c.num_layers_);
  c.rng_.seed(seed);

  int idx = 0;
  for (int l = 0; l < c.num_layers_; ++l) {
    c.layer_begin_[l] = idx;
    c.layer_count_[l] = layers[l];
    for (int k = 0; k < layers[l]; ++k, ++idx) {
      c.layer_[idx] = l;
      c.pos_[idx] = layer_pos(k, layers[l]);
      if (l == 0) { c.role_[idx] = GraphRole::INPUT; c.channel_[idx] = k; }
      else if (l == c.num_layers_ - 1) { c.role_[idx] = GraphRole::OUTPUT; c.channel_[idx] = k; }
    }
  }

  // Initial local wiring: each pre connects to its nearest `fanout` neurons in
  // the next layer (a local receptive field), mirroring make_static_snc.
  long long forward_pre = 0;
  for (int l = 0; l + 1 < c.num_layers_; ++l) forward_pre += layers[l];
  const int fanout =
      forward_pre > 0 ? std::max(1, static_cast<int>(synapse_budget / forward_pre)) : 0;
  std::uniform_real_distribution<float> jitter(-0.5f, 0.5f);
  for (int l = 0; l + 1 < c.num_layers_; ++l) {
    const int from0 = c.layer_begin_[l], from_n = c.layer_count_[l];
    const int to0 = c.layer_begin_[l + 1], to_n = c.layer_count_[l + 1];
    const int k = std::min(fanout, to_n);
    for (int a = 0; a < from_n; ++a) {
      const float pa = layer_pos(a, from_n) + jitter(c.rng_) / std::max(1, to_n);
      int centre = static_cast<int>(std::lround(pa * (to_n - 1)));
      centre = std::clamp(centre, 0, std::max(0, to_n - 1));
      const int lo = std::clamp(centre - k / 2, 0, std::max(0, to_n - k));
      // Initial edges are immediately prune-eligible (age = protect threshold).
      for (int j = 0; j < k; ++j)
        c.adj_[from0 + a].push_back({to0 + lo + j, delay, w_init, 1 << 20});
    }
  }
  return c;
}

int Connectome::num_synapses() const {
  int m = 0;
  for (const auto& a : adj_) m += static_cast<int>(a.size());
  return m;
}

bool Connectome::connected(int pre, int post) const {
  for (const Edge& e : adj_[pre])
    if (e.post == post) return true;
  return false;
}

SNNGraph Connectome::to_graph() const {
  SNNGraph g;
  g.num_neurons = n_;
  g.role = role_;
  g.channel = channel_;
  g.sign.assign(n_, 1);  // Dale sign unused by the trainer (signed weights)
  g.row_ptr.resize(n_ + 1, 0);
  for (int i = 0; i < n_; ++i)
    g.row_ptr[i + 1] = g.row_ptr[i] + static_cast<int>(adj_[i].size());
  const int m = g.row_ptr[n_];
  g.post_ids.resize(m);
  g.weights.resize(m);
  g.delays.resize(m);
  g.branch_ids.assign(m, 0);
  int s = 0;
  for (int i = 0; i < n_; ++i)
    for (const Edge& e : adj_[i]) {
      g.post_ids[s] = e.post;
      g.weights[s] = std::fabs(e.weight);
      g.delays[s] = e.delay;
      ++s;
    }
  g.rebuild_io_index();
  return g;
}

std::vector<float> Connectome::edge_weights() const {
  std::vector<float> w;
  w.reserve(num_synapses());
  for (int i = 0; i < n_; ++i)
    for (const Edge& e : adj_[i]) w.push_back(e.weight);
  return w;
}

void Connectome::set_edge_weights(const std::vector<float>& w) {
  std::size_t idx = 0;
  for (int i = 0; i < n_; ++i)
    for (Edge& e : adj_[i]) {
      if (idx >= w.size()) return;
      e.weight = w[idx++];
    }
}

StructReport Connectome::structural_update(const ActivityStats& stats,
                                           const StructConfig& cfg, float reward) {
  // Reward-modulated rewiring rate: competence = how far reward is above chance,
  // in [0,1]. Rewire a (1 - competence) fraction of grow_per_epoch (floored), so
  // the clock searches topology aggressively when reward is low and anneals to a
  // trickle as the circuit masters the task.
  int grow_target = cfg.grow_per_epoch;
  if (cfg.reward_modulated) {
    const float denom = std::max(1e-3f, 1.0f - cfg.reward_chance);
    const float comp = std::clamp((reward - cfg.reward_chance) / denom, 0.0f, 1.0f);
    const float frac = std::clamp(1.0f - comp, cfg.reward_floor, 1.0f);
    grow_target = std::max(0, static_cast<int>(std::lround(cfg.grow_per_epoch * frac)));
  }

  // --- Prune: K weakest synapses by |weight|, among prune-eligible ones. ---
  struct Cand { float score; int i; int k; };
  std::vector<Cand> cand;
  for (int i = 0; i < n_; ++i)
    for (int k = 0; k < static_cast<int>(adj_[i].size()); ++k)
      if (adj_[i][k].age >= cfg.protect_rounds)
        cand.push_back({std::fabs(adj_[i][k].weight), i, k});

  int K = std::min(grow_target, static_cast<int>(cand.size()));
  if (K > 0) {
    std::nth_element(cand.begin(), cand.begin() + K, cand.end(),
                     [](const Cand& a, const Cand& b) { return a.score < b.score; });
  }
  std::vector<std::vector<char>> rm(n_);
  for (int i = 0; i < n_; ++i) rm[i].assign(adj_[i].size(), 0);
  for (int j = 0; j < K; ++j) rm[cand[j].i][cand[j].k] = 1;
  int pruned = 0;
  std::vector<int> freed_posts;  // posts that lost a synapse -> regrow targets
  freed_posts.reserve(K);
  for (int i = 0; i < n_; ++i) {
    std::vector<Edge> keep;
    keep.reserve(adj_[i].size());
    for (int k = 0; k < static_cast<int>(adj_[i].size()); ++k) {
      if (rm[i][k]) { ++pruned; freed_posts.push_back(adj_[i][k].post); }
      else keep.push_back(adj_[i][k]);
    }
    adj_[i].swap(keep);
  }

  // --- Grow: replace each pruned synapse with a local, demand-driven one. ---
  const auto& fire = stats.neuron_fires;
  auto fire_of = [&](int id) {
    return id < static_cast<int>(fire.size()) ? fire[id] : 0LL;
  };
  // Most-active local pre not yet wired to post p (positions are monotonic in
  // index, so the window is a contiguous index range). Returns pre id or -1.
  auto grow_local = [&](int p) -> int {
    const int pre_l = layer_[p] - 1;
    if (pre_l < 0) return -1;
    const int b0 = layer_begin_[pre_l], bn = layer_count_[pre_l];
    const float w = cfg.locality_window;
    int klo = std::clamp(static_cast<int>(std::ceil((pos_[p] - w) * (bn - 1))), 0, bn - 1);
    int khi = std::clamp(static_cast<int>(std::floor((pos_[p] + w) * (bn - 1))), 0, bn - 1);
    int best = -1;
    long long best_fire = -1;
    for (int kk = klo; kk <= khi; ++kk) {
      const int q = b0 + kk;
      if (connected(q, p)) continue;
      if (fire_of(q) > best_fire) { best_fire = fire_of(q); best = q; }
    }
    return best;
  };

  int grown = 0;
  // 1:1 replacement -- grow a fresh local input for each post that lost one.
  for (int p : freed_posts) {
    if (grown >= pruned) break;
    const int q = grow_local(p);
    if (q >= 0) { adj_[q].push_back({p, cfg.delay, cfg.w_grow_init, 0}); ++grown; }
  }
  // Fallback for posts with no free local pre: grow onto under-driven neurons.
  std::uniform_int_distribution<int> any_post(
      layer_begin_.size() > 1 ? layer_begin_[1] : 0, std::max(0, n_ - 1));
  int attempts = 0;
  const int max_attempts = cfg.max_attempts_mult * std::max(1, pruned);
  while (grown < pruned && attempts < max_attempts) {
    ++attempts;
    int p = -1;
    for (int t = 0; t < 4; ++t) {
      const int q = any_post(rng_);
      if (layer_[q] == 0) continue;
      if (p < 0 || fire_of(q) < fire_of(p)) p = q;
    }
    if (p < 0) continue;
    const int q = grow_local(p);
    if (q >= 0) { adj_[q].push_back({p, cfg.delay, cfg.w_grow_init, 0}); ++grown; }
  }

  // Age every surviving / new synapse by one structural round.
  for (int i = 0; i < n_; ++i)
    for (Edge& e : adj_[i])
      if (e.age < (1 << 20)) ++e.age;

  return {pruned, grown, num_synapses()};
}

}  // namespace snc
