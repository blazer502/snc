#include "snc/trainer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace snc {

Trainer::Trainer(const SNNGraph& graph, const TrainConfig& cfg)
    : g_(graph), cfg_(cfg) {
  classes_ = std::max(1, g_.num_output_channels);

  // Signed weights initialised positive so the frozen graph spikes from step 0.
  std::mt19937_64 rng(cfg_.seed);
  std::uniform_real_distribution<float> wu(0.0f, cfg_.w_init);
  w_.resize(g_.num_synapses());
  for (float& w : w_) w = wu(rng);
  syn_deliv_.assign(g_.num_synapses(), 0);
  neur_fire_.assign(g_.num_neurons, 0);

  // Channel -> input neuron ids (resolved once).
  chan_to_neurons_.assign(g_.num_input_channels, {});
  for (int id : g_.input_neurons) {
    const int c = g_.channel[id];
    if (c >= 0 && c < g_.num_input_channels) chan_to_neurons_[c].push_back(id);
  }

  // Random fixed feedback matrix B for hidden learning signals (broadcast
  // alignment). One row per internal (hidden) neuron.
  internal_idx_.assign(g_.num_neurons, -1);
  int n_internal = 0;
  for (int i = 0; i < g_.num_neurons; ++i)
    if (g_.role[i] == GraphRole::INTERNAL) internal_idx_[i] = n_internal++;
  std::normal_distribution<float> bn(0.0f, cfg_.feedback_scale);
  B_.resize(static_cast<std::size_t>(n_internal) * classes_);
  for (float& b : B_) b = bn(rng);
}

void Trainer::set_weights(const std::vector<float>& w) {
  if (w.size() == w_.size()) w_ = w;
}

void Trainer::reset_stats() {
  std::fill(syn_deliv_.begin(), syn_deliv_.end(), 0);
  std::fill(neur_fire_.begin(), neur_fire_.end(), 0);
}

void Trainer::run_sample(const std::vector<float>& x, uint64_t sample_seed,
                         std::vector<float>& counts, std::vector<double>* elig,
                         long long& spikes, long long& events,
                         std::vector<long long>* syn_deliv,
                         std::vector<long long>* neur_fire) const {
  const int n = g_.num_neurons;
  const auto& rp = g_.row_ptr;
  const auto& post = g_.post_ids;
  const auto& del = g_.delays;
  const float thr = cfg_.lif.threshold;
  const float decay = cfg_.lif.decay;
  const float reset = cfg_.lif.reset;
  const int refr_steps = cfg_.lif.refractory;
  const float gamma = cfg_.surrogate_scale;
  const int T = cfg_.num_steps;

  int max_delay = 1;
  for (int s = 0; s < g_.num_synapses(); ++s)
    max_delay = std::max(max_delay, del[s]);
  const int ring_len = max_delay + 1;

  std::vector<float> v(n, 0.0f), ext(n, 0.0f), tr(n, 0.0f), psi(n, 0.0f);
  std::vector<int> refr(n, 0);
  std::vector<unsigned char> fired(n, 0);
  std::vector<float> ring(static_cast<std::size_t>(ring_len) * n, 0.0f);

  // Encode this sample into a per-step injection schedule.
  auto evs = encode(cfg_.encoder, x.data(), g_.num_input_channels, T, cfg_.enc,
                    sample_seed);
  std::vector<std::vector<std::pair<int, float>>> by_step(T);
  for (const InputEvent& e : evs) {
    if (e.time < 0 || e.time >= T) continue;
    if (e.channel_id < 0 || e.channel_id >= g_.num_input_channels) continue;
    for (int nid : chan_to_neurons_[e.channel_id])
      by_step[e.time].emplace_back(nid, e.amplitude);
  }

  std::fill(counts.begin(), counts.end(), 0.0f);

  for (int t = 0; t < T; ++t) {
    float* slot = &ring[static_cast<std::size_t>(t % ring_len) * n];
    for (const auto& kv : by_step[t]) ext[kv.first] += kv.second;

    // Integrate / fire / surrogate.
    for (int j = 0; j < n; ++j) {
      float vpre;
      if (refr[j] > 0) {
        --refr[j];
        v[j] = reset;
        vpre = reset;
        fired[j] = 0;
      } else {
        v[j] = decay * v[j] + ext[j] + slot[j];
        vpre = v[j];
        if (v[j] >= thr) {
          v[j] = reset;
          refr[j] = refr_steps;
          fired[j] = 1;
        } else {
          fired[j] = 0;
        }
      }
      ext[j] = 0.0f;
      slot[j] = 0.0f;
      // Triangular surrogate derivative, width = threshold.
      const float d = std::fabs(vpre - thr) / (thr > 0 ? thr : 1.0f);
      psi[j] = d < 1.0f ? gamma * (1.0f - d) : 0.0f;
      if (fired[j]) {
        ++spikes;
        if (neur_fire) ++(*neur_fire)[j];
        if (g_.role[j] == GraphRole::OUTPUT && g_.channel[j] >= 0)
          counts[g_.channel[j]] += 1.0f;
      }
    }

    // Low-passed pre-spike trace (bounded ~[0,1] at steady firing).
    for (int j = 0; j < n; ++j)
      tr[j] = decay * tr[j] + (1.0f - decay) * (fired[j] ? 1.0f : 0.0f);

    // Eligibility: E_s += psi_post(s) * tr_pre(s).  (pre = the CSR row owner)
    if (elig) {
      for (int i = 0; i < n; ++i) {
        const float tri = tr[i];
        if (tri == 0.0f) continue;
        for (int e = rp[i]; e < rp[i + 1]; ++e)
          (*elig)[e] += static_cast<double>(psi[post[e]]) * tri;
      }
    }

    // Deliver fired spikes into the future ring slots (signed weight).
    for (int i = 0; i < n; ++i) {
      if (!fired[i]) continue;
      for (int e = rp[i]; e < rp[i + 1]; ++e) {
        const int dt = (t + del[e]) % ring_len;
        ring[static_cast<std::size_t>(dt) * n + post[e]] += w_[e];
        ++events;
        if (syn_deliv) ++(*syn_deliv)[e];
      }
    }
  }
}

void Trainer::apply_update(const std::vector<double>& elig,
                           const std::vector<float>& counts, int label) {
  // Softmax over output-channel spike counts.
  std::vector<float> p(classes_);
  float mx = *std::max_element(counts.begin(), counts.end());
  double sum = 0.0;
  for (int k = 0; k < classes_; ++k) { p[k] = std::exp(counts[k] - mx); sum += p[k]; }
  for (int k = 0; k < classes_; ++k) p[k] /= static_cast<float>(sum);

  // Output error delta_k = p_k - onehot_k.
  std::vector<float> delta(classes_);
  for (int k = 0; k < classes_; ++k) delta[k] = p[k] - (k == label ? 1.0f : 0.0f);

  // Per-neuron learning signal L_j.
  std::vector<float> L(g_.num_neurons, 0.0f);
  for (int j = 0; j < g_.num_neurons; ++j) {
    if (g_.role[j] == GraphRole::OUTPUT && g_.channel[j] >= 0 &&
        g_.channel[j] < classes_) {
      L[j] = delta[g_.channel[j]];
    } else if (cfg_.train_hidden && internal_idx_[j] >= 0) {
      const float* b = &B_[static_cast<std::size_t>(internal_idx_[j]) * classes_];
      float s = 0.0f;
      for (int k = 0; k < classes_; ++k) s += b[k] * delta[k];
      L[j] = s;
    }
  }

  // Gradient-descent step: w_s -= lr * L_post(s) * E_s / T  (T-normalised).
  const float scale = cfg_.lr / std::max(1, cfg_.num_steps);
  const auto& rp = g_.row_ptr;
  for (int i = 0; i < g_.num_neurons; ++i)
    for (int e = rp[i]; e < rp[i + 1]; ++e) {
      const float lpost = L[g_.post_ids[e]];
      if (lpost == 0.0f) continue;
      w_[e] -= scale * lpost * static_cast<float>(elig[e]);
      w_[e] = std::clamp(w_[e], -cfg_.w_max, cfg_.w_max);
    }
}

EpochStats Trainer::train_epoch(const Dataset& train, const Dataset& test,
                                int epoch) {
  EpochStats st;
  std::vector<int> order(train.size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937_64 rng(cfg_.seed ^ (0x9e3779b97f4a7c15ULL * (epoch + 1)));
  std::shuffle(order.begin(), order.end(), rng);

  std::vector<float> counts(classes_, 0.0f);
  std::vector<double> elig(g_.num_synapses());
  double loss_sum = 0.0;
  int correct = 0;

  for (int idx : order) {
    std::fill(elig.begin(), elig.end(), 0.0);
    const uint64_t sseed = cfg_.seed ^ (0xD1B54A32D192ED03ULL * (idx + 1)) ^
                           (static_cast<uint64_t>(epoch) << 32);
    run_sample(train.x[idx], sseed, counts, &elig, st.spikes,
               st.synaptic_events, &syn_deliv_, &neur_fire_);

    // Loss + prediction for reporting.
    float mx = *std::max_element(counts.begin(), counts.end());
    double sum = 0.0;
    for (int k = 0; k < classes_; ++k) sum += std::exp(counts[k] - mx);
    const int y = train.y[idx];
    loss_sum += -(counts[y] - mx) + std::log(sum);
    int pred = 0;
    for (int k = 1; k < classes_; ++k)
      if (counts[k] > counts[pred]) pred = k;
    if (pred == y) ++correct;

    apply_update(elig, counts, y);
  }

  st.loss = train.size() ? loss_sum / train.size() : 0.0;
  st.train_acc = train.size() ? static_cast<double>(correct) / train.size() : 0.0;
  st.test_acc = evaluate(test);
  st.energy = 1.0 * st.spikes + 0.2 * st.synaptic_events;
  return st;
}

double Trainer::evaluate(const Dataset& d) const {
  std::vector<float> counts(classes_, 0.0f);
  long long spikes = 0, events = 0;
  int correct = 0;
  for (int i = 0; i < d.size(); ++i) {
    // Fixed seed per sample so evaluation is deterministic across epochs.
    const uint64_t sseed = cfg_.seed ^ (0xD1B54A32D192ED03ULL * (i + 1));
    run_sample(d.x[i], sseed, counts, nullptr, spikes, events, nullptr, nullptr);
    int pred = 0;
    for (int k = 1; k < classes_; ++k)
      if (counts[k] > counts[pred]) pred = k;
    if (pred == d.y[i]) ++correct;
  }
  return d.size() ? static_cast<double>(correct) / d.size() : 0.0;
}

}  // namespace snc
