#include "snc/snn_graph.hpp"

#include <algorithm>
#include <cstdio>
#include <random>
#include <unordered_map>

#include "neuron.hpp"
#include "simulator.hpp"

namespace snc {

void SNNGraph::rebuild_io_index() {
  input_neurons.clear();
  output_neurons.clear();
  int max_in = -1, max_out = -1;
  for (int i = 0; i < num_neurons; ++i) {
    if (role[i] == GraphRole::INPUT) {
      input_neurons.push_back(i);
      max_in = std::max(max_in, channel[i]);
    } else if (role[i] == GraphRole::OUTPUT) {
      output_neurons.push_back(i);
      max_out = std::max(max_out, channel[i]);
    }
  }
  num_input_channels = max_in + 1;
  num_output_channels = max_out + 1;
}

bool SNNGraph::validate(std::string& err) const {
  err.clear();
  auto fail = [&](const char* m) { err = m; return false; };
  if (num_neurons < 0) return fail("negative num_neurons");
  if (static_cast<int>(row_ptr.size()) != num_neurons + 1)
    return fail("row_ptr size != num_neurons + 1");
  if (static_cast<int>(role.size()) != num_neurons ||
      static_cast<int>(sign.size()) != num_neurons ||
      static_cast<int>(channel.size()) != num_neurons)
    return fail("per-neuron metadata size mismatch");
  const int m = num_synapses();
  if (static_cast<int>(weights.size()) != m ||
      static_cast<int>(delays.size()) != m ||
      static_cast<int>(branch_ids.size()) != m)
    return fail("CSR column array size mismatch");
  if (num_neurons > 0 && row_ptr.front() != 0) return fail("row_ptr[0] != 0");
  if (!row_ptr.empty() && row_ptr.back() != m)
    return fail("row_ptr.back() != num_synapses");
  for (int i = 0; i < num_neurons; ++i)
    if (row_ptr[i] > row_ptr[i + 1]) return fail("row_ptr not monotone");
  for (int s = 0; s < m; ++s) {
    if (post_ids[s] < 0 || post_ids[s] >= num_neurons)
      return fail("post id out of range");
    if (delays[s] < 1) return fail("delay < 1");
  }
  return true;
}

GraphStats compute_stats(const SNNGraph& g) {
  GraphStats s;
  s.num_neurons = g.num_neurons;
  s.num_synapses = g.num_synapses();
  std::vector<int> fan_in(g.num_neurons, 0);
  double sum_delay = 0.0, sum_weight = 0.0;
  for (int i = 0; i < g.num_neurons; ++i) {
    const int fo = g.row_ptr[i + 1] - g.row_ptr[i];
    s.max_fan_out = std::max(s.max_fan_out, fo);
    if (g.role[i] == GraphRole::INPUT) ++s.num_inputs;
    else if (g.role[i] == GraphRole::OUTPUT) ++s.num_outputs;
  }
  for (int sId = 0; sId < s.num_synapses; ++sId) {
    ++fan_in[g.post_ids[sId]];
    sum_delay += g.delays[sId];
    sum_weight += g.weights[sId];
  }
  for (int i = 0; i < g.num_neurons; ++i)
    s.max_fan_in = std::max(s.max_fan_in, fan_in[i]);
  if (g.num_neurons > 0) {
    s.avg_fan_out = static_cast<double>(s.num_synapses) / g.num_neurons;
    s.avg_fan_in = s.avg_fan_out;  // sum(fan_in) == sum(fan_out) == synapses
  }
  if (s.num_synapses > 0) {
    s.avg_delay = sum_delay / s.num_synapses;
    s.avg_weight = sum_weight / s.num_synapses;
  }
  const double possible =
      static_cast<double>(g.num_neurons) * (g.num_neurons - 1);
  s.density = possible > 0 ? s.num_synapses / possible : 0.0;
  // CSR columns + per-neuron metadata, the device-resident hot footprint.
  s.bytes = sizeof(int) * (g.row_ptr.size() + g.post_ids.size() +
                           g.delays.size()) +
            sizeof(float) * g.weights.size() + g.branch_ids.size() +
            g.role.size() + g.sign.size() + sizeof(int) * g.channel.size();
  return s;
}

std::string format_stats(const GraphStats& s) {
  char buf[640];
  std::snprintf(
      buf, sizeof(buf),
      "neurons=%d synapses=%d (in=%d out=%d)\n"
      "  fan-out avg=%.2f max=%d | fan-in avg=%.2f max=%d\n"
      "  delay avg=%.2f | weight avg=%.4f | density=%.4g | mem=%.2f MB",
      s.num_neurons, s.num_synapses, s.num_inputs, s.num_outputs, s.avg_fan_out,
      s.max_fan_out, s.avg_fan_in, s.max_fan_in, s.avg_delay, s.avg_weight,
      s.density, s.bytes / (1024.0 * 1024.0));
  return std::string(buf);
}

// ---- Generators -----------------------------------------------------------

namespace {

int8_t sign_of(NeuronPolarity p) {
  return p == NeuronPolarity::EXCITATORY ? int8_t{1} : int8_t{-1};
}

GraphRole role_of(NeuronRole r) {
  switch (r) {
    case NeuronRole::INPUT: return GraphRole::INPUT;
    case NeuronRole::OUTPUT: return GraphRole::OUTPUT;
    default: return GraphRole::INTERNAL;
  }
}

// Build CSR from a per-neuron adjacency list of (post, weight, delay, branch).
struct Edge {
  int post;
  float weight;
  int delay;
  uint8_t branch;
};

// Assign roles/channels for a layered graph: layer 0 -> INPUT, last -> OUTPUT.
void layered_roles(const std::vector<int>& layers, std::vector<GraphRole>& role,
                   std::vector<int>& channel) {
  const int L = static_cast<int>(layers.size());
  int idx = 0;
  for (int l = 0; l < L; ++l) {
    for (int k = 0; k < layers[l]; ++k, ++idx) {
      if (l == 0) {
        role[idx] = GraphRole::INPUT;
        channel[idx] = k;
      } else if (l == L - 1) {
        role[idx] = GraphRole::OUTPUT;
        channel[idx] = k;
      } else {
        role[idx] = GraphRole::INTERNAL;
        channel[idx] = -1;
      }
    }
  }
}

SNNGraph finalize(int n, std::vector<std::vector<Edge>>& adj,
                  std::vector<GraphRole>&& role, std::vector<int8_t>&& sign,
                  std::vector<int>&& channel) {
  SNNGraph g;
  g.num_neurons = n;
  g.role = std::move(role);
  g.sign = std::move(sign);
  g.channel = std::move(channel);
  g.row_ptr.resize(n + 1, 0);
  for (int i = 0; i < n; ++i) g.row_ptr[i + 1] = g.row_ptr[i] + (int)adj[i].size();
  const int m = g.row_ptr[n];
  g.post_ids.resize(m);
  g.weights.resize(m);
  g.delays.resize(m);
  g.branch_ids.resize(m);
  for (int i = 0; i < n; ++i) {
    int s = g.row_ptr[i];
    for (const Edge& e : adj[i]) {
      g.post_ids[s] = e.post;
      g.weights[s] = e.weight;
      g.delays[s] = e.delay;
      g.branch_ids[s] = e.branch;
      ++s;
    }
  }
  g.rebuild_io_index();
  return g;
}

}  // namespace

SNNGraph make_static_snc(const std::vector<int>& layers, int synapse_budget,
                         float weight, int delay, float inhibitory_fraction,
                         uint64_t seed) {
  int n = 0;
  for (int c : layers) n += c;
  std::vector<GraphRole> role(n, GraphRole::INTERNAL);
  std::vector<int8_t> sign(n, 1);
  std::vector<int> channel(n, -1);
  layered_roles(layers, role, channel);

  // Number of pre-synaptic neurons across all forward layers; used to turn a
  // synapse budget into a per-neuron local fan-out.
  long long num_forward_pre = 0;
  for (std::size_t l = 0; l + 1 < layers.size(); ++l) num_forward_pre += layers[l];
  int fanout = num_forward_pre > 0
                   ? std::max(1, static_cast<int>(synapse_budget / num_forward_pre))
                   : 0;

  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::uniform_real_distribution<float> jitter(-0.5f, 0.5f);

  for (int i = 0; i < n; ++i)
    if (role[i] != GraphRole::INPUT && uni(rng) < inhibitory_fraction)
      sign[i] = -1;

  // 1-D normalised position within each layer (so "nearest in next layer"
  // is well defined). A pre at position p connects to the `fanout` next-layer
  // neurons whose positions are closest to p -- a local receptive field.
  std::vector<std::vector<Edge>> adj(n);
  int base = 0;
  for (std::size_t l = 0; l + 1 < layers.size(); ++l) {
    const int from0 = base, from_n = layers[l];
    const int to0 = base + from_n, to_n = layers[l + 1];
    auto pos = [](int k, int cnt) {
      return cnt > 1 ? static_cast<double>(k) / (cnt - 1) : 0.5;
    };
    const int k = std::min(fanout, to_n);
    for (int a = 0; a < from_n; ++a) {
      const double pa = pos(a, from_n) + jitter(rng) / std::max(1, to_n);
      // Map the pre position onto the next layer and take a contiguous local
      // window of `k` neurons centred there -- the nearest-by-position set.
      int centre = static_cast<int>(std::lround(pa * (to_n - 1)));
      centre = std::clamp(centre, 0, std::max(0, to_n - 1));
      int lo = std::clamp(centre - k / 2, 0, std::max(0, to_n - k));
      for (int j = 0; j < k; ++j)
        adj[from0 + a].push_back({to0 + lo + j, weight, delay, 0});
    }
    base += from_n;
  }
  return finalize(n, adj, std::move(role), std::move(sign), std::move(channel));
}

SNNGraph compile_from_simulator(const Simulator& sim) {
  const std::vector<Neuron>& neurons = sim.neurons();
  const int n = static_cast<int>(neurons.size());

  // Neuron ids are 1-based and may be non-contiguous; map to dense [0, n).
  std::unordered_map<uint32_t, int> id_to_idx;
  id_to_idx.reserve(n * 2);
  std::vector<GraphRole> role(n);
  std::vector<int8_t> sign(n);
  std::vector<int> channel(n);
  for (int i = 0; i < n; ++i) {
    id_to_idx[neurons[i].id] = i;
    role[i] = role_of(neurons[i].role);
    sign[i] = sign_of(neurons[i].polarity);
    channel[i] = neurons[i].channel;
  }

  std::vector<std::vector<Edge>> adj(n);
  for (int i = 0; i < n; ++i) {
    for (const SynapseEdge& e : neurons[i].outgoing) {
      auto it = id_to_idx.find(e.target_neuron);
      if (it == id_to_idx.end()) continue;  // dangling target; skip defensively
      const int delay = e.conduction_delay >= 1 ? e.conduction_delay : 1;
      adj[i].push_back({it->second, e.weight, delay, e.branch});
    }
  }
  return finalize(n, adj, std::move(role), std::move(sign), std::move(channel));
}

SNNGraph make_dense(const std::vector<int>& layers, float weight, int delay) {
  int n = 0;
  for (int c : layers) n += c;
  std::vector<GraphRole> role(n, GraphRole::INTERNAL);
  std::vector<int8_t> sign(n, 1);
  std::vector<int> channel(n, -1);
  layered_roles(layers, role, channel);

  std::vector<std::vector<Edge>> adj(n);
  int base = 0;
  for (std::size_t l = 0; l + 1 < layers.size(); ++l) {
    const int from0 = base, to0 = base + layers[l];
    const int to1 = to0 + layers[l + 1];
    for (int a = from0; a < to0; ++a)
      for (int b = to0; b < to1; ++b)
        adj[a].push_back({b, weight, delay, 0});
    base += layers[l];
  }
  return finalize(n, adj, std::move(role), std::move(sign), std::move(channel));
}

SNNGraph make_random_sparse(const std::vector<int>& layers, int synapse_budget,
                            float weight, int delay, float inhibitory_fraction,
                            uint64_t seed) {
  int n = 0;
  for (int c : layers) n += c;
  std::vector<GraphRole> role(n, GraphRole::INTERNAL);
  std::vector<int8_t> sign(n, 1);
  std::vector<int> channel(n, -1);
  layered_roles(layers, role, channel);

  // Count the total number of forward (layer L -> L+1) edge slots so we can
  // turn a synapse budget into a per-edge keep-probability.
  long long possible = 0;
  for (std::size_t l = 0; l + 1 < layers.size(); ++l)
    possible += static_cast<long long>(layers[l]) * layers[l + 1];
  const double keep_p =
      possible > 0 ? std::min(1.0, static_cast<double>(synapse_budget) / possible)
                   : 0.0;

  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> uni(0.0, 1.0);

  // Dale's principle: mark inhibitory_fraction of non-INPUT neurons inhibitory.
  for (int i = 0; i < n; ++i)
    if (role[i] != GraphRole::INPUT && uni(rng) < inhibitory_fraction)
      sign[i] = -1;

  std::vector<std::vector<Edge>> adj(n);
  int base = 0;
  for (std::size_t l = 0; l + 1 < layers.size(); ++l) {
    const int from0 = base, to0 = base + layers[l];
    const int to1 = to0 + layers[l + 1];
    for (int a = from0; a < to0; ++a)
      for (int b = to0; b < to1; ++b)
        if (uni(rng) < keep_p) adj[a].push_back({b, weight, delay, 0});
    base += layers[l];
  }
  return finalize(n, adj, std::move(role), std::move(sign), std::move(channel));
}

}  // namespace snc
