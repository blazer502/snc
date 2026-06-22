#include <algorithm>
#include <chrono>
#include <vector>

#include "snc/encoders.hpp"
#include "snc/runtime.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace snc {

bool parse_backend(const std::string& name, Backend& out) {
  if (name == "cpu") { out = Backend::Cpu; return true; }
  if (name == "openmp") { out = Backend::OpenMP; return true; }
  if (name == "cuda-atomic") { out = Backend::CudaAtomic; return true; }
  if (name == "cuda-bucket") { out = Backend::CudaBucket; return true; }
  if (name == "cuda-sort") { out = Backend::CudaSort; return true; }
  return false;
}

const char* backend_name(Backend b) {
  switch (b) {
    case Backend::Cpu: return "cpu";
    case Backend::OpenMP: return "openmp";
    case Backend::CudaAtomic: return "cuda-atomic";
    case Backend::CudaBucket: return "cuda-bucket";
    case Backend::CudaSort: return "cuda-sort";
  }
  return "?";
}

bool backend_is_cuda(Backend b) {
  return b == Backend::CudaAtomic || b == Backend::CudaBucket ||
         b == Backend::CudaSort;
}

namespace {

// Pre-expand a sample's events into a per-step list of (neuron, amplitude)
// deltas: resolves channel -> INPUT-neuron(s) once so the hot loop just adds.
struct InjectSchedule {
  std::vector<std::vector<std::pair<int, float>>> by_step;  // [num_steps]
};

InjectSchedule build_schedule(const SNNGraph& g,
                              const std::vector<InputEvent>& events,
                              int num_steps) {
  // channel -> input neuron ids (a channel may drive several input neurons).
  std::vector<std::vector<int>> chan_to_neurons(g.num_input_channels);
  for (int id : g.input_neurons) {
    const int c = g.channel[id];
    if (c >= 0 && c < g.num_input_channels) chan_to_neurons[c].push_back(id);
  }
  InjectSchedule sch;
  sch.by_step.resize(num_steps);
  for (const InputEvent& e : events) {
    if (e.time < 0 || e.time >= num_steps) continue;
    if (e.channel_id < 0 || e.channel_id >= g.num_input_channels) continue;
    for (int nid : chan_to_neurons[e.channel_id])
      sch.by_step[e.time].emplace_back(nid, e.amplitude);
  }
  return sch;
}

ForwardResult run_cpu(const SNNGraph& g, const std::vector<InputEvent>& events,
                      int num_steps, const LIFParams& lif, bool use_omp) {
  const int n = g.num_neurons;
  ForwardResult res;
  res.logits.assign(std::max(0, g.num_output_channels), 0.0f);
  res.used_backend = use_omp ? Backend::OpenMP : Backend::Cpu;
  if (n == 0 || num_steps <= 0) return res;

  // Ring buffer sized one past the largest delay so a delivery scheduled at
  // t+delay never lands on the slot being read (and cleared) at step t.
  int max_delay = 1;
  for (int s = 0; s < g.num_synapses(); ++s)
    max_delay = std::max(max_delay, g.delays[s]);
  const int ring_len = max_delay + 1;

  std::vector<float> v(n, 0.0f);
  std::vector<float> ext(n, 0.0f);        // external drive accumulated per step
  std::vector<int> refr(n, 0);
  std::vector<uint8_t> fired(n, 0);
  std::vector<float> ring(static_cast<std::size_t>(ring_len) * n, 0.0f);
  std::vector<int> fired_list;
  fired_list.reserve(n);

  const InjectSchedule sch = build_schedule(g, events, num_steps);

  long long spikes = 0, syn_events = 0;
  const auto t0 = std::chrono::steady_clock::now();

  for (int t = 0; t < num_steps; ++t) {
    float* slot = &ring[static_cast<std::size_t>(t % ring_len) * n];

    // Inject external drive for this step.
    for (const auto& kv : sch.by_step[t]) ext[kv.first] += kv.second;

    // Integrate + fire (independent per neuron).
    fired_list.clear();
#ifdef _OPENMP
    if (use_omp) {
#pragma omp parallel for schedule(static)
      for (int i = 0; i < n; ++i) {
        if (refr[i] > 0) {
          --refr[i];
          v[i] = lif.reset;
          fired[i] = 0;
        } else {
          v[i] = v[i] * lif.decay + ext[i] + slot[i];
          if (v[i] >= lif.threshold) {
            v[i] = lif.reset;
            refr[i] = lif.refractory;
            fired[i] = 1;
          } else {
            fired[i] = 0;
          }
        }
        ext[i] = 0.0f;
        slot[i] = 0.0f;
      }
      for (int i = 0; i < n; ++i)
        if (fired[i]) fired_list.push_back(i);
    } else
#endif
    {
      for (int i = 0; i < n; ++i) {
        if (refr[i] > 0) {
          --refr[i];
          v[i] = lif.reset;
        } else {
          v[i] = v[i] * lif.decay + ext[i] + slot[i];
          if (v[i] >= lif.threshold) {
            v[i] = lif.reset;
            refr[i] = lif.refractory;
            fired_list.push_back(i);
          }
        }
        ext[i] = 0.0f;
        slot[i] = 0.0f;
      }
    }

    spikes += static_cast<long long>(fired_list.size());

    // Decode: count output-neuron spikes into per-channel logits.
    for (int i : fired_list)
      if (g.role[i] == GraphRole::OUTPUT && g.channel[i] >= 0)
        res.logits[g.channel[i]] += 1.0f;

    // Expand fired neurons' outgoing synapses into the delay ring.
#ifdef _OPENMP
    if (use_omp) {
      long long local_events = 0;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : local_events)
      for (int fi = 0; fi < static_cast<int>(fired_list.size()); ++fi) {
        const int i = fired_list[fi];
        const float s = static_cast<float>(g.sign[i]);
        for (int e = g.row_ptr[i]; e < g.row_ptr[i + 1]; ++e) {
          const int dt = (t + g.delays[e]) % ring_len;
          float* tgt = &ring[static_cast<std::size_t>(dt) * n + g.post_ids[e]];
#pragma omp atomic
          *tgt += s * g.weights[e];
          ++local_events;
        }
      }
      syn_events += local_events;
    } else
#endif
    {
      for (int i : fired_list) {
        const float s = static_cast<float>(g.sign[i]);
        for (int e = g.row_ptr[i]; e < g.row_ptr[i + 1]; ++e) {
          const int dt = (t + g.delays[e]) % ring_len;
          ring[static_cast<std::size_t>(dt) * n + g.post_ids[e]] +=
              s * g.weights[e];
          ++syn_events;
        }
      }
    }
  }

  const auto t1 = std::chrono::steady_clock::now();
  res.stats.spikes = spikes;
  res.stats.synaptic_events = syn_events;
  res.stats.ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  return res;
}

}  // namespace

ForwardResult forward(const SNNGraph& g, const std::vector<InputEvent>& events,
                      int num_steps, const LIFParams& lif, Backend backend) {
  if (backend_is_cuda(backend)) {
    ForwardResult out;
    if (cuda::available() &&
        cuda::forward(g, events, num_steps, lif, backend, out))
      return out;
    // Fall back to deterministic CPU; used_backend reports Cpu so the caller
    // can see the GPU path was unavailable.
    return run_cpu(g, events, num_steps, lif, /*use_omp=*/false);
  }
  return run_cpu(g, events, num_steps, lif, backend == Backend::OpenMP);
}

}  // namespace snc
