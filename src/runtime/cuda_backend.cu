// CUDA backend -- Stage 1 "atomic" forward fast path (new-plan.md section 4.2).
//
// This implements the naive-but-correct event delivery from the plan: every
// fired neuron expands its outgoing synapses and atomicAdds each weighted spike
// into a delay ring buffer. It is the reference the bucketed/sorted reduction
// backends (next PR) are validated against. cuda-bucket / cuda-sort currently
// alias this kernel so the CLI surface is complete; the dedicated reduction
// kernels land in Phase 4.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <vector>

#include "snc/encoders.hpp"
#include "snc/runtime.hpp"

namespace snc {
namespace cuda {
namespace {

#define CUDA_OK(call)                       \
  do {                                      \
    if ((call) != cudaSuccess) return false; \
  } while (0)

__global__ void k_inject(const int* inj_neuron, const float* inj_amp, int begin,
                         int end, float* ext) {
  int j = begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= end) return;
  atomicAdd(&ext[inj_neuron[j]], inj_amp[j]);
}

// Fused integrate / fire / expand step. One thread per neuron.
__global__ void k_step(int t, int n, int ring_len, float threshold, float decay,
                       float reset, int refractory, const int* row_ptr,
                       const int* post_ids, const float* weights,
                       const int* delays, const signed char* sign,
                       const unsigned char* role, const int* channel, float* v,
                       float* ext, int* refr, float* ring, int* chan_count,
                       unsigned long long* spikes,
                       unsigned long long* syn_events) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int cur = (t % ring_len) * n + i;
  bool fired = false;
  if (refr[i] > 0) {
    --refr[i];
    v[i] = reset;
  } else {
    float nv = v[i] * decay + ext[i] + ring[cur];
    if (nv >= threshold) {
      v[i] = reset;
      refr[i] = refractory;
      fired = true;
    } else {
      v[i] = nv;
    }
  }
  ext[i] = 0.0f;
  ring[cur] = 0.0f;
  if (!fired) return;

  atomicAdd(spikes, 1ULL);
  if (role[i] == 2 /*OUTPUT*/ && channel[i] >= 0)
    atomicAdd(&chan_count[channel[i]], 1);

  const float s = static_cast<float>(sign[i]);
  const int beg = row_ptr[i], fin = row_ptr[i + 1];
  for (int e = beg; e < fin; ++e) {
    const int dt = (t + delays[e]) % ring_len;
    atomicAdd(&ring[dt * n + post_ids[e]], s * weights[e]);
  }
  if (fin > beg) atomicAdd(syn_events, (unsigned long long)(fin - beg));
}

template <typename T>
bool dev_alloc_copy(T** dptr, const T* host, size_t count) {
  if (cudaMalloc(dptr, count * sizeof(T)) != cudaSuccess) return false;
  if (count == 0) return true;
  return cudaMemcpy(*dptr, host, count * sizeof(T), cudaMemcpyHostToDevice) ==
         cudaSuccess;
}

}  // namespace

bool available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

bool forward(const SNNGraph& g, const std::vector<InputEvent>& events,
             int num_steps, const LIFParams& lif, Backend /*variant*/,
             ForwardResult& out) {
  const int n = g.num_neurons;
  const int oc = g.num_output_channels;
  if (n == 0 || num_steps <= 0) return false;

  int max_delay = 1;
  for (int s = 0; s < g.num_synapses(); ++s)
    if (g.delays[s] > max_delay) max_delay = g.delays[s];
  const int ring_len = max_delay + 1;

  // Flatten the injection schedule (channel -> input neurons) into per-step
  // CSR ranges so the device can apply each step's external drive in one pass.
  std::vector<std::vector<int>> chan_to_neurons(g.num_input_channels);
  for (int id : g.input_neurons) {
    const int c = g.channel[id];
    if (c >= 0 && c < g.num_input_channels) chan_to_neurons[c].push_back(id);
  }
  std::vector<std::vector<std::pair<int, float>>> by_step(num_steps);
  for (const InputEvent& e : events) {
    if (e.time < 0 || e.time >= num_steps) continue;
    if (e.channel_id < 0 || e.channel_id >= g.num_input_channels) continue;
    for (int nid : chan_to_neurons[e.channel_id])
      by_step[e.time].emplace_back(nid, e.amplitude);
  }
  std::vector<int> inj_row(num_steps + 1, 0);
  std::vector<int> inj_neuron;
  std::vector<float> inj_amp;
  for (int t = 0; t < num_steps; ++t) {
    inj_row[t + 1] = inj_row[t] + static_cast<int>(by_step[t].size());
    for (auto& kv : by_step[t]) {
      inj_neuron.push_back(kv.first);
      inj_amp.push_back(kv.second);
    }
  }

  // Convert per-neuron metadata to device-friendly POD arrays.
  std::vector<signed char> sign(n);
  std::vector<unsigned char> role(n);
  for (int i = 0; i < n; ++i) {
    sign[i] = static_cast<signed char>(g.sign[i]);
    role[i] = static_cast<unsigned char>(g.role[i]);
  }

  // Device buffers (manual lifetime; CUDA_OK early-returns leak the few that
  // were already allocated -- acceptable for a fail-fast fallback path).
  int *d_row = nullptr, *d_post = nullptr, *d_delays = nullptr,
      *d_channel = nullptr, *d_refr = nullptr, *d_chan = nullptr,
      *d_inj_n = nullptr;
  float *d_w = nullptr, *d_v = nullptr, *d_ext = nullptr, *d_ring = nullptr,
        *d_inj_a = nullptr;
  signed char* d_sign = nullptr;
  unsigned char* d_role = nullptr;
  unsigned long long *d_spikes = nullptr, *d_syn = nullptr;

  bool ok = true;
  ok = ok && dev_alloc_copy(&d_row, g.row_ptr.data(), g.row_ptr.size());
  ok = ok && dev_alloc_copy(&d_post, g.post_ids.data(), g.post_ids.size());
  ok = ok && dev_alloc_copy(&d_w, g.weights.data(), g.weights.size());
  ok = ok && dev_alloc_copy(&d_delays, g.delays.data(), g.delays.size());
  ok = ok && dev_alloc_copy(&d_channel, g.channel.data(), g.channel.size());
  ok = ok && dev_alloc_copy(&d_sign, sign.data(), sign.size());
  ok = ok && dev_alloc_copy(&d_role, role.data(), role.size());
  ok = ok && dev_alloc_copy(&d_inj_n, inj_neuron.data(), inj_neuron.size());
  ok = ok && dev_alloc_copy(&d_inj_a, inj_amp.data(), inj_amp.size());
  if (!ok) goto cleanup;

  ok = false;  // any goto below this point is a failure until proven otherwise
  if (cudaMalloc(&d_v, n * sizeof(float)) != cudaSuccess) goto cleanup;
  if (cudaMalloc(&d_ext, n * sizeof(float)) != cudaSuccess) goto cleanup;
  if (cudaMalloc(&d_refr, n * sizeof(int)) != cudaSuccess) goto cleanup;
  if (cudaMalloc(&d_ring, (size_t)ring_len * n * sizeof(float)) != cudaSuccess)
    goto cleanup;
  if (cudaMalloc(&d_chan, std::max(1, oc) * sizeof(int)) != cudaSuccess)
    goto cleanup;
  if (cudaMalloc(&d_spikes, sizeof(unsigned long long)) != cudaSuccess)
    goto cleanup;
  if (cudaMalloc(&d_syn, sizeof(unsigned long long)) != cudaSuccess)
    goto cleanup;

  cudaMemset(d_v, 0, n * sizeof(float));
  cudaMemset(d_ext, 0, n * sizeof(float));
  cudaMemset(d_refr, 0, n * sizeof(int));
  cudaMemset(d_ring, 0, (size_t)ring_len * n * sizeof(float));
  cudaMemset(d_chan, 0, std::max(1, oc) * sizeof(int));
  cudaMemset(d_spikes, 0, sizeof(unsigned long long));
  cudaMemset(d_syn, 0, sizeof(unsigned long long));

  {
    const int B = 256;
    const int grid_n = (n + B - 1) / B;
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < num_steps; ++t) {
      const int b = inj_row[t], en = inj_row[t + 1];
      if (en > b) {
        const int gi = (en - b + B - 1) / B;
        k_inject<<<gi, B>>>(d_inj_n, d_inj_a, b, en, d_ext);
      }
      k_step<<<grid_n, B>>>(t, n, ring_len, lif.threshold, lif.decay, lif.reset,
                            lif.refractory, d_row, d_post, d_w, d_delays, d_sign,
                            d_role, d_channel, d_v, d_ext, d_refr, d_ring, d_chan,
                            d_spikes, d_syn);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) goto cleanup;
    const auto t1 = std::chrono::steady_clock::now();
    out.stats.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  {
    std::vector<int> chan(std::max(1, oc), 0);
    unsigned long long spikes = 0, syn = 0;
    cudaMemcpy(chan.data(), d_chan, std::max(1, oc) * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&spikes, d_spikes, sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&syn, d_syn, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    out.logits.assign(oc, 0.0f);
    for (int c = 0; c < oc; ++c) out.logits[c] = static_cast<float>(chan[c]);
    out.stats.spikes = static_cast<long long>(spikes);
    out.stats.synaptic_events = static_cast<long long>(syn);
    out.used_backend = Backend::CudaAtomic;
  }

  ok = true;
cleanup:
  cudaFree(d_row); cudaFree(d_post); cudaFree(d_w); cudaFree(d_delays);
  cudaFree(d_channel); cudaFree(d_sign); cudaFree(d_role); cudaFree(d_inj_n);
  cudaFree(d_inj_a); cudaFree(d_v); cudaFree(d_ext);
  cudaFree(d_refr); cudaFree(d_ring); cudaFree(d_chan); cudaFree(d_spikes);
  cudaFree(d_syn);
  return ok;
}

}  // namespace cuda
}  // namespace snc
