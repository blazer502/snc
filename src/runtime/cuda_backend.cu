// CUDA backend -- event-delivery strategies (new-plan.md section 4.2).
//
//   cuda-atomic : Stage 1. Fused integrate/fire/expand; every fired neuron
//                 atomicAdds its weighted spikes into a delay ring buffer.
//                 Simple, correct, the reference the others are checked against.
//   cuda-bucket : Stage 2. Delivery is sharded across B copies of the ring by
//                 source neuron, so atomic contention on any one address drops
//                 ~B-fold; the shards are reduced into the ring each step.
//   cuda-sort   : Stage 3. Fired neurons' synapses are expanded into (ring-index
//                 key, signed value) events, sorted by key, segment-reduced
//                 (Thrust), and scattered into the ring -- fully atomic-free.
//
// All three share one integrate/fire step and are validated to produce the same
// spike/event counts as the CPU backend.

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <vector>

#include "snc/encoders.hpp"
#include "snc/runtime.hpp"

namespace snc {
namespace cuda {
namespace {

constexpr int kBlock = 256;
constexpr int kShards = 8;  // power of two -> mask with kShards-1

__global__ void k_inject(const int* inj_neuron, const float* inj_amp, int begin,
                         int end, float* ext) {
  int j = begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= end) return;
  atomicAdd(&ext[inj_neuron[j]], inj_amp[j]);
}

// Fused integrate / fire / atomic-expand (cuda-atomic). One thread per neuron.
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
    if (nv >= threshold) { v[i] = reset; refr[i] = refractory; fired = true; }
    else { v[i] = nv; }
  }
  ext[i] = 0.0f;
  ring[cur] = 0.0f;
  if (!fired) return;
  atomicAdd(spikes, 1ULL);
  if (role[i] == 2 && channel[i] >= 0) atomicAdd(&chan_count[channel[i]], 1);
  const float s = static_cast<float>(sign[i]);
  const int beg = row_ptr[i], fin = row_ptr[i + 1];
  for (int e = beg; e < fin; ++e)
    atomicAdd(&ring[((t + delays[e]) % ring_len) * n + post_ids[e]],
              s * weights[e]);
  if (fin > beg) atomicAdd(syn_events, (unsigned long long)(fin - beg));
}

// Integrate / fire only (cuda-bucket, cuda-sort): records fired[], no delivery.
__global__ void k_integrate_fire(int t, int n, int ring_len, float threshold,
                                 float decay, float reset, int refractory,
                                 const unsigned char* role, const int* channel,
                                 float* v, float* ext, int* refr, float* ring,
                                 int* chan_count, unsigned long long* spikes,
                                 unsigned char* fired) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int cur = (t % ring_len) * n + i;
  bool f = false;
  if (refr[i] > 0) {
    --refr[i];
    v[i] = reset;
  } else {
    float nv = v[i] * decay + ext[i] + ring[cur];
    if (nv >= threshold) { v[i] = reset; refr[i] = refractory; f = true; }
    else { v[i] = nv; }
  }
  ext[i] = 0.0f;
  ring[cur] = 0.0f;
  fired[i] = f ? 1 : 0;
  if (f) {
    atomicAdd(spikes, 1ULL);
    if (role[i] == 2 && channel[i] >= 0) atomicAdd(&chan_count[channel[i]], 1);
  }
}

// Deliver into source-sharded ring copies to spread atomic contention.
__global__ void k_deliver_bucket(int t, int n, int ring_len, int rn,
                                 const int* row_ptr, const int* post_ids,
                                 const float* weights, const int* delays,
                                 const signed char* sign,
                                 const unsigned char* fired, float* shards,
                                 unsigned long long* syn_events) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || !fired[i]) return;
  const int sh = (i & (kShards - 1)) * rn;
  const float s = static_cast<float>(sign[i]);
  const int beg = row_ptr[i], fin = row_ptr[i + 1];
  for (int e = beg; e < fin; ++e)
    atomicAdd(&shards[sh + ((t + delays[e]) % ring_len) * n + post_ids[e]],
              s * weights[e]);
  if (fin > beg) atomicAdd(syn_events, (unsigned long long)(fin - beg));
}

// Sum the shard copies into the ring and clear them (one thread per ring slot).
__global__ void k_reduce_shards(int rn, float* ring, float* shards) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rn) return;
  float acc = 0.0f;
  for (int s = 0; s < kShards; ++s) {
    acc += shards[s * rn + idx];
    shards[s * rn + idx] = 0.0f;
  }
  ring[idx] += acc;
}

// Expand every synapse into an event keyed by its target ring slot. Synapses of
// non-fired neurons get a sentinel key (sorted to the end, scattered nowhere).
__global__ void k_build_events(int t, int n, int ring_len, int sentinel,
                               const int* row_ptr, const int* post_ids,
                               const float* weights, const int* delays,
                               const signed char* sign,
                               const unsigned char* fired, int* keys,
                               float* vals, unsigned long long* syn_events) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int beg = row_ptr[i], fin = row_ptr[i + 1];
  const bool f = fired[i];
  const float s = static_cast<float>(sign[i]);
  for (int e = beg; e < fin; ++e) {
    if (f) {
      keys[e] = ((t + delays[e]) % ring_len) * n + post_ids[e];
      vals[e] = s * weights[e];
    } else {
      keys[e] = sentinel;
      vals[e] = 0.0f;
    }
  }
  if (f && fin > beg) atomicAdd(syn_events, (unsigned long long)(fin - beg));
}

// Add each segment-reduced (unique key, summed value) into the ring. Keys are
// unique after reduce_by_key, so no atomics are needed.
__global__ void k_scatter(int u, int rn, const int* okeys, const float* ovals,
                          float* ring) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= u) return;
  const int k = okeys[j];
  if (k < rn) ring[k] += ovals[j];
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
             int num_steps, const LIFParams& lif, Backend variant,
             ForwardResult& out) {
  const int n = g.num_neurons;
  const int oc = g.num_output_channels;
  const int S = g.num_synapses();
  if (n == 0 || num_steps <= 0) return false;
  const bool use_bucket = variant == Backend::CudaBucket;
  const bool use_sort = variant == Backend::CudaSort;

  int max_delay = 1;
  for (int s = 0; s < S; ++s)
    if (g.delays[s] > max_delay) max_delay = g.delays[s];
  const int ring_len = max_delay + 1;
  const int rn = ring_len * n;  // ring slots (also the sort sentinel key)

  // Flatten the injection schedule (channel -> input neurons) into per-step
  // CSR ranges so the device applies each step's external drive in one pass.
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

  std::vector<signed char> sign(n);
  std::vector<unsigned char> role(n);
  for (int i = 0; i < n; ++i) {
    sign[i] = static_cast<signed char>(g.sign[i]);
    role[i] = static_cast<unsigned char>(g.role[i]);
  }

  // Device buffers. All pointers declared up front so the goto-cleanup path
  // never jumps over an initializer.
  int *d_row = nullptr, *d_post = nullptr, *d_delays = nullptr,
      *d_channel = nullptr, *d_refr = nullptr, *d_chan = nullptr,
      *d_inj_n = nullptr, *d_keys = nullptr, *d_okeys = nullptr;
  float *d_w = nullptr, *d_v = nullptr, *d_ext = nullptr, *d_ring = nullptr,
        *d_inj_a = nullptr, *d_shards = nullptr, *d_vals = nullptr,
        *d_ovals = nullptr;
  signed char* d_sign = nullptr;
  unsigned char* d_role = nullptr;
  unsigned char* d_fired = nullptr;
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
  if (cudaMalloc(&d_ring, (size_t)rn * sizeof(float)) != cudaSuccess) goto cleanup;
  if (cudaMalloc(&d_chan, std::max(1, oc) * sizeof(int)) != cudaSuccess) goto cleanup;
  if (cudaMalloc(&d_spikes, sizeof(unsigned long long)) != cudaSuccess) goto cleanup;
  if (cudaMalloc(&d_syn, sizeof(unsigned long long)) != cudaSuccess) goto cleanup;
  if (use_bucket || use_sort)
    if (cudaMalloc(&d_fired, n * sizeof(unsigned char)) != cudaSuccess) goto cleanup;
  if (use_bucket)
    if (cudaMalloc(&d_shards, (size_t)kShards * rn * sizeof(float)) != cudaSuccess) goto cleanup;
  if (use_sort) {
    if (cudaMalloc(&d_keys, (size_t)std::max(1, S) * sizeof(int)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_vals, (size_t)std::max(1, S) * sizeof(float)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_okeys, (size_t)std::max(1, S) * sizeof(int)) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&d_ovals, (size_t)std::max(1, S) * sizeof(float)) != cudaSuccess) goto cleanup;
  }

  cudaMemset(d_v, 0, n * sizeof(float));
  cudaMemset(d_ext, 0, n * sizeof(float));
  cudaMemset(d_refr, 0, n * sizeof(int));
  cudaMemset(d_ring, 0, (size_t)rn * sizeof(float));
  cudaMemset(d_chan, 0, std::max(1, oc) * sizeof(int));
  cudaMemset(d_spikes, 0, sizeof(unsigned long long));
  cudaMemset(d_syn, 0, sizeof(unsigned long long));
  if (d_shards) cudaMemset(d_shards, 0, (size_t)kShards * rn * sizeof(float));

  {
    const int grid_n = (n + kBlock - 1) / kBlock;
    const int grid_rn = (rn + kBlock - 1) / kBlock;
    bool loop_ok = true;
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    try {
      for (int t = 0; t < num_steps; ++t) {
        const int b = inj_row[t], en = inj_row[t + 1];
        if (en > b)
          k_inject<<<(en - b + kBlock - 1) / kBlock, kBlock>>>(d_inj_n, d_inj_a, b, en, d_ext);

        if (!use_bucket && !use_sort) {
          k_step<<<grid_n, kBlock>>>(t, n, ring_len, lif.threshold, lif.decay,
                                     lif.reset, lif.refractory, d_row, d_post, d_w,
                                     d_delays, d_sign, d_role, d_channel, d_v,
                                     d_ext, d_refr, d_ring, d_chan, d_spikes, d_syn);
          continue;
        }

        k_integrate_fire<<<grid_n, kBlock>>>(t, n, ring_len, lif.threshold,
                                             lif.decay, lif.reset, lif.refractory,
                                             d_role, d_channel, d_v, d_ext, d_refr,
                                             d_ring, d_chan, d_spikes, d_fired);
        if (use_bucket) {
          k_deliver_bucket<<<grid_n, kBlock>>>(t, n, ring_len, rn, d_row, d_post,
                                               d_w, d_delays, d_sign, d_fired,
                                               d_shards, d_syn);
          k_reduce_shards<<<grid_rn, kBlock>>>(rn, d_ring, d_shards);
        } else {  // use_sort
          k_build_events<<<grid_n, kBlock>>>(t, n, ring_len, /*sentinel=*/rn,
                                             d_row, d_post, d_w, d_delays, d_sign,
                                             d_fired, d_keys, d_vals, d_syn);
          thrust::device_ptr<int> keys(d_keys), okeys(d_okeys);
          thrust::device_ptr<float> vals(d_vals), ovals(d_ovals);
          thrust::sort_by_key(thrust::device, keys, keys + S, vals);
          auto end = thrust::reduce_by_key(thrust::device, keys, keys + S, vals,
                                           okeys, ovals);
          const int u = static_cast<int>(end.first - okeys);
          if (u > 0)
            k_scatter<<<(u + kBlock - 1) / kBlock, kBlock>>>(u, rn, d_okeys,
                                                             d_ovals, d_ring);
        }
      }
    } catch (...) {
      loop_ok = false;
    }
    if (!loop_ok || cudaDeviceSynchronize() != cudaSuccess) goto cleanup;
    const auto t1 = std::chrono::steady_clock::now();
    out.stats.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  {
    std::vector<int> chan(std::max(1, oc), 0);
    unsigned long long spikes = 0, syn = 0;
    cudaMemcpy(chan.data(), d_chan, std::max(1, oc) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&spikes, d_spikes, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&syn, d_syn, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    out.logits.assign(oc, 0.0f);
    for (int c = 0; c < oc; ++c) out.logits[c] = static_cast<float>(chan[c]);
    out.stats.spikes = static_cast<long long>(spikes);
    out.stats.synaptic_events = static_cast<long long>(syn);
    out.used_backend = variant;
  }

  ok = true;
cleanup:
  cudaFree(d_row); cudaFree(d_post); cudaFree(d_w); cudaFree(d_delays);
  cudaFree(d_channel); cudaFree(d_sign); cudaFree(d_role); cudaFree(d_inj_n);
  cudaFree(d_inj_a); cudaFree(d_v); cudaFree(d_ext); cudaFree(d_refr);
  cudaFree(d_ring); cudaFree(d_chan); cudaFree(d_spikes); cudaFree(d_syn);
  cudaFree(d_fired); cudaFree(d_shards); cudaFree(d_keys); cudaFree(d_vals);
  cudaFree(d_okeys); cudaFree(d_ovals);
  return ok;
}

}  // namespace cuda
}  // namespace snc
