// GPU batched e-prop trainer (see include/snc/cuda_trainer.hpp).

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "snc/cuda_trainer.hpp"

namespace snc {

struct CudaTrainSession {
  // dims
  int n = 0, S = 0, oc = 0, dim = 0, T = 0, ring_len = 0, batch = 0;
  // LIF / learning hyper-params
  float thr, decay, reset, gamma, lr_over_T, wmax, gain, max_rate;
  int refr_steps;
  bool poisson = true, train_hidden = true;
  // device graph (resident)
  int *d_row = nullptr, *d_post = nullptr, *d_delays = nullptr, *d_src = nullptr;
  int *d_role = nullptr, *d_channel = nullptr, *d_internal_row = nullptr;
  float *d_w = nullptr, *d_B = nullptr;
  // device per-batch state
  float *d_v = nullptr, *d_ext = nullptr, *d_tr = nullptr, *d_psi = nullptr,
        *d_ring = nullptr, *d_E = nullptr, *d_counts = nullptr, *d_X = nullptr,
        *d_delta = nullptr, *d_L = nullptr;
  int *d_refr = nullptr, *d_y = nullptr;
  unsigned char* d_fired = nullptr;
  curandState* d_rng = nullptr;
  unsigned long long *d_spikes = nullptr, *d_syn = nullptr;
};

namespace {
constexpr int kBlock = 256;
inline int grid(int total) { return (total + kBlock - 1) / kBlock; }

__global__ void k_rng_init(int total, unsigned long long seed, curandState* st) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) curand_init(seed, i, 0, &st[i]);
}

__global__ void k_inject_poisson(int n, int dim, int bs, float max_rate,
                                 float gain, const float* X, curandState* rng,
                                 float* ext) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= bs * dim) return;
  const int b = tid / dim, c = tid % dim;
  float p = X[b * dim + c] * max_rate;
  if (p > 1.0f) p = 1.0f;
  if (curand_uniform(&rng[tid]) < p) atomicAdd(&ext[b * n + c], gain);
}

__global__ void k_inject_direct(int n, int dim, int bs, float gain,
                                const float* X, float* ext) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= bs * dim) return;
  const int b = tid / dim, c = tid % dim;
  ext[b * n + c] += X[b * dim + c] * gain;
}

__global__ void k_integrate_fire(int t, int n, int bs, int ring_len, float thr,
                                 float decay, float reset, int refr_steps,
                                 float gamma, const int* role,
                                 const int* channel, int oc, float* v,
                                 float* ext, float* tr, float* psi, int* refr,
                                 unsigned char* fired, float* ring,
                                 float* counts, unsigned long long* spikes) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= bs * n) return;
  const int b = tid / n, j = tid % n;
  const int cur = b * ring_len * n + (t % ring_len) * n + j;
  float vpre;
  bool f = false;
  if (refr[tid] > 0) {
    --refr[tid];
    v[tid] = reset;
    vpre = reset;
  } else {
    float nv = v[tid] * decay + ext[tid] + ring[cur];
    vpre = nv;
    if (nv >= thr) { v[tid] = reset; refr[tid] = refr_steps; f = true; }
    else { v[tid] = nv; }
  }
  ext[tid] = 0.0f;
  ring[cur] = 0.0f;
  const float d = fabsf(vpre - thr) / (thr > 0.0f ? thr : 1.0f);
  psi[tid] = d < 1.0f ? gamma * (1.0f - d) : 0.0f;
  tr[tid] = decay * tr[tid] + (1.0f - decay) * (f ? 1.0f : 0.0f);
  fired[tid] = f ? 1 : 0;
  if (f) {
    atomicAdd(spikes, 1ULL);
    if (role[j] == 2 && channel[j] >= 0 && channel[j] < oc)
      atomicAdd(&counts[b * oc + channel[j]], 1.0f);
  }
}

__global__ void k_eligibility(int n, int S, int bs, const int* post,
                              const int* src, const float* psi, const float* tr,
                              float* E) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= bs * S) return;
  const int b = tid / S, s = tid % S;
  E[tid] += psi[b * n + post[s]] * tr[b * n + src[s]];
}

__global__ void k_deliver(int t, int n, int bs, int ring_len, const int* row,
                         const int* post, const int* delays, const float* w,
                         const unsigned char* fired, float* ring,
                         unsigned long long* syn) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= bs * n) return;
  const int b = tid / n, i = tid % n;
  if (!fired[tid]) return;
  const int rn = ring_len * n;
  const int beg = row[i], fin = row[i + 1];
  for (int e = beg; e < fin; ++e)
    atomicAdd(&ring[b * rn + ((t + delays[e]) % ring_len) * n + post[e]], w[e]);
  if (fin > beg) atomicAdd(syn, (unsigned long long)(fin - beg));
}

__global__ void k_softmax_delta(int oc, int bs, const float* counts,
                               const int* y, float* delta) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= bs) return;
  const float* c = &counts[b * oc];
  float mx = c[0];
  for (int k = 1; k < oc; ++k) mx = fmaxf(mx, c[k]);
  float sum = 0.0f;
  for (int k = 0; k < oc; ++k) sum += expf(c[k] - mx);
  for (int k = 0; k < oc; ++k)
    delta[b * oc + k] = expf(c[k] - mx) / sum - (k == y[b] ? 1.0f : 0.0f);
}

__global__ void k_learning_signal(int n, int oc, int bs, bool train_hidden,
                                  const int* role, const int* channel,
                                  const int* internal_row, const float* Bmat,
                                  const float* delta, float* L) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= bs * n) return;
  const int b = tid / n, j = tid % n;
  if (role[j] == 2 && channel[j] >= 0 && channel[j] < oc) {
    L[tid] = delta[b * oc + channel[j]];
  } else if (train_hidden && internal_row[j] >= 0) {
    const float* br = &Bmat[internal_row[j] * oc];
    float s = 0.0f;
    for (int k = 0; k < oc; ++k) s += br[k] * delta[b * oc + k];
    L[tid] = s;
  } else {
    L[tid] = 0.0f;
  }
}

__global__ void k_apply(int n, int S, int bs, float lr_over_T, float wmax,
                       const int* post, const float* L, const float* E,
                       float* w) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= S) return;
  float acc = 0.0f;
  for (int b = 0; b < bs; ++b) acc += L[b * n + post[s]] * E[b * S + s];
  float nw = w[s] - lr_over_T * acc / bs;
  w[s] = fminf(fmaxf(nw, -wmax), wmax);
}

template <typename T>
bool alloc_copy(T** d, const T* h, size_t cnt) {
  if (cudaMalloc(d, cnt * sizeof(T)) != cudaSuccess) return false;
  return cnt == 0 ||
         cudaMemcpy(*d, h, cnt * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}
}  // namespace

namespace cudatrain {

bool available(const TrainConfig& cfg) {
  if (cfg.encoder == Encoder::Latency) return false;  // GPU path: poisson/direct
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

CudaTrainSession* create(const SNNGraph& g, const TrainConfig& cfg, int batch) {
  if (!available(cfg)) return nullptr;
  auto* s = new CudaTrainSession();
  s->n = g.num_neurons;
  s->S = g.num_synapses();
  s->oc = std::max(1, g.num_output_channels);
  s->dim = std::max(1, g.num_input_channels);
  s->T = cfg.num_steps;
  s->batch = std::max(1, batch);
  s->thr = cfg.lif.threshold;
  s->decay = cfg.lif.decay;
  s->reset = cfg.lif.reset;
  s->refr_steps = cfg.lif.refractory;
  s->gamma = cfg.surrogate_scale;
  s->lr_over_T = cfg.lr / std::max(1, cfg.num_steps);
  s->wmax = cfg.w_max;
  s->gain = cfg.enc.gain;
  s->max_rate = cfg.enc.max_rate;
  s->poisson = cfg.encoder == Encoder::Poisson;
  s->train_hidden = cfg.train_hidden;

  int max_delay = 1;
  for (int i = 0; i < s->S; ++i) max_delay = std::max(max_delay, g.delays[i]);
  s->ring_len = max_delay + 1;

  // Host-side derived arrays.
  std::vector<int> src(s->S);
  for (int i = 0; i < s->n; ++i)
    for (int e = g.row_ptr[i]; e < g.row_ptr[i + 1]; ++e) src[e] = i;
  std::vector<int> role(s->n), channel(s->n), internal_row(s->n, -1);
  int n_internal = 0;
  for (int i = 0; i < s->n; ++i) {
    role[i] = static_cast<int>(g.role[i]);
    channel[i] = g.channel[i];
    if (g.role[i] == GraphRole::INTERNAL) internal_row[i] = n_internal++;
  }
  std::mt19937_64 rng(cfg.seed);
  std::normal_distribution<float> bn(0.0f, cfg.feedback_scale);
  std::vector<float> Bmat(static_cast<size_t>(n_internal) * s->oc);
  for (float& b : Bmat) b = bn(rng);
  std::uniform_real_distribution<float> wu(0.0f, cfg.w_init);
  std::vector<float> w(s->S);
  for (float& x : w) x = wu(rng);

  const int B = s->batch, N = s->n, SS = s->S, RN = s->ring_len * s->n;
  bool ok = true;
  ok = ok && alloc_copy(&s->d_row, g.row_ptr.data(), g.row_ptr.size());
  ok = ok && alloc_copy(&s->d_post, g.post_ids.data(), g.post_ids.size());
  ok = ok && alloc_copy(&s->d_delays, g.delays.data(), g.delays.size());
  ok = ok && alloc_copy(&s->d_src, src.data(), src.size());
  ok = ok && alloc_copy(&s->d_role, role.data(), role.size());
  ok = ok && alloc_copy(&s->d_channel, channel.data(), channel.size());
  ok = ok && alloc_copy(&s->d_internal_row, internal_row.data(), internal_row.size());
  ok = ok && alloc_copy(&s->d_B, Bmat.data(), Bmat.size());
  ok = ok && alloc_copy(&s->d_w, w.data(), w.size());
  auto am = [&](void** p, size_t bytes) {
    ok = ok && (cudaMalloc(p, bytes) == cudaSuccess);
  };
  am((void**)&s->d_v, (size_t)B * N * sizeof(float));
  am((void**)&s->d_ext, (size_t)B * N * sizeof(float));
  am((void**)&s->d_tr, (size_t)B * N * sizeof(float));
  am((void**)&s->d_psi, (size_t)B * N * sizeof(float));
  am((void**)&s->d_ring, (size_t)B * RN * sizeof(float));
  am((void**)&s->d_E, (size_t)B * SS * sizeof(float));
  am((void**)&s->d_counts, (size_t)B * s->oc * sizeof(float));
  am((void**)&s->d_X, (size_t)B * s->dim * sizeof(float));
  am((void**)&s->d_delta, (size_t)B * s->oc * sizeof(float));
  am((void**)&s->d_L, (size_t)B * N * sizeof(float));
  am((void**)&s->d_refr, (size_t)B * N * sizeof(int));
  am((void**)&s->d_y, (size_t)B * sizeof(int));
  am((void**)&s->d_fired, (size_t)B * N * sizeof(unsigned char));
  am((void**)&s->d_rng, (size_t)B * s->dim * sizeof(curandState));
  am((void**)&s->d_spikes, sizeof(unsigned long long));
  am((void**)&s->d_syn, sizeof(unsigned long long));
  if (!ok) { destroy(s); return nullptr; }

  k_rng_init<<<grid(B * s->dim), kBlock>>>(B * s->dim, cfg.seed, s->d_rng);
  if (cudaDeviceSynchronize() != cudaSuccess) { destroy(s); return nullptr; }
  return s;
}

void destroy(CudaTrainSession* s) {
  if (!s) return;
  for (void* p : {(void*)s->d_row, (void*)s->d_post, (void*)s->d_delays,
                  (void*)s->d_src, (void*)s->d_role, (void*)s->d_channel,
                  (void*)s->d_internal_row, (void*)s->d_w, (void*)s->d_B,
                  (void*)s->d_v, (void*)s->d_ext, (void*)s->d_tr, (void*)s->d_psi,
                  (void*)s->d_ring, (void*)s->d_E, (void*)s->d_counts,
                  (void*)s->d_X, (void*)s->d_delta, (void*)s->d_L,
                  (void*)s->d_refr, (void*)s->d_y, (void*)s->d_fired,
                  (void*)s->d_rng, (void*)s->d_spikes, (void*)s->d_syn})
    cudaFree(p);
  delete s;
}

void set_weights(CudaTrainSession* s, const std::vector<float>& w) {
  if (s && static_cast<int>(w.size()) == s->S)
    cudaMemcpy(s->d_w, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice);
}

std::vector<float> get_weights(const CudaTrainSession* s) {
  std::vector<float> w(s ? s->S : 0);
  if (s) cudaMemcpy(w.data(), s->d_w, w.size() * sizeof(float), cudaMemcpyDeviceToHost);
  return w;
}

namespace {
// Run one minibatch forward (and, if training, eligibility + update). Fills the
// host `counts` buffer [bs*oc]. Assumes d_X/d_y already uploaded for this batch.
void run_batch(CudaTrainSession* s, int bs, bool train, std::vector<float>& counts) {
  const int N = s->n, SS = s->S, RN = s->ring_len * s->n;
  cudaMemset(s->d_v, 0, (size_t)bs * N * sizeof(float));
  cudaMemset(s->d_ext, 0, (size_t)bs * N * sizeof(float));
  cudaMemset(s->d_tr, 0, (size_t)bs * N * sizeof(float));
  cudaMemset(s->d_refr, 0, (size_t)bs * N * sizeof(int));
  cudaMemset(s->d_ring, 0, (size_t)bs * RN * sizeof(float));
  cudaMemset(s->d_counts, 0, (size_t)bs * s->oc * sizeof(float));
  if (train) cudaMemset(s->d_E, 0, (size_t)bs * SS * sizeof(float));

  for (int t = 0; t < s->T; ++t) {
    if (s->poisson)
      k_inject_poisson<<<grid(bs * s->dim), kBlock>>>(N, s->dim, bs, s->max_rate,
                                                      s->gain, s->d_X, s->d_rng, s->d_ext);
    else
      k_inject_direct<<<grid(bs * s->dim), kBlock>>>(N, s->dim, bs, s->gain, s->d_X, s->d_ext);

    k_integrate_fire<<<grid(bs * N), kBlock>>>(
        t, N, bs, s->ring_len, s->thr, s->decay, s->reset, s->refr_steps, s->gamma,
        s->d_role, s->d_channel, s->oc, s->d_v, s->d_ext, s->d_tr, s->d_psi,
        s->d_refr, s->d_fired, s->d_ring, s->d_counts, s->d_spikes);

    if (train)
      k_eligibility<<<grid(bs * SS), kBlock>>>(N, SS, bs, s->d_post, s->d_src,
                                               s->d_psi, s->d_tr, s->d_E);

    k_deliver<<<grid(bs * N), kBlock>>>(t, N, bs, s->ring_len, s->d_row, s->d_post,
                                        s->d_delays, s->d_w, s->d_fired, s->d_ring,
                                        s->d_syn);
  }

  if (train) {
    k_softmax_delta<<<grid(bs), kBlock>>>(s->oc, bs, s->d_counts, s->d_y, s->d_delta);
    k_learning_signal<<<grid(bs * N), kBlock>>>(N, s->oc, bs, s->train_hidden,
                                                s->d_role, s->d_channel,
                                                s->d_internal_row, s->d_B,
                                                s->d_delta, s->d_L);
    k_apply<<<grid(SS), kBlock>>>(N, SS, bs, s->lr_over_T, s->wmax, s->d_post,
                                  s->d_L, s->d_E, s->d_w);
  }

  counts.resize((size_t)bs * s->oc);
  cudaMemcpy(counts.data(), s->d_counts, counts.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
}

// Upload one batch of features/labels from `d` at indices order[off..off+bs).
void upload_batch(CudaTrainSession* s, const Dataset& d,
                  const std::vector<int>& order, int off, int bs,
                  std::vector<float>& hX, std::vector<int>& hy) {
  hX.resize((size_t)bs * s->dim);
  hy.resize(bs);
  for (int b = 0; b < bs; ++b) {
    const int idx = order[off + b];
    std::copy(d.x[idx].begin(), d.x[idx].end(), hX.begin() + (size_t)b * s->dim);
    hy[b] = d.y[idx];
  }
  cudaMemcpy(s->d_X, hX.data(), hX.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(s->d_y, hy.data(), hy.size() * sizeof(int), cudaMemcpyHostToDevice);
}

int argmax(const float* c, int oc) {
  int best = 0;
  for (int k = 1; k < oc; ++k) if (c[k] > c[best]) best = k;
  return best;
}
}  // namespace

double evaluate(CudaTrainSession* s, const Dataset& d) {
  if (!s || d.size() == 0) return 0.0;
  std::vector<int> order(d.size());
  std::iota(order.begin(), order.end(), 0);
  std::vector<float> hX, counts;
  std::vector<int> hy;
  int correct = 0;
  for (int off = 0; off < d.size(); off += s->batch) {
    const int bs = std::min(s->batch, d.size() - off);
    upload_batch(s, d, order, off, bs, hX, hy);
    run_batch(s, bs, /*train=*/false, counts);
    for (int b = 0; b < bs; ++b)
      if (argmax(&counts[b * s->oc], s->oc) == hy[b]) ++correct;
  }
  cudaDeviceSynchronize();
  return static_cast<double>(correct) / d.size();
}

EpochStats train_epoch(CudaTrainSession* s, const Dataset& train,
                       const Dataset& test, int epoch) {
  EpochStats st;
  if (!s || train.size() == 0) return st;
  std::vector<int> order(train.size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937_64 rng(0x9e3779b97f4a7c15ULL * (epoch + 1));
  std::shuffle(order.begin(), order.end(), rng);

  cudaMemset(s->d_spikes, 0, sizeof(unsigned long long));
  cudaMemset(s->d_syn, 0, sizeof(unsigned long long));

  std::vector<float> hX, counts;
  std::vector<int> hy;
  double loss_sum = 0.0;
  int correct = 0;
  for (int off = 0; off < train.size(); off += s->batch) {
    const int bs = std::min(s->batch, train.size() - off);
    upload_batch(s, train, order, off, bs, hX, hy);
    run_batch(s, bs, /*train=*/true, counts);
    for (int b = 0; b < bs; ++b) {
      const float* c = &counts[b * s->oc];
      float mx = c[0];
      for (int k = 1; k < s->oc; ++k) mx = std::max(mx, c[k]);
      double sum = 0.0;
      for (int k = 0; k < s->oc; ++k) sum += std::exp(c[k] - mx);
      loss_sum += -(c[hy[b]] - mx) + std::log(sum);
      if (argmax(c, s->oc) == hy[b]) ++correct;
    }
  }
  cudaDeviceSynchronize();

  unsigned long long spikes = 0, syn = 0;
  cudaMemcpy(&spikes, s->d_spikes, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(&syn, s->d_syn, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

  st.loss = loss_sum / train.size();
  st.train_acc = static_cast<double>(correct) / train.size();
  st.test_acc = evaluate(s, test);
  st.spikes = static_cast<long long>(spikes);
  st.synaptic_events = static_cast<long long>(syn);
  st.energy = 1.0 * st.spikes + 0.2 * st.synaptic_events;
  return st;
}

}  // namespace cudatrain
}  // namespace snc
