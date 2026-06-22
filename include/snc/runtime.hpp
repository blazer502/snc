// Runtime layer: execute a compiled SNNGraph (new-plan.md sections 2 & 4).
//
// A backend takes a read-only CSR graph plus one sample's InputEvent stream and
// runs `num_steps` of leaky-integrate-and-fire dynamics with delayed,
// sign-aware synaptic delivery, then decodes output-neuron spike counts into
// per-class logits. The CPU and OpenMP backends are implemented here; the
// cuda-* variants dispatch to the CUDA backend and fall back to CPU when CUDA
// is unavailable (so the bench always runs).

#pragma once

#include <string>
#include <vector>

#include "snc/snn_graph.hpp"

namespace snc {

struct InputEvent;  // from snc/encoders.hpp

enum class Backend : uint8_t {
  Cpu = 0,
  OpenMP = 1,
  CudaAtomic = 2,
  CudaBucket = 3,
  CudaSort = 4,
};

bool parse_backend(const std::string& name, Backend& out);
const char* backend_name(Backend b);
bool backend_is_cuda(Backend b);

// Leaky integrate-and-fire parameters shared by every backend.
struct LIFParams {
  float threshold = 1.0f;  // fire when membrane >= threshold
  float decay = 0.9f;      // membrane leak per step (v *= decay)
  float reset = 0.0f;      // membrane value after a spike
  int refractory = 1;      // steps a neuron ignores input after firing
};

struct RunStats {
  long long spikes = 0;           // total neuron firings
  long long synaptic_events = 0;  // total spike deliveries expanded
  int overflow = 0;               // event-buffer overflow count (CUDA paths)
  double ms = 0.0;                // wall-clock of the simulation loop
};

struct ForwardResult {
  std::vector<float> logits;  // size g.num_output_channels (summed spike counts)
  RunStats stats;
  Backend used_backend = Backend::Cpu;  // actual backend after any fallback
};

// Forward inference for ONE sample's event stream. `backend` selects the path;
// a cuda-* backend that cannot run (not built / no device) transparently falls
// back to Cpu and reports that in ForwardResult::used_backend.
ForwardResult forward(const SNNGraph& g, const std::vector<InputEvent>& events,
                      int num_steps, const LIFParams& lif, Backend backend);

// ---- CUDA backend hook (defined in src/runtime/cuda_backend.*) -------------
namespace cuda {
// True only when compiled with CUDA support AND a usable device is present.
bool available();
// Run on the GPU. Returns false (leaving `out` untouched) when unavailable so
// the caller can fall back. `variant` is one of the CudaAtomic/Bucket/Sort.
bool forward(const SNNGraph& g, const std::vector<InputEvent>& events,
             int num_steps, const LIFParams& lif, Backend variant,
             ForwardResult& out);
}  // namespace cuda

}  // namespace snc
