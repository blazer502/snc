// GPU-accelerated e-prop training (new-plan.md Phase 5 / "CUDA training").
//
// The CPU Trainer processes one sample at a time, which under-uses a GPU on the
// small graphs here. This path instead runs a whole MINIBATCH of samples in
// parallel on the device -- each sample is independent under a frozen topology,
// so state/eligibility are sized [batch x N] / [batch x S] and the kernels fan
// out over batch*N or batch*S threads. Weights stay resident on the GPU across
// the epoch; this is minibatch SGD (not bit-identical to the per-sample CPU
// path), validated to reach comparable accuracy much faster.
//
// Encoding is Poisson (cuRAND) or direct on the device; latency encoding is not
// supported here -- callers should fall back to the CPU Trainer for it.

#pragma once

#include <vector>

#include "snc/dataset.hpp"
#include "snc/snn_graph.hpp"
#include "snc/trainer.hpp"  // TrainConfig, EpochStats

namespace snc {

struct CudaTrainSession;  // opaque; defined in cuda_trainer.cu

namespace cudatrain {

// True when built with CUDA, a device is present, AND cfg.encoder is supported
// on the GPU (Poisson or Direct).
bool available(const TrainConfig& cfg);

// Allocate a device session for a frozen graph; weights are randomly initialised
// (uniform [0, cfg.w_init]). Returns nullptr if unavailable. `batch` is the
// minibatch size.
CudaTrainSession* create(const SNNGraph& g, const TrainConfig& cfg, int batch);
void destroy(CudaTrainSession*);

// Carry weights in/out across structural epochs (signed, CSR order).
void set_weights(CudaTrainSession*, const std::vector<float>& w);
std::vector<float> get_weights(const CudaTrainSession*);

// One shuffled pass over `train` (minibatch SGD), then evaluate `test`.
EpochStats train_epoch(CudaTrainSession*, const Dataset& train,
                       const Dataset& test, int epoch);

// Argmax accuracy over a dataset using current weights (no learning).
double evaluate(CudaTrainSession*, const Dataset& test);

// Activity statistics for the structural update (two-timescale co-training).
// Accumulated over training batches only (not evaluation); call reset_stats()
// before the inner loop and collect_stats() after. Indexed by current-graph
// synapse / neuron id.
void reset_stats(CudaTrainSession*);
void collect_stats(const CudaTrainSession*, std::vector<long long>& syn_deliveries,
                   std::vector<long long>& neuron_fires);

}  // namespace cudatrain
}  // namespace snc
