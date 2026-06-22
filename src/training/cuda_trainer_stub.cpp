// No-op GPU trainer for builds without CUDA (SNC_ENABLE_CUDA=OFF). Keeps the
// cudatrain::* symbols defined so callers link; available() is always false, so
// the real entry points are never reached.

#include "snc/cuda_trainer.hpp"

namespace snc {
namespace cudatrain {

bool available(const TrainConfig&) { return false; }
CudaTrainSession* create(const SNNGraph&, const TrainConfig&, int) { return nullptr; }
void destroy(CudaTrainSession*) {}
void set_weights(CudaTrainSession*, const std::vector<float>&) {}
std::vector<float> get_weights(const CudaTrainSession*) { return {}; }
EpochStats train_epoch(CudaTrainSession*, const Dataset&, const Dataset&, int) {
  return {};
}
double evaluate(CudaTrainSession*, const Dataset&) { return 0.0; }

}  // namespace cudatrain
}  // namespace snc
