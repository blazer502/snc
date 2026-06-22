// No-op CUDA backend used when the project is built without CUDA support
// (SNC_ENABLE_CUDA=OFF). Keeps the cuda::* symbols defined so the runtime can
// link and transparently fall back to the CPU path.

#include <vector>

#include "snc/runtime.hpp"

namespace snc {
struct InputEvent;
namespace cuda {

bool available() { return false; }

bool forward(const SNNGraph&, const std::vector<InputEvent>&, int,
             const LIFParams&, Backend, ForwardResult&) {
  return false;
}

}  // namespace cuda
}  // namespace snc
