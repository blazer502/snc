#include "brain_grid.hpp"

namespace snc {

BrainGrid::BrainGrid(int X, int Y, int Z) : X_(X), Y_(Y), Z_(Z) {
  assert(X > 0 && Y > 0 && Z > 0);
  // Note: a future per-voxel parallel CA pass that splits work by z-slice
  // would additionally require `(X * Y) % 32 == 0` so that no two threads
  // touch bits inside the same packed word. The current simulator only
  // parallelises across neurons (not voxels), so any positive (X, Y, Z) is
  // safe and growing the volume mid-simulation is allowed.

  const std::size_t cells = static_cast<std::size_t>(X) * Y * Z;
  const std::size_t words = (cells + 31) / 32;
  front_.assign(words, 0);
  back_.assign(words, 0);
}

}  // namespace snc
