// Coarse-grained energy budget over the 3D volume.
//
// The brain volume is partitioned into cubic regions of `region_size` voxels.
// Each region holds a single floating-point energy budget. Activity (firing,
// sprouting, synapse formation, synapse use) depletes the local region's
// budget; energy regenerates by a fixed amount per simulation step, capped at
// `max_energy`. Structural growth is gated on the local budget, providing a
// global "metabolic" throttle: hot, frequently-firing regions cannot also
// keep growing aggressively, mirroring metabolic competition in biology.

#pragma once

#include <cstddef>
#include <vector>

namespace snc {

class EnergyField {
 public:
  EnergyField(int X, int Y, int Z, int region_size, float max_energy);

  int region_size() const noexcept { return R_; }
  int rX() const noexcept { return rX_; }
  int rY() const noexcept { return rY_; }
  int rZ() const noexcept { return rZ_; }
  float max_energy() const noexcept { return max_; }

  float& at(int rx, int ry, int rz) noexcept {
    return energy_[index(rx, ry, rz)];
  }
  float at(int rx, int ry, int rz) const noexcept {
    return energy_[index(rx, ry, rz)];
  }

  // World-coordinate accessors -- they map a voxel coord to its region cell.
  float& energy_at(int x, int y, int z) noexcept {
    return at(x / R_, y / R_, z / R_);
  }
  float energy_at(int x, int y, int z) const noexcept {
    return at(x / R_, y / R_, z / R_);
  }

  // Add `amount` to every region, clamped to `max_energy`.
  void regenerate(float amount);

  // Raw access to the per-region float array; used by sleep (save/load).
  const std::vector<float>& raw_data() const noexcept { return energy_; }
  std::vector<float>& raw_data() noexcept { return energy_; }

 private:
  std::size_t index(int rx, int ry, int rz) const noexcept {
    return static_cast<std::size_t>(rx) +
           static_cast<std::size_t>(ry) * rX_ +
           static_cast<std::size_t>(rz) * rX_ * rY_;
  }

  int R_;
  int rX_, rY_, rZ_;
  float max_;
  std::vector<float> energy_;
};

}  // namespace snc
