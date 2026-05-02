#include "energy_field.hpp"

#include <algorithm>

namespace snc {

EnergyField::EnergyField(int X, int Y, int Z, int region_size, float max_energy)
    : R_(region_size),
      rX_((X + region_size - 1) / region_size),
      rY_((Y + region_size - 1) / region_size),
      rZ_((Z + region_size - 1) / region_size),
      max_(max_energy),
      energy_(static_cast<std::size_t>(rX_) * rY_ * rZ_, max_energy) {}

void EnergyField::regenerate(float amount) {
  for (auto& e : energy_) {
    e = std::min(max_, e + amount);
  }
}

}  // namespace snc
