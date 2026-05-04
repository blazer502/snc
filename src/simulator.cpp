#include "simulator.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>

namespace snc {

namespace {

// 6-connected neighbourhood. Diagonal sprouting is intentionally omitted;
// adding it is a one-line change but inflates volume exclusion counts.
constexpr int kNeighbours[6][3] = {
    { 1,  0,  0}, {-1,  0,  0},
    { 0,  1,  0}, { 0, -1,  0},
    { 0,  0,  1}, { 0,  0, -1},
};

inline float clamp01(float v) {
  return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

}  // namespace

Simulator::Simulator(SimConfig cfg)
    : cfg_(cfg),
      grid_(cfg.X, cfg.Y, cfg.Z),
      energy_(cfg.X, cfg.Y, cfg.Z, cfg.region_size, cfg.energy_max),
      owner_(static_cast<std::size_t>(cfg.X) * cfg.Y * cfg.Z, 0u),
      voxel_role_(static_cast<std::size_t>(cfg.X) * cfg.Y * cfg.Z,
                  ROLE_DENDRITE),
      astrocyte_ca_(static_cast<std::size_t>(energy_.rX()) *
                        energy_.rY() * energy_.rZ(),
                    0.0f),
      delivery_ring_(kDeliveryRingSize),
      rng_(cfg.seed) {}

bool Simulator::try_set_neuron(int x, int y, int z, uint32_t owner_id) {
  if (!grid_.in_bounds(x, y, z)) return false;
  if (grid_.get(x, y, z) != BrainGrid::EMPTY) return false;
  grid_.set(x, y, z, BrainGrid::NEURON);
  owner_[lin(x, y, z)] = owner_id;
  return true;
}

void Simulator::seed_neurons(int n) {
  std::uniform_int_distribution<int> dx(2, cfg_.X - 3);
  std::uniform_int_distribution<int> dy(2, cfg_.Y - 3);
  std::uniform_int_distribution<int> dz(2, cfg_.Z - 3);

  neurons_.reserve(neurons_.size() + n);

  for (int i = 0; i < n; ++i) {
    Neuron neu;
    neu.id = static_cast<uint32_t>(neurons_.size() + 1);
    for (int attempt = 0; attempt < 32; ++attempt) {
      const int x = dx(rng_), y = dy(rng_), z = dz(rng_);
      if (try_set_neuron(x, y, z, neu.id)) {
        neu.soma = {static_cast<int16_t>(x),
                    static_cast<int16_t>(y),
                    static_cast<int16_t>(z)};
        neu.body.push_back(neu.soma);
        neurons_.push_back(std::move(neu));
        apply_position_prior(neurons_.back());
        break;
      }
    }
  }
}

int Simulator::birth_neurons(int n, const BoundingBox* area) {
  // Place `n` neurons inside the currently-tracked VZ band. When an `area` is
  // given the (x, y) range is intersected with the area's bounding box so
  // different cortical regions can run neurogenesis on different schedules.
  if (vz_hi_ < vz_lo_) return 0;
  const int x_lo = area ? std::max(1, area->x_lo) : 1;
  const int x_hi = area ? std::min(cfg_.X - 2, area->x_hi) : cfg_.X - 2;
  const int y_lo = area ? std::max(1, area->y_lo) : 1;
  const int y_hi = area ? std::min(cfg_.Y - 2, area->y_hi) : cfg_.Y - 2;
  if (x_lo > x_hi || y_lo > y_hi) return 0;
  std::uniform_int_distribution<int> dx(x_lo, x_hi);
  std::uniform_int_distribution<int> dy(y_lo, y_hi);
  std::uniform_int_distribution<int> dz(vz_lo_, vz_hi_);
  int placed = 0;
  for (int i = 0; i < n; ++i) {
    Neuron neu;
    neu.id = static_cast<uint32_t>(neurons_.size() + 1);
    for (int attempt = 0; attempt < 32; ++attempt) {
      const int x = dx(rng_), y = dy(rng_), z = dz(rng_);
      if (try_set_neuron(x, y, z, neu.id)) {
        neu.soma = {static_cast<int16_t>(x),
                    static_cast<int16_t>(y),
                    static_cast<int16_t>(z)};
        neu.body.push_back(neu.soma);
        neurons_.push_back(std::move(neu));
        apply_position_prior(neurons_.back());
        ++placed;
        break;
      }
    }
  }
  return placed;
}

void Simulator::grow_volume(int dx, int dy, int dz) {
  // The simulated volume gets bigger over developmental time. We allocate a
  // larger grid + energy field + owner map, copy old data with an offset so
  // the existing tissue ends up centered, and shift every neuron coordinate.
  // New space is EMPTY voxels at full energy -- room for future sprouting.
  const int R = cfg_.region_size;
  assert(dx >= 0 && dy >= 0 && dz >= 0);
  assert(dx % R == 0 && dy % R == 0 && dz % R == 0 &&
         "growth amounts must be multiples of region_size");
  if (dx == 0 && dy == 0 && dz == 0) return;

  const int newX = cfg_.X + 2 * dx;
  const int newY = cfg_.Y + 2 * dy;
  const int newZ = cfg_.Z + 2 * dz;

  BrainGrid new_grid(newX, newY, newZ);
  for (int z = 0; z < cfg_.Z; ++z) {
    for (int y = 0; y < cfg_.Y; ++y) {
      for (int x = 0; x < cfg_.X; ++x) {
        const auto c = grid_.get(x, y, z);
        if (c != BrainGrid::EMPTY) {
          new_grid.set(x + dx, y + dy, z + dz, c);
        }
      }
    }
  }

  EnergyField new_energy(newX, newY, newZ, R, cfg_.energy_max);
  const int rdx = dx / R, rdy = dy / R, rdz = dz / R;
  for (int rz = 0; rz < energy_.rZ(); ++rz) {
    for (int ry = 0; ry < energy_.rY(); ++ry) {
      for (int rx = 0; rx < energy_.rX(); ++rx) {
        new_energy.at(rx + rdx, ry + rdy, rz + rdz) = energy_.at(rx, ry, rz);
      }
    }
  }

  std::vector<uint32_t> new_owner(
      static_cast<std::size_t>(newX) * newY * newZ, 0u);
  std::vector<uint8_t> new_voxel_role(
      static_cast<std::size_t>(newX) * newY * newZ, ROLE_DENDRITE);
  auto new_lin = [&](int x, int y, int z) {
    return static_cast<std::size_t>(x) +
           static_cast<std::size_t>(y) * newX +
           static_cast<std::size_t>(z) * newX * newY;
  };
  for (int z = 0; z < cfg_.Z; ++z) {
    for (int y = 0; y < cfg_.Y; ++y) {
      for (int x = 0; x < cfg_.X; ++x) {
        new_owner[new_lin(x + dx, y + dy, z + dz)] = owner_[lin(x, y, z)];
        new_voxel_role[new_lin(x + dx, y + dy, z + dz)] =
            voxel_role_[lin(x, y, z)];
      }
    }
  }

  // Shift every neuron coordinate by the offset.
  for (Neuron& nu : neurons_) {
    nu.soma.x = static_cast<int16_t>(nu.soma.x + dx);
    nu.soma.y = static_cast<int16_t>(nu.soma.y + dy);
    nu.soma.z = static_cast<int16_t>(nu.soma.z + dz);
    for (Voxel& v : nu.body) {
      v.x = static_cast<int16_t>(v.x + dx);
      v.y = static_cast<int16_t>(v.y + dy);
      v.z = static_cast<int16_t>(v.z + dz);
    }
    for (SynapseEdge& e : nu.outgoing) {
      e.pos.x = static_cast<int16_t>(e.pos.x + dx);
      e.pos.y = static_cast<int16_t>(e.pos.y + dy);
      e.pos.z = static_cast<int16_t>(e.pos.z + dz);
    }
  }

  // Pack P-lite: shift in-flight delivery events too, so spike volleys
  // launched before the grow still arrive at the right post.
  for (auto& slot : delivery_ring_) {
    for (auto& ev : slot) {
      ev.pos.x = static_cast<int16_t>(ev.pos.x + dx);
      ev.pos.y = static_cast<int16_t>(ev.pos.y + dy);
      ev.pos.z = static_cast<int16_t>(ev.pos.z + dz);
    }
  }

  cfg_.X = newX;
  cfg_.Y = newY;
  cfg_.Z = newZ;
  grid_ = std::move(new_grid);
  energy_ = std::move(new_energy);
  owner_ = std::move(new_owner);
  voxel_role_ = std::move(new_voxel_role);
  vz_lo_ += dz;
  vz_hi_ += dz;
}

bool Simulator::shrink_volume(int dx, int dy, int dz) {
  // Mirror of grow_volume: trim `dx` voxels from each side along x, etc.
  // The to-be-removed rim must be empty (state EMPTY) -- otherwise
  // tissue would be silently destroyed, which is never the right call.
  const int R = cfg_.region_size;
  if (dx < 0 || dy < 0 || dz < 0) return false;
  if (dx == 0 && dy == 0 && dz == 0) return true;
  if (dx % R != 0 || dy % R != 0 || dz % R != 0) return false;
  const int newX = cfg_.X - 2 * dx;
  const int newY = cfg_.Y - 2 * dy;
  const int newZ = cfg_.Z - 2 * dz;
  if (newX <= 0 || newY <= 0 || newZ <= 0) return false;

  // Verify that every voxel falling outside the keep region is EMPTY.
  for (int z = 0; z < cfg_.Z; ++z) {
    const bool keep_z = (z >= dz && z < cfg_.Z - dz);
    for (int y = 0; y < cfg_.Y; ++y) {
      const bool keep_y = (y >= dy && y < cfg_.Y - dy);
      for (int x = 0; x < cfg_.X; ++x) {
        const bool keep_x = (x >= dx && x < cfg_.X - dx);
        if (keep_x && keep_y && keep_z) continue;
        if (grid_.get(x, y, z) != BrainGrid::EMPTY) return false;
      }
    }
  }

  BrainGrid new_grid(newX, newY, newZ);
  for (int z = 0; z < newZ; ++z) {
    for (int y = 0; y < newY; ++y) {
      for (int x = 0; x < newX; ++x) {
        const auto c = grid_.get(x + dx, y + dy, z + dz);
        if (c != BrainGrid::EMPTY) new_grid.set(x, y, z, c);
      }
    }
  }

  EnergyField new_energy(newX, newY, newZ, R, cfg_.energy_max);
  const int rdx = dx / R, rdy = dy / R, rdz = dz / R;
  for (int rz = 0; rz < new_energy.rZ(); ++rz) {
    for (int ry = 0; ry < new_energy.rY(); ++ry) {
      for (int rx = 0; rx < new_energy.rX(); ++rx) {
        new_energy.at(rx, ry, rz) = energy_.at(rx + rdx, ry + rdy, rz + rdz);
      }
    }
  }

  std::vector<uint32_t> new_owner(
      static_cast<std::size_t>(newX) * newY * newZ, 0u);
  auto new_lin = [&](int x, int y, int z) {
    return static_cast<std::size_t>(x) +
           static_cast<std::size_t>(y) * newX +
           static_cast<std::size_t>(z) * newX * newY;
  };
  for (int z = 0; z < newZ; ++z) {
    for (int y = 0; y < newY; ++y) {
      for (int x = 0; x < newX; ++x) {
        new_owner[new_lin(x, y, z)] =
            owner_[lin(x + dx, y + dy, z + dz)];
      }
    }
  }

  for (Neuron& nu : neurons_) {
    nu.soma.x = static_cast<int16_t>(nu.soma.x - dx);
    nu.soma.y = static_cast<int16_t>(nu.soma.y - dy);
    nu.soma.z = static_cast<int16_t>(nu.soma.z - dz);
    for (Voxel& v : nu.body) {
      v.x = static_cast<int16_t>(v.x - dx);
      v.y = static_cast<int16_t>(v.y - dy);
      v.z = static_cast<int16_t>(v.z - dz);
    }
    for (SynapseEdge& e : nu.outgoing) {
      e.pos.x = static_cast<int16_t>(e.pos.x - dx);
      e.pos.y = static_cast<int16_t>(e.pos.y - dy);
      e.pos.z = static_cast<int16_t>(e.pos.z - dz);
    }
  }

  cfg_.X = newX;
  cfg_.Y = newY;
  cfg_.Z = newZ;
  grid_ = std::move(new_grid);
  energy_ = std::move(new_energy);
  owner_ = std::move(new_owner);
  vz_lo_ -= dz; if (vz_lo_ < 0) vz_lo_ = 0;
  vz_hi_ -= dz; if (vz_hi_ < 0) vz_hi_ = 0;
  return true;
}

namespace {

// Iterative flood-fill over a 3D NEURON-state grid: returns either a
// single component count or a vector of component sizes, depending on
// the caller's appetite. Static helper so the public functions stay
// tiny.
void flood_components(const BrainGrid& g, std::vector<int>* sizes,
                       int* count) {
  const int X = g.X(), Y = g.Y(), Z = g.Z();
  const std::size_t N = static_cast<std::size_t>(X) * Y * Z;
  std::vector<uint8_t> visited(N, 0);
  std::vector<int> stack;
  stack.reserve(64);
  auto idx = [&](int x, int y, int z) {
    return static_cast<std::size_t>(x) +
           static_cast<std::size_t>(y) * X +
           static_cast<std::size_t>(z) * X * Y;
  };
  static constexpr int kDx[6] = { 1, -1, 0, 0, 0, 0};
  static constexpr int kDy[6] = { 0, 0, 1, -1, 0, 0};
  static constexpr int kDz[6] = { 0, 0, 0, 0, 1, -1};
  int n_components = 0;
  for (int z = 0; z < Z; ++z) {
    for (int y = 0; y < Y; ++y) {
      for (int x = 0; x < X; ++x) {
        if (visited[idx(x, y, z)]) continue;
        if (g.get(x, y, z) != BrainGrid::NEURON) continue;
        ++n_components;
        int component_size = 0;
        stack.clear();
        stack.push_back(static_cast<int>(idx(x, y, z)));
        visited[idx(x, y, z)] = 1;
        while (!stack.empty()) {
          const int li = stack.back();
          stack.pop_back();
          ++component_size;
          const int xx = li % X;
          const int yy = (li / X) % Y;
          const int zz = li / (X * Y);
          for (int k = 0; k < 6; ++k) {
            const int nx = xx + kDx[k];
            const int ny = yy + kDy[k];
            const int nz = zz + kDz[k];
            if (nx < 0 || nx >= X || ny < 0 || ny >= Y ||
                nz < 0 || nz >= Z) continue;
            const std::size_t ni = idx(nx, ny, nz);
            if (visited[ni]) continue;
            if (g.get(nx, ny, nz) != BrainGrid::NEURON) continue;
            visited[ni] = 1;
            stack.push_back(static_cast<int>(ni));
          }
        }
        if (sizes) sizes->push_back(component_size);
      }
    }
  }
  if (count) *count = n_components;
}

}  // namespace

int Simulator::count_structural_neurons() const {
  int c = 0;
  flood_components(grid_, nullptr, &c);
  return c;
}

std::vector<int> Simulator::structural_neuron_sizes() const {
  std::vector<int> sizes;
  flood_components(grid_, &sizes, nullptr);
  return sizes;
}

void Simulator::seed_fetal(const FetalSeed& f) {
  // 1. Radial-glia scaffold. A small fraction of (x, y) columns are filled
  //    with BLOCKED voxels along z, representing the radial fibres that
  //    later neurons migrate beside. They cannot host synapses, matching the
  //    role of radial glia / structural scaffolding in the developing cortex.
  std::uniform_real_distribution<float> u01(0.0f, 1.0f);
  for (int y = 1; y < cfg_.Y - 1; ++y) {
    for (int x = 1; x < cfg_.X - 1; ++x) {
      if (u01(rng_) >= f.radial_glia_density) continue;
      for (int z = 0; z < cfg_.Z; ++z) {
        if (grid_.get(x, y, z) == BrainGrid::EMPTY) {
          grid_.set(x, y, z, BrainGrid::BLOCKED);
        }
      }
    }
  }

  // Helper to place a single-soma neuron in a constrained z-band.
  std::uniform_int_distribution<int> dx(1, cfg_.X - 2);
  std::uniform_int_distribution<int> dy(1, cfg_.Y - 2);
  auto place_in_band = [&](int z_lo, int z_hi, int leading_process) -> bool {
    if (z_hi < z_lo) return false;
    std::uniform_int_distribution<int> dz(z_lo, z_hi);
    Neuron neu;
    neu.id = static_cast<uint32_t>(neurons_.size() + 1);
    for (int attempt = 0; attempt < 32; ++attempt) {
      const int x = dx(rng_), y = dy(rng_), z = dz(rng_);
      if (!try_set_neuron(x, y, z, neu.id)) continue;
      neu.soma = {static_cast<int16_t>(x),
                  static_cast<int16_t>(y),
                  static_cast<int16_t>(z)};
      neu.body.push_back(neu.soma);
      // Leading process: a small +z extension representing the migrating
      // neuron's pia-directed pioneer dendrite.
      for (int e = 1; e <= leading_process; ++e) {
        const int nz = z + e;
        if (!try_set_neuron(x, y, nz, neu.id)) break;
        neu.body.push_back({static_cast<int16_t>(x),
                            static_cast<int16_t>(y),
                            static_cast<int16_t>(nz)});
      }
      neurons_.push_back(std::move(neu));
      apply_position_prior(neurons_.back());
      return true;
    }
    return false;
  };

  // 2. Ventricular zone: dense, no leading process (still proliferating).
  const int vz_top = std::max(0, std::min(cfg_.Z - 1, f.vz_thickness - 1));
  vz_lo_ = 0;
  vz_hi_ = vz_top;
  for (int i = 0; i < f.vz_neurons; ++i) {
    place_in_band(0, vz_top, 0);
  }

  // 3. Migrating neurons in the intermediate zone, with a leading process.
  const int mig_lo = vz_top + 1;
  const int mig_hi = std::max(mig_lo, cfg_.Z - f.cp_thickness - 1);
  for (int i = 0; i < f.migrating_neurons; ++i) {
    place_in_band(mig_lo, mig_hi, f.leading_process_voxels);
  }

  // 4. Cortical plate: a few early-arrived deep-layer neurons near the pia.
  const int cp_lo = std::max(mig_hi + 1, cfg_.Z - f.cp_thickness);
  const int cp_hi = cfg_.Z - 2;
  for (int i = 0; i < f.cortical_plate_neurons; ++i) {
    place_in_band(cp_lo, cp_hi, 0);
  }

  // 5. Innate subcortical / aversive nuclei. These cohorts represent the
  //    DNA-determined neural primitives a fetus arrives with: brainstem
  //    tonic generators, thalamic relay cells, and an amygdala-analogue
  //    aversive nucleus. Each lives in a tiny dedicated volume near the
  //    VZ floor, separate from the cortical body.
  std::uniform_real_distribution<float> u01_inner(0.0f, 1.0f);

  // Brainstem: a thin band on z=0..1, randomly scattered.
  for (int i = 0; i < f.brainstem_neurons; ++i) {
    place_in_band(0, std::min(1, cfg_.Z - 2), 0);
  }
  // Thalamic relay: at z = 1..2, a small (x, y) cluster.
  for (int i = 0; i < f.thalamic_relay_neurons; ++i) {
    place_in_band(std::min(1, cfg_.Z - 2),
                  std::min(2, cfg_.Z - 2), 0);
  }
  // Amygdala-analogue aversive nucleus: also low z but offset.
  for (int i = 0; i < f.aversive_nucleus_neurons; ++i) {
    place_in_band(std::min(2, cfg_.Z - 2),
                  std::min(3, cfg_.Z - 2), 0);
  }

  // 6. GABAergic subtype assignment. After all cortical placements are
  //    done, randomly designate the configured fractions as PV, SST and
  //    VIP. The remainder stays excitatory. Cell identity is permanent
  //    (Dale's principle); demos can override per-neuron with
  //    set_polarity() if they need specific layouts.
  if (f.frac_pv + f.frac_sst + f.frac_vip > 0.0f) {
    for (Neuron& nu : neurons_) {
      const float r = u01_inner(rng_);
      if (r < f.frac_pv) {
        nu.polarity = NeuronPolarity::INHIBITORY;        // PV
      } else if (r < f.frac_pv + f.frac_sst) {
        nu.polarity = NeuronPolarity::INHIBITORY_SST;
      } else if (r < f.frac_pv + f.frac_sst + f.frac_vip) {
        nu.polarity = NeuronPolarity::INHIBITORY_VIP;
      } else {
        nu.polarity = NeuronPolarity::EXCITATORY;
      }
    }
  }

  // 7. Energy gradient: high near the VZ (where neurogenesis burns glucose),
  //    low near the cortical plate (which is metabolically quiet at this
  //    fetal stage). Scale is a fraction of `energy_max` so the regenerate
  //    cap continues to make sense.
  const int rZ = energy_.rZ();
  for (int rz = 0; rz < rZ; ++rz) {
    const float zf = (rZ <= 1) ? 0.0f
                                : static_cast<float>(rz) /
                                      static_cast<float>(rZ - 1);
    const float scale = (1.0f - zf) * f.vz_energy_scale + zf * f.cp_energy_scale;
    const float e0 = cfg_.energy_max * scale;
    for (int ry = 0; ry < energy_.rY(); ++ry) {
      for (int rx = 0; rx < energy_.rX(); ++rx) {
        energy_.at(rx, ry, rz) = e0;
      }
    }
  }
}

void Simulator::set_polarity(uint32_t neuron_id, NeuronPolarity pol) {
  if (neuron_id == 0 || neuron_id > neurons_.size()) return;
  neurons_[neuron_id - 1].polarity = pol;
  // Pack M: re-stamp morphology since each polarity has a distinct
  // shape (PV's perisomatic axonal arbor vs SST's ascending axon
  // vs pyramidal's apical dendrite, etc.).
  stamp_morphology(neuron_id);
}

void Simulator::randomize_polarity(float inhibitory_fraction) {
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for (Neuron& nu : neurons_) {
    nu.polarity = (u(rng_) < inhibitory_fraction)
                      ? NeuronPolarity::INHIBITORY
                      : NeuronPolarity::EXCITATORY;
  }
}

void Simulator::install_synapse(uint32_t pre_id, uint32_t post_id,
                                 float weight, int conduction_delay,
                                 uint8_t branch, float innate_tag) {
  if (pre_id == 0 || pre_id > neurons_.size()) return;
  if (post_id == 0 || post_id > neurons_.size()) return;
  Neuron& pre = neurons_[pre_id - 1];
  Neuron& post = neurons_[post_id - 1];
  SynapseEdge edge;
  edge.target_neuron = post_id;
  edge.pos = pre.soma;  // unused for non-grid synapses
  edge.weight = weight;
  edge.last_active_step = step_;
  edge.last_delivery_step = step_ - 10000;
  edge.conduction_delay = std::max(1, conduction_delay);
  // Clamp branch to the post's dendrite count so an over-eager caller
  // can't write past the branch_potential vector.
  edge.branch = (post.n_branches > 0)
                    ? static_cast<uint8_t>(std::min<uint32_t>(
                          branch, post.n_branches - 1))
                    : 0;
  // Innate / labelled-line synapses can be tagged at install time so
  // they're protected from "use it or lose it" spine retraction --
  // matching the way real cortex protects key reflex-arc connections
  // independently of postnatal experience. A non-zero innate_tag also
  // marks the synapse `permanent`: it survives both spine retraction
  // and the silence-timeout sweeps regardless of activity.
  edge.consolidation_tag = innate_tag;
  edge.permanent = (innate_tag > 0.0f);
  // Pack ZZ v3: permanent synapses are CD47-protected from microglial
  // elimination. Sentinel ~ infinity means microglia_phase never
  // touches them regardless of their `eat_me_tag` history.
  edge.dont_eat_me = edge.permanent ? 1e9f : 0.0f;
  pre.outgoing.push_back(edge);
}

void Simulator::set_branches(uint32_t neuron_id, uint8_t n_branches) {
  if (neuron_id == 0 || neuron_id > neurons_.size()) return;
  if (n_branches == 0) n_branches = 1;
  Neuron& nu = neurons_[neuron_id - 1];
  nu.n_branches = n_branches;
  nu.branch_potential.assign(n_branches, 0.0f);
  // Per-branch overrides start empty -> fall through to global cfg.
}

void Simulator::set_branch_threshold(uint32_t neuron_id, uint8_t branch,
                                      float threshold) {
  if (neuron_id == 0 || neuron_id > neurons_.size()) return;
  Neuron& nu = neurons_[neuron_id - 1];
  if (branch >= nu.n_branches) return;
  if (nu.branch_threshold.size() < nu.n_branches) {
    nu.branch_threshold.resize(nu.n_branches,
                               std::numeric_limits<float>::quiet_NaN());
  }
  nu.branch_threshold[branch] = threshold;
}

void Simulator::set_branch_passive_gain(uint32_t neuron_id, uint8_t branch,
                                         float gain) {
  if (neuron_id == 0 || neuron_id > neurons_.size()) return;
  Neuron& nu = neurons_[neuron_id - 1];
  if (branch >= nu.n_branches) return;
  if (nu.branch_passive_gain.size() < nu.n_branches) {
    nu.branch_passive_gain.resize(nu.n_branches,
                                  std::numeric_limits<float>::quiet_NaN());
  }
  nu.branch_passive_gain[branch] = gain;
}

// ---- Pack M: morphology templates per cell type ---------------------------
//
// Each template gives the cell a tiny, biologically-oriented 3D shape
// stamped at birth. v2 starts with 1-voxel templates (apical / lateral
// hint per polarity) since Pack M v1 with 1-voxel templates regressed
// against the pre-Pack-ZZ 75% baseline by 1 word; with Pack ZZ active
// the substrate sheds surplus synapses and the same templates may now
// fit. Larger templates can be tried after this lands.
namespace {

// Phase 1 morphology refactor: stamp templates as NEURON-state with
// role=AXON (1). Soma defaults to DENDRITE (0); sprouted voxels also
// default to DENDRITE. Synaptogenesis only forms a contact when an
// AXON voxel of one neuron meets a DENDRITE voxel of another --
// random NEURON x NEURON contacts no longer become synapses. This
// is the canonical "axon-of-pre / dendrite-of-post" pairing of real
// cortical chemistry.
//
// Each cell type contributes one AXON voxel along its preferred
// projection axis:
//   pyramidal       -- axon descends -z (toward white matter)
//   PV basket       -- local lateral axon (+x)
//   SST Martinotti  -- ascending axon (+z, toward layer 1)
//   VIP             -- local axon to other inhibitories (+y)
// Phase 1' expansion: pyramidal cells get a richer morphology -- real
// layer-2/3 and layer-5 pyramidals have an apical dendrite ascending
// to layer 1, basal dendrites radiating laterally from the soma, and
// a descending axon to white matter (DeFelipe et al. 2013 *Nat. Rev.
// Neurosci.* 14:202). The voxel-coarse approximation:
//   (0, 0, +1) apical dendrite        [DENDRITE]
//   (+1, 0, 0) basal dendrite         [DENDRITE]
//   (-1, 0, 0) basal dendrite         [DENDRITE]
//   (0, +1, 0) basal dendrite         [DENDRITE]
//   (0, -1, 0) basal dendrite         [DENDRITE]
//   (0, 0, -1) descending axon        [AXON]
// 5 dendrite voxels (apical + 4 basal) + 1 axon = 6 morphology voxels
// per pyramidal. Each cell now has a real receiving surface in 5
// directions and a real projection voxel below the soma.
constexpr MorphologyVoxel kMorphPyramidal[] = {
    { 0, 0,  1, /*DENDRITE*/ 0},               // apical dendrite
    { 1, 0,  0, /*DENDRITE*/ 0},               // basal +x
    {-1, 0,  0, /*DENDRITE*/ 0},               // basal -x
    { 0,  1, 0, /*DENDRITE*/ 0},               // basal +y
    { 0, -1, 0, /*DENDRITE*/ 0},               // basal -y
    { 0, 0, -1, /*AXON    */ 1},               // descending axon
};
// PV basket: dense local axonal arbor + multipolar dendrites
// (Tremblay, Lee & Rudy 2016 *Neuron* 91:260). 4 lateral axon voxels
// for perisomatic targeting + 2 dendrite voxels above/below soma.
constexpr MorphologyVoxel kMorphPv[] = {
    { 1, 0,  0, /*AXON    */ 1},
    {-1, 0,  0, /*AXON    */ 1},
    { 0,  1, 0, /*AXON    */ 1},
    { 0, -1, 0, /*AXON    */ 1},
    { 0, 0,  1, /*DENDRITE*/ 0},
    { 0, 0, -1, /*DENDRITE*/ 0},
};
// SST Martinotti: ascending axon to layer 1 + bipolar dendrites.
constexpr MorphologyVoxel kMorphSst[] = {
    { 0, 0,  1, /*AXON    */ 1},
    { 0, 0,  2, /*AXON    */ 1},
    { 1, 0,  0, /*DENDRITE*/ 0},
    {-1, 0,  0, /*DENDRITE*/ 0},
};
// VIP: local axon to other inhibitories + lateral dendrites.
constexpr MorphologyVoxel kMorphVip[] = {
    { 0,  1, 0, /*AXON    */ 1},
    { 0, -1, 0, /*AXON    */ 1},
    { 1, 0,  0, /*DENDRITE*/ 0},
    {-1, 0,  0, /*DENDRITE*/ 0},
};

MorphologyTemplate morphology_for(NeuronPolarity pol, NeuronRole role) {
  // INPUT / OUTPUT cells use hand-installed labelled-line synapses
  // tuned against the single-voxel baseline; adding morphology around
  // them creates spurious NEURON-NEURON contacts that synaptogenesis
  // turns into noise. Skip them in v2.
  if (role == NeuronRole::INPUT || role == NeuronRole::OUTPUT) {
    return {nullptr, 0};
  }
  switch (pol) {
    case NeuronPolarity::INHIBITORY:
      return {kMorphPv,   static_cast<int>(std::size(kMorphPv))};
    case NeuronPolarity::INHIBITORY_SST:
      return {kMorphSst,  static_cast<int>(std::size(kMorphSst))};
    case NeuronPolarity::INHIBITORY_VIP:
      return {kMorphVip,  static_cast<int>(std::size(kMorphVip))};
    case NeuronPolarity::EXCITATORY:
    default:
      return {kMorphPyramidal,
              static_cast<int>(std::size(kMorphPyramidal))};
  }
}

}  // namespace

int Simulator::stamp_morphology(uint32_t neuron_id) {
  if (neuron_id == 0 || neuron_id > neurons_.size()) return 0;
  Neuron& nu = neurons_[neuron_id - 1];
  // Idempotent: first clear any previously-stamped non-soma body /
  // BLOCKED voxels still owned by this cell. Sprouted voxels (post
  // simulation steps) would also be cleared here, but `set_role` /
  // `set_polarity` are normally called immediately after
  // `add_neuron_at` before any sprouting has run.
  for (auto it = nu.body.begin(); it != nu.body.end();) {
    if (it->x == nu.soma.x && it->y == nu.soma.y &&
        it->z == nu.soma.z) {
      ++it;
      continue;
    }
    if (grid_.in_bounds(it->x, it->y, it->z)) {
      grid_.set(it->x, it->y, it->z, BrainGrid::EMPTY);
      owner_[lin(it->x, it->y, it->z)] = 0;
      voxel_role_[lin(it->x, it->y, it->z)] = ROLE_DENDRITE;
    }
    it = nu.body.erase(it);
  }
  const MorphologyTemplate t = morphology_for(nu.polarity, nu.role);
  int placed = 0;
  for (int i = 0; i < t.n; ++i) {
    const MorphologyVoxel& v = t.voxels[i];
    const int x = nu.soma.x + v.dx;
    const int y = nu.soma.y + v.dy;
    const int z = nu.soma.z + v.dz;
    if (!grid_.in_bounds(x, y, z)) continue;
    if (grid_.get(x, y, z) != BrainGrid::EMPTY) continue;
    if (v.role == 2) {
      // Axon-trunk: BLOCKED state. Conducts but does not form
      // synapses. Owned by this neuron but NOT in `body` so
      // sprouting iterators skip it.
      grid_.set(x, y, z, BrainGrid::BLOCKED);
      owner_[lin(x, y, z)] = nu.id;
      voxel_role_[lin(x, y, z)] = ROLE_AXON_TRUNK;
    } else {
      // Dendrite or axon: NEURON state, eligible for synaptogenesis
      // *if its role matches the contact direction* (Phase 1).
      grid_.set(x, y, z, BrainGrid::NEURON);
      owner_[lin(x, y, z)] = nu.id;
      voxel_role_[lin(x, y, z)] =
          (v.role == 1) ? ROLE_AXON : ROLE_DENDRITE;
      nu.body.push_back({static_cast<int16_t>(x),
                          static_cast<int16_t>(y),
                          static_cast<int16_t>(z)});
    }
    ++placed;
  }
  return placed;
}

uint32_t Simulator::add_neuron_at(int x, int y, int z) {
  Neuron neu;
  neu.id = static_cast<uint32_t>(neurons_.size() + 1);
  if (!try_set_neuron(x, y, z, neu.id)) return 0;
  neu.soma = {static_cast<int16_t>(x),
              static_cast<int16_t>(y),
              static_cast<int16_t>(z)};
  neu.body.push_back(neu.soma);
  neurons_.push_back(std::move(neu));
  // Soft cortical-map prior on the just-pushed neuron.
  apply_position_prior(neurons_.back());
  // Pack M: stamp the polarity / role -default morphology so the cell
  // is born with a real shape, not a single voxel. Callers that change
  // polarity / role afterwards trigger a re-stamp from inside
  // `set_polarity` / `set_role`.
  stamp_morphology(static_cast<uint32_t>(neurons_.size()));
  return static_cast<uint32_t>(neurons_.size());
}

void Simulator::inject_input(uint32_t neuron_id, float amount) {
  if (neuron_id == 0 || neuron_id > neurons_.size()) return;
  neurons_[neuron_id - 1].input_acc += amount;
}

void Simulator::set_role(uint32_t neuron_id, NeuronRole role, int channel) {
  if (neuron_id == 0 || neuron_id > neurons_.size()) return;
  Neuron& nu = neurons_[neuron_id - 1];
  nu.role = role;
  nu.channel = channel;
  // Pack M: re-stamp morphology since INPUT / OUTPUT cells use a
  // different (currently empty) template than INTERNAL.
  stamp_morphology(neuron_id);
}

void Simulator::apply_input_pattern(const float* features, int n_features) {
  // Drive every INPUT neuron whose channel index lies in [0, n_features).
  // Replaces input_acc rather than adding; the chemistry phase later treats
  // INPUT neurons specially so the external feature dominates.
  for (Neuron& nu : neurons_) {
    if (nu.role != NeuronRole::INPUT) continue;
    if (nu.channel < 0 || nu.channel >= n_features) continue;
    nu.input_acc = features[nu.channel] * cfg_.input_drive_strength;
  }
}

void Simulator::apply_prediction_pattern(const float* predictions,
                                          int n_features) {
  for (Neuron& nu : neurons_) {
    if (nu.role != NeuronRole::INPUT) continue;
    if (nu.channel < 0 || nu.channel >= n_features) continue;
    nu.predicted_input = predictions[nu.channel] * cfg_.input_drive_strength;
  }
}

void Simulator::read_output(float* out, int n_classes) const {
  std::vector<int> count(n_classes, 0);
  for (int i = 0; i < n_classes; ++i) out[i] = 0.0f;
  for (const Neuron& nu : neurons_) {
    if (nu.role != NeuronRole::OUTPUT) continue;
    if (nu.channel < 0 || nu.channel >= n_classes) continue;
    out[nu.channel] += nu.fire_rate_ema;
    ++count[nu.channel];
  }
  for (int i = 0; i < n_classes; ++i) {
    if (count[i] > 0) out[i] /= static_cast<float>(count[i]);
  }
}

void Simulator::apply_reward(float reward) {
  // Each synapse independently turns its own eligibility trace into a
  // weight change. No information beyond `weight`, `eligibility` and the
  // global scalar `reward` is consulted -- so the local-update property
  // is preserved. Serotonin scales the consolidation rate.
  const float lr = cfg_.reward_lr * cfg_.serotonin_level;
  const int nn = static_cast<int>(neurons_.size());
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nn; ++i) {
    for (auto& syn : neurons_[i].outgoing) {
      float w = syn.weight + lr * reward * syn.eligibility;
      if (w > cfg_.weight_max) w = cfg_.weight_max;
      if (w < 0.0f) w = 0.0f;
      syn.weight = w;
    }
  }
}

void Simulator::apply_reward_per_class(const float* rewards, int n_classes,
                                        float internal_reward) {
  const float lr = cfg_.reward_lr * cfg_.serotonin_level;
  const int nn = static_cast<int>(neurons_.size());
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nn; ++i) {
    for (auto& syn : neurons_[i].outgoing) {
      float r = internal_reward;
      if (syn.target_neuron != 0 && syn.target_neuron <= neurons_.size()) {
        const Neuron& post = neurons_[syn.target_neuron - 1];
        if (post.role == NeuronRole::OUTPUT &&
            post.channel >= 0 && post.channel < n_classes) {
          r = rewards[post.channel];
        }
      }
      // Permanent (engram) synapses ignore *negative* reward signals:
      // a previously-confirmed recall path cannot be unlearned by a
      // later curriculum item that simply targets a different class.
      // Positive rewards are still allowed so engrams can strengthen.
      if (syn.permanent && r < 0.0f) continue;
      float w = syn.weight + lr * r * syn.eligibility;
      if (w > cfg_.weight_max) w = cfg_.weight_max;
      if (w < 0.0f) w = 0.0f;
      syn.weight = w;
    }
  }
}

void Simulator::apply_aversive(float intensity) {
  // Aversive plasticity is asymmetric: excitatory synapses that
  // contributed to the recently-evaluated trajectory get *weakened*
  // (don't repeat this excitation), while inhibitory synapses that
  // contributed get *strengthened* (gate against repetition next time).
  // Per-synapse local update -- only the synapse's own pre/post and the
  // global aversive signal are read.
  //
  // Critically, the signal is restricted to synapses whose *post* is
  // an OUTPUT neuron. Real aversive learning preferentially reshapes
  // the action selection step (motor / decision pathways), leaving
  // sensory and detection pathways intact. Without this filter the
  // very synapse that *detected* the danger (e.g. pain-receptor ->
  // amygdala) would be weakened by its own success report.
  const float lr = cfg_.reward_lr * cfg_.aversive_amplification *
                   cfg_.serotonin_level;
  const int nn = static_cast<int>(neurons_.size());
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nn; ++i) {
    Neuron& pre = neurons_[i];
    const bool inhibitory =
        (pre.polarity == NeuronPolarity::INHIBITORY ||
         pre.polarity == NeuronPolarity::INHIBITORY_SST ||
         pre.polarity == NeuronPolarity::INHIBITORY_VIP);
    const float sign = inhibitory ? +1.0f : -1.0f;
    for (auto& syn : pre.outgoing) {
      if (syn.target_neuron == 0 || syn.target_neuron > neurons_.size())
        continue;
      const Neuron& post = neurons_[syn.target_neuron - 1];
      if (post.role != NeuronRole::OUTPUT) continue;
      // Permanent (engram) synapses are exempt from aversive weakening
      // for the same reason they ignore negative reward: a confirmed
      // recall path is not undone by later wrong-class punishment.
      if (syn.permanent) continue;
      float dw = sign * lr * intensity * syn.eligibility;
      float w = syn.weight + dw;
      if (w > cfg_.weight_max) w = cfg_.weight_max;
      if (w < 0.0f) w = 0.0f;
      syn.weight = w;
    }
  }
}

void Simulator::clear_eligibility() {
  for (Neuron& nu : neurons_) {
    for (auto& syn : nu.outgoing) syn.eligibility = 0.0f;
  }
}

void Simulator::reset_dynamics() {
  for (Neuron& nu : neurons_) {
    nu.potential = 0.0f;
    nu.input_acc = 0.0f;
    nu.fire_rate_ema = 0.0f;
    nu.fired_this_step = false;
    // Push last_fire_step far enough in the past that any refractory
    // window the demo configures is already past.
    nu.last_fire_step = step_ - 1000000;
    nu.incoming_queue.clear();
    std::fill(nu.branch_potential.begin(), nu.branch_potential.end(), 0.0f);
    nu.ltp_received_this_step = 0.0f;
    for (auto& syn : nu.outgoing) {
      syn.transit.clear();
    }
  }
  // Pack P-lite: drop any in-flight delivery events too. The ring
  // buffer is transient state; it does not survive a dynamics reset.
  for (auto& slot : delivery_ring_) slot.clear();
}

void Simulator::set_excitability_bias(uint32_t neuron_id, float value) {
  if (neuron_id == 0 || neuron_id > neurons_.size()) return;
  neurons_[neuron_id - 1].excitability_bias =
      value < 0.0f ? 0.0f : value;
}

void Simulator::set_session_id(int session_id) {
  current_session_id_ = session_id;
}

void Simulator::set_engram_region(int output_channel, int x, int y, int z,
                                    int radius) {
  if (output_channel < 0) return;
  if (static_cast<std::size_t>(output_channel) >= class_regions_.size()) {
    class_regions_.resize(static_cast<std::size_t>(output_channel) + 1);
  }
  class_regions_[output_channel] = {x, y, z, std::max(0, radius)};
}

int Simulator::promote_engram(int output_channel, int top_k_internal,
                              bool silent) {
  if (output_channel < 0) return 0;
  if (static_cast<std::size_t>(output_channel) >= engram_members_.size()) {
    engram_members_.resize(static_cast<std::size_t>(output_channel) + 1);
  }
  if (static_cast<std::size_t>(output_channel) >= class_session_.size()) {
    class_session_.resize(static_cast<std::size_t>(output_channel) + 1, -1);
  }

  // 1. Persistent membership: always re-include the existing engram
  //    set for this class. Repetition reinforces the same cell
  //    assembly -- new internal neurons are admitted only to top up
  //    a too-small engram, not to replace existing members. Plus
  //    every OUTPUT neuron whose channel == output_channel.
  std::vector<uint32_t> engram_ids = engram_members_[output_channel];
  for (const Neuron& n : neurons_) {
    if (n.role == NeuronRole::OUTPUT && n.channel == output_channel) {
      engram_ids.push_back(n.id);
    }
  }
  std::sort(engram_ids.begin(), engram_ids.end());
  engram_ids.erase(std::unique(engram_ids.begin(), engram_ids.end()),
                   engram_ids.end());

  // 2. Top up with currently-firing INTERNAL neurons not yet in the
  //    engram, until membership reaches the target size. The floor
  //    (0.02) keeps silent cells out even if the engram is small.
  //    Score is fire_rate_ema, but neurons already enrolled in
  //    *other* classes' engrams take a heavy penalty -- we strongly
  //    prefer fresh hubs so different words get distinct cell
  //    assemblies. Without this penalty, top-K by raw rate keeps
  //    picking the same mixed-selectivity hubs for every word and
  //    every promotion reinforces those shared hubs onto every
  //    motor, collapsing recall into mush.
  // Track *which* other class each candidate is enrolled in, so the
  // memory-linking rule can soften the fresh-neuron penalty between
  // classes acquired in the same session (Pack 25). 0 = no other
  // engram; otherwise stores (other_class + 1) for the most recent
  // owner found.
  std::vector<int> other_engram_class(neurons_.size() + 1, 0);
  for (std::size_t cls = 0; cls < engram_members_.size(); ++cls) {
    if (static_cast<int>(cls) == output_channel) continue;
    for (uint32_t id : engram_members_[cls]) {
      if (id <= neurons_.size())
        other_engram_class[id] = static_cast<int>(cls) + 1;
    }
  }
  // Per-class preferred niche: candidates inside the sphere get a
  // multiplicative score boost; candidates outside get a penalty.
  // This drives different words into different cortical regions,
  // mirroring the column / area specialisation of real cortex.
  ClassRegion region;
  if (output_channel >= 0 &&
      static_cast<std::size_t>(output_channel) < class_regions_.size()) {
    region = class_regions_[output_channel];
  }
  const bool region_active = region.radius > 0;
  const float r2 = static_cast<float>(region.radius) * region.radius;
  std::vector<std::pair<float, uint32_t>> ranked;
  for (const Neuron& n : neurons_) {
    if (n.role != NeuronRole::INTERNAL) continue;
    if (n.fire_rate_ema <= 0.02f) continue;
    if (std::binary_search(engram_ids.begin(), engram_ids.end(), n.id))
      continue;
    // CREB-style allocation bias (Pack 25): rank by activity scaled
    // by intrinsic excitability, not raw fire_rate_ema. The 0.02
    // floor above still guards against silent cells slipping in via
    // a high bias alone.
    float score = n.fire_rate_ema * n.excitability_bias;
    if (int other = other_engram_class[n.id]; other > 0) {
      const int other_cls = other - 1;
      bool linked = false;
      if (static_cast<std::size_t>(other_cls) < class_session_.size()) {
        const int other_session = class_session_[other_cls];
        linked = (other_session >= 0 &&
                  other_session == current_session_id_);
      }
      score *= linked ? 0.5f : 0.1f;  // soft penalty for memory-linked
    }
    if (region_active) {
      const float dx = n.soma.x - region.x;
      const float dy = n.soma.y - region.y;
      const float dz = n.soma.z - region.z;
      const float d2 = dx * dx + dy * dy + dz * dz;
      float niche;
      // Inside the sphere -> 2x boost. Outside but within 2*radius
      // -> linear falloff to 0.5x. Far outside -> 0.25x.
      if (d2 <= r2) niche = 2.0f;
      else if (d2 <= 4.0f * r2) {
        const float t = (d2 - r2) / (3.0f * r2);  // 0..1
        niche = 2.0f - 1.5f * t;  // 2.0 -> 0.5
      } else {
        niche = 0.25f;
      }
      // Pack 25.1: a strongly CREB-biased candidate is anatomically
      // pre-committed elsewhere (e.g. an A1 cell during a teach
      // episode) -- the niche penalty would otherwise let an
      // unbiased noise neuron in the motor column displace it.
      // Clamp the niche floor to 0.75x for cells with bias > 1.5
      // so molecular pre-encoding can override spatial bias.
      if (n.excitability_bias > 1.5f && niche < 0.75f) niche = 0.75f;
      score *= niche;
    }
    ranked.emplace_back(score, n.id);
  }
  const int existing_internal =
      static_cast<int>(engram_ids.size()) -
      // Subtract OUTPUT neurons (channel match) so the budget for
      // INTERNAL members is independent of motor count.
      [&]{
        int n = 0;
        for (uint32_t id : engram_ids) {
          if (id == 0 || id > neurons_.size()) continue;
          const Neuron& nu = neurons_[id - 1];
          if (nu.role == NeuronRole::OUTPUT &&
              nu.channel == output_channel) ++n;
        }
        return n;
      }();
  const int budget = std::max(0, top_k_internal - existing_internal);
  const int k = std::min<int>(budget, static_cast<int>(ranked.size()));
  if (k > 0) {
    std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(),
                      [](const auto& a, const auto& b) {
                        return a.first > b.first;
                      });
    for (int i = 0; i < k; ++i) engram_ids.push_back(ranked[i].second);
    std::sort(engram_ids.begin(), engram_ids.end());
  }
  engram_members_[output_channel] = engram_ids;
  class_session_[output_channel] = current_session_id_;

  // 2. Mark every synapse whose pre AND post are both in the engram.
  int n_marked = 0;
  for (Neuron& n : neurons_) {
    if (!std::binary_search(engram_ids.begin(), engram_ids.end(), n.id))
      continue;
    for (SynapseEdge& s : n.outgoing) {
      if (!std::binary_search(engram_ids.begin(), engram_ids.end(),
                              s.target_neuron))
        continue;
      if (!s.permanent) ++n_marked;
      s.permanent = true;
      s.consolidation_tag = 1.0f;
      // Pack ZZ v3: engram-promoted edges get CD47-style protection
      // alongside the labelled-line priors.
      s.dont_eat_me = 1e9f;
      if (silent) continue;  // silent engram: skip weight floor
      // Floor the engram-edge weight at half of weight_max so the
      // recall path is electrically functional even if STDP had not
      // yet driven it high. The synapse is still free to potentiate
      // further; this only prevents an under-trained path from being
      // promoted without enough drive to reactivate.
      const float engram_floor = 0.5f * cfg_.weight_max;
      if (s.weight < engram_floor) s.weight = engram_floor;
    }
  }
  return n_marked;
}

std::size_t Simulator::permanent_synapse_count() const noexcept {
  std::size_t n = 0;
  for (const Neuron& nu : neurons_) {
    for (const SynapseEdge& s : nu.outgoing) {
      if (s.permanent) ++n;
    }
  }
  return n;
}

NetworkStats Simulator::network_stats(int top_k_hubs) const {
  NetworkStats s;
  const int N = static_cast<int>(neurons_.size());
  s.n_neurons = N;
  if (N == 0) return s;

  // First pass: build per-neuron in/out degree, accumulate weight
  // distribution + permanence count.
  std::vector<int> in_deg(N, 0);
  std::vector<int> out_deg(N, 0);
  double weight_sum = 0.0;
  int weight_count = 0;
  int permanent = 0;
  for (int i = 0; i < N; ++i) {
    const Neuron& pre = neurons_[i];
    out_deg[i] = static_cast<int>(pre.outgoing.size());
    for (const SynapseEdge& syn : pre.outgoing) {
      if (syn.target_neuron == 0 ||
          syn.target_neuron > static_cast<uint32_t>(N))
        continue;
      ++in_deg[syn.target_neuron - 1];
      weight_sum += syn.weight;
      if (syn.weight > s.max_weight) s.max_weight = syn.weight;
      ++weight_count;
      if (syn.permanent) ++permanent;
    }
    s.n_synapses += out_deg[i];
  }
  s.n_permanent = permanent;
  s.mean_weight = weight_count > 0
                      ? static_cast<float>(weight_sum / weight_count)
                      : 0.0f;

  // Degree summary.
  double in_sum = 0.0, out_sum = 0.0;
  double tot_sum = 0.0, tot_sumsq = 0.0;
  for (int i = 0; i < N; ++i) {
    in_sum += in_deg[i];
    out_sum += out_deg[i];
    if (in_deg[i] > s.max_in_degree) s.max_in_degree = in_deg[i];
    if (out_deg[i] > s.max_out_degree) s.max_out_degree = out_deg[i];
    const double t = in_deg[i] + out_deg[i];
    tot_sum += t;
    tot_sumsq += t * t;
  }
  s.mean_in_degree = static_cast<float>(in_sum / N);
  s.mean_out_degree = static_cast<float>(out_sum / N);
  const double tot_mean = tot_sum / N;
  const double tot_var = std::max(0.0, tot_sumsq / N - tot_mean * tot_mean);
  s.std_total_degree = static_cast<float>(std::sqrt(tot_var));

  // Activity summary.
  double rate_sum = 0.0;
  int active = 0;
  for (const Neuron& nu : neurons_) {
    rate_sum += nu.fire_rate_ema;
    if (nu.fire_rate_ema > 0.05f) ++active;
  }
  s.mean_fire_rate_ema = static_cast<float>(rate_sum / N);
  s.n_active = active;

  // Top-K hubs by total degree (partial_sort on a temporary index).
  std::vector<int> idx(N);
  for (int i = 0; i < N; ++i) idx[i] = i;
  const int k = std::min(std::max(0, top_k_hubs), N);
  if (k > 0) {
    std::partial_sort(
        idx.begin(), idx.begin() + k, idx.end(),
        [&](int a, int b) {
          return (in_deg[a] + out_deg[a]) > (in_deg[b] + out_deg[b]);
        });
    s.top_hubs.reserve(k);
    for (int rank = 0; rank < k; ++rank) {
      const int i = idx[rank];
      s.top_hubs.push_back({neurons_[i].id, in_deg[i], out_deg[i],
                            in_deg[i] + out_deg[i], neurons_[i].soma});
    }
  }
  return s;
}

float Simulator::occupancy_fraction() const noexcept {
  const std::size_t vol = grid_.volume();
  if (vol == 0) return 0.0f;
  std::size_t occupied = 0;
  for (int z = 0; z < grid_.Z(); ++z) {
    for (int y = 0; y < grid_.Y(); ++y) {
      for (int x = 0; x < grid_.X(); ++x) {
        const auto c = grid_.get(x, y, z);
        if (c == BrainGrid::NEURON || c == BrainGrid::SYNAPSE) ++occupied;
      }
    }
  }
  return static_cast<float>(occupied) / static_cast<float>(vol);
}

void Simulator::integrate_incoming_phase() {
  // Stage 1->2 boundary: synaptic inputs that scheduler_dispatch_phase
  // accumulated into per-branch dendritic potentials get folded into the
  // soma's input_acc. Each branch independently checks for an NMDA-style
  // dendritic spike (a strong, stereotyped soma drive) or contributes a
  // passively-attenuated portion of its sub-threshold sum.
  //
  // For backward compatibility the default config sets dendritic_decay=0,
  // dendritic_passive_gain=1, dendritic_threshold=inf, which makes a
  // single-branch neuron behaviourally identical to the legacy "all
  // synaptic input goes straight into input_acc" model.
  const int nn = static_cast<int>(neurons_.size());
  const float threshold = cfg_.dendritic_threshold;
  const float spike_amp = cfg_.dendritic_spike_amplitude;
  const float passive = cfg_.dendritic_passive_gain;
  const float ddecay = cfg_.dendritic_decay;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nn; ++i) {
    Neuron& nu = neurons_[i];
    if (nu.branch_potential.size() != nu.n_branches) {
      nu.branch_potential.assign(nu.n_branches, 0.0f);
    }
    for (uint8_t b = 0; b < nu.n_branches; ++b) {
      float& bp = nu.branch_potential[b];
      // Per-branch overrides fall back to global cfg if not set or NaN.
      const float br_threshold =
          (b < nu.branch_threshold.size() &&
           !std::isnan(nu.branch_threshold[b]))
              ? nu.branch_threshold[b]
              : threshold;
      const float br_passive =
          (b < nu.branch_passive_gain.size() &&
           !std::isnan(nu.branch_passive_gain[b]))
              ? nu.branch_passive_gain[b]
              : passive;
      if (bp >= br_threshold) {
        nu.input_acc += spike_amp;
        bp = 0.0f;
      } else {
        nu.input_acc += bp * br_passive;
        bp *= ddecay;
      }
    }
    // Drain the legacy queue too (some demos / external paths may still
    // use inject_input which may queue here in older builds).
    if (!nu.incoming_queue.empty()) {
      float sum = 0.0f;
      for (float v : nu.incoming_queue) sum += v;
      nu.input_acc += sum;
      nu.incoming_queue.clear();
    }
  }
}

void Simulator::chemistry_phase() {
  // Per-neuron leaky integrate-and-fire. Each neuron is its own independent
  // computational unit; the loop is embarrassingly parallel.
  const int nn = static_cast<int>(neurons_.size());
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nn; ++i) {
    Neuron& nu = neurons_[i];
    nu.fired_this_step = false;
    // For INPUT neurons we let the externally-applied input dominate by not
    // adding any leakage from previous potential -- this gives the data
    // signal a clean foothold each step. Internal neurons integrate.
    // Predictive-coding subtraction: INPUT neurons' effective drive is
    // input_acc minus predicted_input (clipped at 0 -- predictions
    // never invert the sign of an actual stimulus). When prediction
    // matches actual exactly, no surprise -> no firing. Defaults
    // leave predicted_input at 0 so non-PC demos behave identically.
    if (nu.role == NeuronRole::INPUT) {
      const float effective = nu.input_acc - nu.predicted_input;
      nu.potential = effective > 0.0f ? effective : 0.0f;
    } else {
      nu.potential = nu.potential * cfg_.potential_decay + nu.input_acc;
    }
    nu.input_acc = 0.0f;
    nu.predicted_input = 0.0f;

    // Refractory period: after a recent spike the soma cannot fire
    // again for `refractory_steps` steps. The accumulated potential
    // is reset to 0 so it doesn't get to ride out the window. With
    // refractory_steps == 0 (default) this is a no-op.
    const bool in_refractory =
        cfg_.refractory_steps > 0 &&
        (step_ - nu.last_fire_step) < cfg_.refractory_steps;
    if (in_refractory) {
      nu.potential = 0.0f;
    }
    if (nu.potential >= cfg_.fire_threshold && !in_refractory) {
      nu.fired_this_step = true;
      nu.last_fire_step = step_;
      nu.potential = 0.0f;
    }
    nu.fire_rate_ema =
        nu.fire_rate_ema * (1.0f - cfg_.fire_rate_alpha) +
        (nu.fired_this_step ? 1.0f : 0.0f) * cfg_.fire_rate_alpha;

    // BCM sliding threshold: a much slower EMA than fire_rate_ema. This
    // tracks the neuron's long-term activity level and serves as the
    // post-synaptic potentiation threshold during stdp_phase.
    nu.activity_baseline =
        nu.activity_baseline * (1.0f - cfg_.bcm_baseline_alpha) +
        nu.fire_rate_ema * cfg_.bcm_baseline_alpha;

    // Reset the per-step LTP accumulator that heterosynaptic_phase will
    // read.
    nu.ltp_received_this_step = 0.0f;
  }
  // Spike count for this step (post-parallel reduction).
  int spikes = 0;
  for (const Neuron& nu : neurons_) if (nu.fired_this_step) ++spikes;
  last_stats_.spikes = spikes;
}

float Simulator::developmental_factor() const noexcept {
  if (cfg_.sensitive_period_tau <= 0) return 1.0f;
  const float age = static_cast<float>(step_) /
                    static_cast<float>(cfg_.sensitive_period_tau);
  // Boost decays from `sensitive_period_boost` at step 0 to 1.0 at infty.
  return 1.0f + (cfg_.sensitive_period_boost - 1.0f) * std::exp(-age);
}

void Simulator::stdp_phase() {
  // Spike-timing-dependent plasticity. The biological mechanism: when an
  // action potential reaches the post-synaptic neuron just before it
  // fires, NMDA receptors detect the coincidence (Mg2+ block lifted by
  // post-synaptic depolarisation) and admit Ca2+, which triggers CaMKII
  // and eventually inserts more AMPA receptors -- the synapse strengthens
  // (LTP). When the spike arrives just after the post fires, the same
  // pathway is recruited but in reverse, removing AMPA receptors (LTD).
  //
  // We approximate that with the standard exponential STDP kernels, using
  // strictly local information at every synapse:
  //   LTP : post.fired_this_step && this synapse delivered within window
  //         dt = step - syn.last_delivery_step in (0, W]
  //         dw  = +A_ltp * exp(-dt / tau)
  //   LTD : the post fired earlier and this synapse is delivering only
  //         now, so its action potential is too late. Handled in the
  //         scheduler dispatch where delivery actually happens.
  //
  // The eligibility trace used by reward learning is also kicked here on
  // any successful LTP event -- the same Ca2+/CaMKII signal that sustains
  // LTP is what tags a synapse as eligible for late dopamine consolidation.
  const int W = cfg_.stdp_window;
  const float tau = cfg_.stdp_tau;
  const float dev = developmental_factor();
  const float ach = cfg_.acetylcholine_level;
  const float nor = cfg_.norepinephrine_level;
  const float a_ltp = cfg_.stdp_a_ltp * dev * ach;
  const int nn = static_cast<int>(neurons_.size());
  // We can no longer parallelise this loop over pre because we now write
  // post.ltp_received_this_step (heterosynaptic damping). To keep the
  // per-synapse update local while remaining correct without atomics, we
  // switch to a serial sweep here. STDP cost is small relative to the
  // queue dispatch passes.
  for (int i = 0; i < nn; ++i) {
    Neuron& pre = neurons_[i];
    for (auto& syn : pre.outgoing) {
      // Continuous decay of the eligibility trace and the consolidation
      // tag happens every step regardless of events.
      syn.eligibility *= cfg_.eligibility_decay;
      syn.consolidation_tag *= cfg_.tag_decay;

      if (syn.target_neuron == 0 || syn.target_neuron > neurons_.size())
        continue;
      Neuron& post = neurons_[syn.target_neuron - 1];
      if (!post.fired_this_step) continue;
      const int dt = step_ - syn.last_delivery_step;
      if (dt <= 0 || dt > W) continue;

      const float kernel = std::exp(-static_cast<float>(dt) / tau);

      // Astrocyte calcium gating: high local Ca2+ in the synapse's
      // region (recent synaptic activity reported by neighbouring
      // glia) up-regulates LTP. Default modulation is 0 -> no effect.
      float astro_gain = 1.0f;
      if (cfg_.astrocyte_modulation > 0.0f && !astrocyte_ca_.empty()) {
        const int R = energy_.region_size();
        const std::size_t idx = std::size_t(syn.pos.x / R) +
                                 std::size_t(syn.pos.y / R) *
                                     energy_.rX() +
                                 std::size_t(syn.pos.z / R) *
                                     energy_.rX() * energy_.rY();
        if (idx < astrocyte_ca_.size()) {
          astro_gain += cfg_.astrocyte_modulation * astrocyte_ca_[idx];
        }
      }

      // BCM modulation of LTP amplitude: when the post is firing well
      // above its own baseline, LTP shrinks toward zero; below baseline
      // it stays near the unmodulated rate. The modulation is a smooth
      // multiplier so we never accidentally invert the sign of an LTP
      // event on a low-firing neuron. Norepinephrine effectively raises
      // the apparent baseline (denominator) -> easier LTP under arousal.
      const float bcm_threshold =
          std::max(0.005f, post.activity_baseline + nor * 0.04f);
      const float bcm_factor =
          std::max(0.05f,
                   std::min(1.5f,
                            1.0f - 0.5f * post.fire_rate_ema / bcm_threshold));
      const float dw = a_ltp * kernel * bcm_factor * astro_gain;

      syn.weight += dw;
      if (syn.weight > cfg_.weight_max) syn.weight = cfg_.weight_max;
      if (syn.weight < 0.0f) syn.weight = 0.0f;

      // Accumulate post-side LTP for heterosynaptic damping.
      if (dw > 0.0f) post.ltp_received_this_step += dw;

      // Synaptic tag: large LTP events tag the synapse for long-term
      // consolidation. The tag protects the spine from retraction.
      if (dw > cfg_.tag_threshold) {
        syn.consolidation_tag = std::min(
            1.0f, syn.consolidation_tag + dw * 4.0f);
      }

      ++syn.caused_fire_count;
      syn.eligibility +=
          cfg_.eligibility_potentiation * kernel * (1.0f + post.fire_rate_ema);
      // Pack ZZ v3: a delivery that demonstrably caused the post to
      // fire (LTP just applied) resets the eat-me tag to zero. The
      // synapse is doing what cortex hired it to do; microglia leave
      // it alone.
      syn.eat_me_tag = 0.0f;
    }
  }
}

void Simulator::heterosynaptic_phase() {
  // For every synapse, shrink the weight by a fraction of the post's
  // total received LTP this step (excluding this synapse's own
  // contribution). Because post-synaptic density resources are limited,
  // strong potentiation at one site costs every other incoming synapse.
  const float damp = cfg_.heterosynaptic_damp;
  if (damp <= 0.0f) return;
  const int nn = static_cast<int>(neurons_.size());
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nn; ++i) {
    for (auto& syn : neurons_[i].outgoing) {
      if (syn.target_neuron == 0 || syn.target_neuron > neurons_.size())
        continue;
      const Neuron& post = neurons_[syn.target_neuron - 1];
      const float total = post.ltp_received_this_step;
      if (total <= 0.0f) continue;
      // Permanent (engram) synapses are exempt from heterosynaptic
      // damping: their tag-and-capture state holds the post-synaptic
      // density resource against competition from later-arriving
      // potentiation events.
      if (syn.permanent) continue;
      // Each synapse pays a fixed share of the total LTP. Strong synapses
      // can absorb the cost; weak synapses get pushed lower and may
      // eventually hit the spine retraction floor.
      syn.weight -= damp * total / std::max(1.0f, post.incoming_weight_sum);
      if (syn.weight < 0.0f) syn.weight = 0.0f;
    }
  }
}

void Simulator::homeostatic_phase() {
  // Synaptic scaling. Each post neuron sums its incoming weight; if the
  // total exceeds (or undershoots) the target the neuron emits a
  // multiplicative correction that every pre synapse independently reads
  // and applies. Biologically: TNF-alpha / BDNF retrograde signalling
  // rebalances AMPA receptor density across all synapses on a cell to
  // keep the cell's mean firing rate near its set point (Turrigiano).
  for (Neuron& nu : neurons_) nu.incoming_weight_sum = 0.0f;
  for (Neuron& pre : neurons_) {
    for (const auto& syn : pre.outgoing) {
      if (syn.target_neuron > 0 && syn.target_neuron <= neurons_.size()) {
        neurons_[syn.target_neuron - 1].incoming_weight_sum += syn.weight;
      }
    }
  }
  const float target = cfg_.homeostatic_target_in;
  const float rate = cfg_.homeostatic_rate;
  const int nn = static_cast<int>(neurons_.size());
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nn; ++i) {
    for (auto& syn : neurons_[i].outgoing) {
      if (syn.target_neuron == 0 || syn.target_neuron > neurons_.size())
        continue;
      const float total = neurons_[syn.target_neuron - 1].incoming_weight_sum;
      if (total <= 1e-6f) continue;
      // Multiplicative correction toward the target. When total is low
      // the synapse is scaled up; when total is high it is scaled down.
      const float correction = 1.0f + rate * (target - total) / target;
      // Permanent (engram) synapses are exempt from down-scaling: real
      // tagged-and-captured spines maintain their AMPA receptor count
      // against the cell-wide retrograde scaling signal. They are still
      // allowed to scale *up* when total drops below target.
      if (syn.permanent && correction < 1.0f) continue;
      syn.weight *= correction;
      if (syn.weight > cfg_.weight_max) syn.weight = cfg_.weight_max;
      if (syn.weight < 0.0f) syn.weight = 0.0f;
    }
  }
}

void Simulator::fire_dispatch_phase() {
  // Pack P-lite: when a neuron fires, push one DeliveryEvent per
  // outgoing synapse into the central ring buffer slot at
  // `(step + conduction_delay) % ring_size`. The next time the
  // simulator reaches that step, `event_dispatch_phase` pulls the
  // events and processes them. Replaces the per-synapse transit
  // queue: work is now proportional to spike traffic.
  //
  // Forwarding is deterministic given local state: when the soma's
  // region energy is below `forward_min_energy`, only synapses whose
  // own weight is at least `forward_low_energy_floor` enqueue --
  // ATP-starved axons still transmit through their strongest boutons.
  for (Neuron& nu : neurons_) {
    if (!nu.fired_this_step) continue;
    const float soma_e = energy_.energy_at(nu.soma.x, nu.soma.y, nu.soma.z);
    const bool starved = soma_e < cfg_.forward_min_energy;
    // All three inhibitory subtypes (PV, SST, VIP) flip the spike sign
    // -- they all release GABA. Downstream effect differs by where
    // their synapses land (perisomatic vs dendritic) but the polarity
    // flip on dispatch is uniform.
    const bool inhibitory =
        (nu.polarity == NeuronPolarity::INHIBITORY ||
         nu.polarity == NeuronPolarity::INHIBITORY_SST ||
         nu.polarity == NeuronPolarity::INHIBITORY_VIP);
    const float sign = inhibitory ? -1.0f : 1.0f;
    const uint32_t pre_id = nu.id;
    for (uint32_t i = 0; i < nu.outgoing.size(); ++i) {
      const SynapseEdge& syn = nu.outgoing[i];
      if (starved && syn.weight < cfg_.forward_low_energy_floor) continue;
      const int delay = std::max(1, syn.conduction_delay);
      const int slot = (step_ + delay) % kDeliveryRingSize;
      delivery_ring_[slot].push_back({
          pre_id, i, syn.target_neuron, syn.branch,
          sign * syn.weight, syn.pos});
    }
  }
}

void Simulator::event_dispatch_phase() {
  // Pack P-lite v2 (Option B, hybrid event-driven dispatch with parallel
  // workers). Pulls all DeliveryEvents whose conduction-delay matures
  // at the current step from the central ring buffer and processes
  // them in parallel via OpenMP. Each event carries everything the
  // worker needs (pre id, syn index, post id, branch, signed magnitude,
  // synapse voxel, pre-rolled release-probability random) -- no scan
  // over inactive synapses, no shared rng during dispatch. Work is
  // proportional to spike traffic.
  //
  // Determinism: events are placed in the slot at fire-dispatch time
  // (sequential), and within a slot every (pre_id, syn_idx) pair is
  // unique, so each event writes its own SynapseEdge fields without
  // contention. Cross-event writes -- post.branch_potential, energy,
  // astrocyte_ca -- use #pragma omp atomic to avoid races. The
  // `release_probability` random is pre-rolled at fire-dispatch.
  //
  // STP recovery still runs as a BSP scan over all synapses (slow
  // timescale; fine to keep BSP). STDP-LTP fires inside `stdp_phase`
  // separately when the post fires within the window after delivery.
  const bool stp_active =
      cfg_.release_depression > 0.0f || cfg_.release_recovery > 0.0f;
  if (cfg_.release_recovery > 0.0f) {
    const int nn = static_cast<int>(neurons_.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nn; ++i) {
      for (auto& syn : neurons_[i].outgoing) {
        syn.vesicle_state = std::min(
            1.0f, syn.vesicle_state + cfg_.release_recovery);
      }
    }
  }

  const int slot = step_ % kDeliveryRingSize;
  auto& events = delivery_ring_[slot];
  const int n_events = static_cast<int>(events.size());

  // Sequential pre-pass: validation + RNG roll + branch_potential
  // resize + per-target bucketing. This preserves the v1 RNG
  // consumption pattern (one roll per validated event, in event
  // order) so the parallel dispatch is numerically identical to the
  // single-threaded v1 reference. Bucketing partitions surviving
  // events by `target_neuron % N` so each parallel worker owns a
  // disjoint set of post-synaptic neurons -- writes to
  // `branch_potential[b]` never collide across buckets. Within a
  // bucket the events are processed in original order, so float
  // accumulation is deterministic.
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  const bool stochastic =
      cfg_.release_probability > 0.0f && cfg_.release_probability < 1.0f;
  const int N = std::max(1, cfg_.event_dispatch_buckets);
  std::vector<std::vector<int>> bucket(N);
  for (int idx = 0; idx < n_events; ++idx) {
    const DeliveryEvent& ev = events[idx];
    if (ev.pre_id == 0 || ev.pre_id > neurons_.size()) continue;
    Neuron& pre = neurons_[ev.pre_id - 1];
    if (ev.syn_idx >= pre.outgoing.size()) continue;
    SynapseEdge& syn = pre.outgoing[ev.syn_idx];
    if (syn.target_neuron != ev.target_neuron) continue;
    if (stochastic && u(rng_) >= cfg_.release_probability) continue;
    if (ev.target_neuron == 0 || ev.target_neuron > neurons_.size()) continue;
    Neuron& post = neurons_[ev.target_neuron - 1];
    if (post.branch_potential.size() < post.n_branches) {
      post.branch_potential.assign(post.n_branches, 0.0f);
    }
    bucket[ev.target_neuron % N].push_back(idx);
  }

  // Parallel for over disjoint per-target buckets. Cross-bucket writes
  // (energy, astrocyte) use atomic updates -- those collide rarely
  // and the atomic cost is negligible.
#pragma omp parallel for schedule(static) if (N > 1)
  for (int b_id = 0; b_id < N; ++b_id) {
   for (int idx : bucket[b_id]) {
    const DeliveryEvent& ev = events[idx];
    Neuron& pre = neurons_[ev.pre_id - 1];
    SynapseEdge& syn = pre.outgoing[ev.syn_idx];
    Neuron& post = neurons_[ev.target_neuron - 1];
    uint8_t b = ev.branch;
    if (b >= post.n_branches) b = 0;
    const float effective = stp_active ? ev.magnitude * syn.vesicle_state
                                        : ev.magnitude;
    // Cross-event write: multiple events may target the same post's
    // same branch. Use atomic update.
    post.branch_potential[b] += effective;
    if (cfg_.release_depression > 0.0f) {
      syn.vesicle_state -= cfg_.release_depression;
      if (syn.vesicle_state < 0.0f) syn.vesicle_state = 0.0f;
    }
    if (cfg_.astrocyte_release_increment > 0.0f && !astrocyte_ca_.empty()) {
      const int R = energy_.region_size();
      const std::size_t aidx = std::size_t(ev.pos.x / R) +
                                std::size_t(ev.pos.y / R) * energy_.rX() +
                                std::size_t(ev.pos.z / R) *
                                    energy_.rX() * energy_.rY();
      if (aidx < astrocyte_ca_.size()) {
#pragma omp atomic
        astrocyte_ca_[aidx] += cfg_.astrocyte_release_increment;
      }
    }
    if (!syn.permanent) {
      const int dt = step_ - post.last_fire_step;
      if (dt > 0 && dt <= cfg_.stdp_window) {
        const float kernel = std::exp(-static_cast<float>(dt) / cfg_.stdp_tau);
        syn.weight -= cfg_.stdp_a_ltd * kernel;
        if (syn.weight < 0.0f) syn.weight = 0.0f;
      }
    }
    syn.last_delivery_step = step_;
    syn.last_active_step = step_;
    ++syn.delivered_count;
    // Pack ZZ v3: a delivery that arrived AFTER the post had already
    // fired (the STDP-LTD path above just ran) is "useless" at the
    // moment of arrival. Bump the eat-me tag. Useful deliveries
    // (those that cause an LTP event in stdp_phase next iteration)
    // will reset the tag to zero. Per-target bucketing makes this
    // per-syn write race-free across threads.
    if (!syn.permanent) syn.eat_me_tag += cfg_.microglia_tag_growth;
    // Energy field: cross-event writes possible if events fall in the
    // same region. Atomic decrement; the < 0 clamp from the original
    // serial code is dropped here because the read-after-write pattern
    // is racy under concurrent updates -- energy_regen_phase next step
    // normalises any small negative excursion.
    float& e = energy_.energy_at(ev.pos.x, ev.pos.y, ev.pos.z);
#pragma omp atomic
    e -= cfg_.synapse_use_cost;
    // The < 0 clamp is benign-racy under parallel buckets: at worst
    // one extra clamp gets applied; the operation is idempotent so
    // the final value is bounded. Preserves v1's energy semantics
    // (forward-starvation gate sees clamped values) so default N=1
    // is exactly numerically equivalent to single-threaded v1.
    if (e < 0.0f) e = 0.0f;
   }
  }
  events.clear();
}

void Simulator::pruning_phase() {
  // Spine retraction. Real cortical spines disassemble when their
  // postsynaptic density (PSD-95, AMPA receptors, actin scaffold) drops
  // below a maintenance threshold -- a cytoskeletal phase change, not a
  // probabilistic decision. In our model the analogous condition is
  // `weight < spine_retraction_floor`: STDP and homeostatic scaling have
  // driven this synapse below the structural maintenance level, and the
  // contact voxel demotes back to NEURON tissue (still owned by the post).
  //
  // A separate silence cutoff catches synapses that have weight above the
  // floor but have not fired for `prune_inactive_steps` -- representing
  // the much slower microglial elimination of silent synapses.
  for (Neuron& pre : neurons_) {
    auto& edges = pre.outgoing;
    const auto before = edges.size();
    edges.erase(
        std::remove_if(edges.begin(), edges.end(),
            [&](const SynapseEdge& syn) {
              // Permanently-marked innate synapses are always spared.
              // Real cortex protects labelled-line / reflex-arc wiring
              // from microglial pruning regardless of activity history.
              if (syn.permanent) return false;
              const bool retracted = syn.weight < cfg_.spine_retraction_floor;
              const bool ancient = (step_ - syn.last_active_step) >
                                   cfg_.prune_inactive_steps;
              if (!retracted && !ancient) return false;
              // Tag-and-capture protection: a synapse with a strong
              // consolidation tag has captured plasticity-related
              // proteins and is structurally stabilised. It survives a
              // weight dip that would otherwise retract its spine.
              if (syn.consolidation_tag >= cfg_.tag_protection) return false;
              if (grid_.in_bounds(syn.pos.x, syn.pos.y, syn.pos.z) &&
                  grid_.get(syn.pos.x, syn.pos.y, syn.pos.z) ==
                      BrainGrid::SYNAPSE) {
                grid_.set(syn.pos.x, syn.pos.y, syn.pos.z, BrainGrid::NEURON);
              }
              return true;
            }),
        edges.end());
    last_stats_.synapses_pruned += static_cast<int>(before - edges.size());
  }
}

void Simulator::microglia_phase() {
  // Pack ZZ v3 -- complement-tagged microglial elimination of surplus
  // synapses (Xing et al. 2026 NRR; Schafer & Stevens 2013). Eat-me
  // tags grow inline in event_dispatch_phase on every useless
  // delivery and are reset by stdp_phase on every LTP event.
  // Permanent / engram synapses carry `dont_eat_me = 1e9` and are
  // never touched.
  //
  // Conservative-by-design: removal requires SIMULTANEOUSLY
  //   eat_me_tag        > microglia_eat_threshold
  //   weight            < microglia_weak_weight
  //   consolidation_tag < microglia_tag_protection_max
  //   permanent         == false
  // This mirrors how real microglia preferentially prune the *weakest
  // among redundant* connections, not last-resort silent ones. The
  // warm-up window suppresses pruning during early development when
  // synapses haven't had time to demonstrate usefulness; the
  // per-region cap keeps removals localised so they cannot outpace
  // sprouting globally.
  if (step_ < cfg_.microglia_warmup_steps) return;
  if (cfg_.microglia_pass_period <= 0) return;
  if (step_ % cfg_.microglia_pass_period != 0) return;

  const int rX = energy_.rX();
  const int rY = energy_.rY();
  const int rZ = energy_.rZ();
  const int R  = cfg_.region_size;
  std::vector<int> region_remove(static_cast<std::size_t>(rX) * rY * rZ, 0);

  for (Neuron& pre : neurons_) {
    auto& edges = pre.outgoing;
    const auto before = edges.size();
    edges.erase(
        std::remove_if(
            edges.begin(), edges.end(),
            [&](const SynapseEdge& syn) {
              if (syn.permanent) return false;
              if (syn.dont_eat_me >= cfg_.microglia_eat_threshold) return false;
              // Silence-age criterion: real microglia preferentially
              // engulf synapses that have stayed silent for an
              // extended period (Schafer & Stevens 2013). The
              // existing `pruning_phase` only catches synapses below
              // `spine_retraction_floor` (0.02); microglia handle the
              // weight band 0.02 < w < weak_weight that is too strong
              // for spine retraction but functionally redundant.
              // Never-delivered synapses (last_delivery_step == -1)
              // are allowed if they're old enough to have had a
              // chance: (step_ - syn.last_active_step) doubles as
              // "age since last activity"; a never-active synapse
              // has last_active_step == -1 so the diff is huge and
              // it qualifies as silent.
              const int last = syn.last_delivery_step >= 0
                                   ? syn.last_delivery_step
                                   : syn.last_active_step;
              if (last >= 0 && step_ - last < cfg_.microglia_silence_steps)
                return false;
              if (syn.weight     >= cfg_.microglia_weak_weight)   return false;
              if (syn.consolidation_tag >=
                  cfg_.microglia_tag_protection_max)              return false;
              // Never prune the readout path. Real cortex preserves
              // motor-area projections during developmental pruning
              // (Stiles & Jernigan 2010 *Neuropsych. Rev.*).
              if (syn.target_neuron > 0 &&
                  syn.target_neuron <= neurons_.size()) {
                const Neuron& post = neurons_[syn.target_neuron - 1];
                if (post.role == NeuronRole::OUTPUT) return false;
              }
              const int rx = std::min(rX - 1,
                                       std::max(0, syn.pos.x / R));
              const int ry = std::min(rY - 1,
                                       std::max(0, syn.pos.y / R));
              const int rz = std::min(rZ - 1,
                                       std::max(0, syn.pos.z / R));
              const int idx = rx + ry * rX + rz * rX * rY;
              if (region_remove[idx] >=
                  cfg_.microglia_max_remove_per_region_per_pass)
                return false;
              if (grid_.in_bounds(syn.pos.x, syn.pos.y, syn.pos.z) &&
                  grid_.get(syn.pos.x, syn.pos.y, syn.pos.z) ==
                      BrainGrid::SYNAPSE) {
                grid_.set(syn.pos.x, syn.pos.y, syn.pos.z,
                          BrainGrid::NEURON);
              }
              ++region_remove[idx];
              return true;
            }),
        edges.end());
    last_stats_.synapses_pruned += static_cast<int>(before - edges.size());
  }
}

void Simulator::energy_regen_phase() {
  energy_.regenerate(cfg_.energy_regen);

  // Pay firing cost out of the soma's region. If the region cannot afford it,
  // the budget hits zero and growth gates effectively close until the region
  // has time to recover. This is the metabolic throttle the user described.
  for (Neuron& nu : neurons_) {
    if (!nu.fired_this_step) continue;
    float& e = energy_.energy_at(nu.soma.x, nu.soma.y, nu.soma.z);
    e -= cfg_.fire_cost;
    if (e < 0.0f) e = 0.0f;
  }

  // Astrocyte calcium decay. Each region's Ca2+ accumulator drifts
  // toward zero between releases, modelling astrocyte Ca2+ buffering.
  if (cfg_.astrocyte_decay < 1.0f && !astrocyte_ca_.empty()) {
    for (auto& v : astrocyte_ca_) v *= cfg_.astrocyte_decay;
  }
}

void Simulator::sprouting_phase() {
  // Activity-driven sprouting. For each active neuron we attempt a few random
  // extensions from existing body voxels into empty neighbours. Each attempt
  // must satisfy:
  //   - target voxel is EMPTY (volume exclusion implicit)
  //   - target voxel does not have too many occupied neighbours (explicit
  //     volume exclusion: dense areas are saturated)
  //   - local region has at least `sprout_cost` energy
  //   - probability gate scaled by recent firing rate and energy headroom
  //
  // This phase mutates grid_, owner_ and Neuron::body. It runs serially.
  std::uniform_real_distribution<float> uni(0.0f, 1.0f);
  std::uniform_int_distribution<int> nbr(0, 5);

  for (Neuron& nu : neurons_) {
    const float gate = nu.fire_rate_ema;
    if (gate <= 0.0f && !nu.fired_this_step) continue;
    if (nu.body.empty()) continue;

    for (int a = 0; a < cfg_.sprout_attempts; ++a) {
      std::uniform_int_distribution<int> pick(
          0, static_cast<int>(nu.body.size()) - 1);
      const Voxel& src = nu.body[pick(rng_)];

      const int* d = kNeighbours[nbr(rng_)];
      const int nx = src.x + d[0];
      const int ny = src.y + d[1];
      const int nz = src.z + d[2];
      if (!grid_.in_bounds(nx, ny, nz)) continue;
      if (grid_.get(nx, ny, nz) != BrainGrid::EMPTY) continue;

      // Volume exclusion: count surrounding non-empty cells. SYNAPSE and
      // NEURON both count as "occupied tissue"; BLOCKED also blocks growth.
      int crowd = 0;
      for (int k = 0; k < 6; ++k) {
        const int xx = nx + kNeighbours[k][0];
        const int yy = ny + kNeighbours[k][1];
        const int zz = nz + kNeighbours[k][2];
        if (!grid_.in_bounds(xx, yy, zz)) continue;
        const auto c = grid_.get(xx, yy, zz);
        if (c != BrainGrid::EMPTY) ++crowd;
      }
      if (crowd >= cfg_.max_neighbors_for_sprout) continue;

      float& e = energy_.energy_at(nx, ny, nz);
      if (e < cfg_.sprout_cost) continue;

      const float p = cfg_.sprout_prob *
                      clamp01(gate + (nu.fired_this_step ? 0.5f : 0.0f)) *
                      clamp01(e / cfg_.energy_max);
      if (uni(rng_) >= p) continue;

      grid_.set(nx, ny, nz, BrainGrid::NEURON);
      owner_[lin(nx, ny, nz)] = nu.id;
      // Phase 1 morphology refactor: sprouted voxels default to
      // DENDRITE (most cortical sprouting at this scale is dendritic
      // arborisation; axonal trunks are stamped explicitly by
      // `stamp_morphology`). Dendrites can receive synapses but not
      // initiate them, matching real chemical-synapse asymmetry.
      voxel_role_[lin(nx, ny, nz)] = ROLE_DENDRITE;
      nu.body.push_back({static_cast<int16_t>(nx),
                          static_cast<int16_t>(ny),
                          static_cast<int16_t>(nz)});
      e -= cfg_.sprout_cost;
      ++last_stats_.sprouts;
    }
  }
}

void Simulator::synaptogenesis_phase() {
  // For every body voxel of an active neuron, look at 6-neighbours. If a
  // neighbour is owned by a different neuron and the voxel is in the NEURON
  // (not BLOCKED, not already SYNAPSE) state, the contact may convert to a
  // SYNAPSE under the per-region cap and energy gate. The synapse becomes an
  // outgoing edge from the *active* neuron to the neighbour.
  std::uniform_real_distribution<float> uni(0.0f, 1.0f);

  const int rX = energy_.rX();
  const int rY = energy_.rY();
  const int rZ = energy_.rZ();
  std::vector<int> region_count(static_cast<std::size_t>(rX) * rY * rZ, 0);
  const int R = energy_.region_size();
  auto region_idx = [&](int x, int y, int z) {
    return (x / R) + (y / R) * rX + (z / R) * rX * rY;
  };

  // Census of existing synapses so the per-region cap stays meaningful as
  // synapses accumulate over the run.
  for (const Neuron& nu : neurons_) {
    for (const auto& syn : nu.outgoing) {
      ++region_count[region_idx(syn.pos.x, syn.pos.y, syn.pos.z)];
    }
  }

  for (Neuron& pre : neurons_) {
    if (!pre.fired_this_step && pre.fire_rate_ema < 0.05f) continue;

    for (const Voxel& v : pre.body) {
      // Phase 1 morphology refactor: only AXON voxels of pre can
      // initiate a synaptic contact. Dendrite-side voxels of pre
      // (the default for soma + sprouted body) cannot send signals
      // outward -- they're receivers. This kills random NEURON x
      // NEURON contacts that previously formed spurious synapses.
      const std::size_t v_idx = lin(v.x, v.y, v.z);
      if (voxel_role_[v_idx] != ROLE_AXON) continue;

      for (int k = 0; k < 6; ++k) {
        const int nx = v.x + kNeighbours[k][0];
        const int ny = v.y + kNeighbours[k][1];
        const int nz = v.z + kNeighbours[k][2];
        if (!grid_.in_bounds(nx, ny, nz)) continue;

        const auto c = grid_.get(nx, ny, nz);
        if (c != BrainGrid::NEURON) continue;  // BLOCKED / SYNAPSE / EMPTY skip

        const uint32_t other = owner_[lin(nx, ny, nz)];
        if (other == 0 || other == pre.id) continue;
        // Post-side voxel must be DENDRITE -- axon-axon contacts
        // don't form chemical synapses in real cortex.
        if (voxel_role_[lin(nx, ny, nz)] != ROLE_DENDRITE) continue;

        const int ridx = region_idx(nx, ny, nz);
        if (region_count[ridx] >= cfg_.max_synapses_per_region) continue;

        float& e = energy_.energy_at(nx, ny, nz);
        if (e < cfg_.synapse_form_cost) continue;

        if (uni(rng_) >= cfg_.synapse_form_prob) continue;

        grid_.set(nx, ny, nz, BrainGrid::SYNAPSE);
        e -= cfg_.synapse_form_cost;
        ++region_count[ridx];

        SynapseEdge edge;
        edge.target_neuron = other;
        edge.pos = {static_cast<int16_t>(nx),
                    static_cast<int16_t>(ny),
                    static_cast<int16_t>(nz)};
        edge.weight = cfg_.initial_weight;
        edge.last_active_step = step_;
        // Conduction delay equals the Manhattan distance from the
        // pre-synaptic soma to the contact voxel: the spike propagates one
        // voxel per step along the (already-existing) NEURON path.
        edge.conduction_delay = std::max(
            1,
            std::abs(static_cast<int>(pre.soma.x) - nx) +
                std::abs(static_cast<int>(pre.soma.y) - ny) +
                std::abs(static_cast<int>(pre.soma.z) - nz));
        // New synapses land on the configured default branch. For
        // multi-compartment posts, demos can flip
        // `synaptogenesis_default_branch` so sprouted plasticity targets
        // a different dendrite from hand-installed priors.
        Neuron& post_n = neurons_[other - 1];
        edge.branch = (post_n.n_branches > 0)
                          ? static_cast<uint8_t>(std::min<uint32_t>(
                                cfg_.synaptogenesis_default_branch,
                                post_n.n_branches - 1))
                          : 0;
        pre.outgoing.push_back(edge);
        ++last_stats_.synapses_formed;
      }
    }
  }
}

void Simulator::step() {
  // Per-step pipeline.
  //   1. queue 1 -> potential
  //   2. LIF + fire decision
  //   3. STDP-LTP for the firings just observed
  //   4. fire dispatch (queue 3 -> 4) under deterministic energy gate
  //   5. scheduler delivers in-flight spikes (queue 4 -> 1) and applies
  //      STDP-LTD to deliveries that arrived after their post had fired
  //   6. homeostatic synaptic scaling per post
  //   7. spine retraction: weights driven below the structural floor
  //      lose their physical contact site
  //   8. structural plasticity (sprouting / synaptogenesis) on the
  //      remaining living connectome
  last_stats_ = StepStats{};
  integrate_incoming_phase();
  chemistry_phase();
  stdp_phase();
  heterosynaptic_phase();
  fire_dispatch_phase();
  event_dispatch_phase();         // Pack P-lite: replaces scheduler
  homeostatic_phase();
  pruning_phase();
  microglia_phase();              // Pack ZZ v3
  energy_regen_phase();
  sprouting_phase();
  synaptogenesis_phase();
  ++step_;
}

std::size_t Simulator::total_synapses() const noexcept {
  std::size_t s = 0;
  for (const Neuron& nu : neurons_) s += nu.outgoing.size();
  return s;
}

std::size_t Simulator::total_neuron_voxels() const noexcept {
  std::size_t s = 0;
  for (const Neuron& nu : neurons_) s += nu.body.size();
  return s;
}

float Simulator::total_energy() const noexcept {
  float total = 0.0f;
  for (int z = 0; z < energy_.rZ(); ++z) {
    for (int y = 0; y < energy_.rY(); ++y) {
      for (int x = 0; x < energy_.rX(); ++x) {
        total += energy_.at(x, y, z);
      }
    }
  }
  return total;
}

void Simulator::sleep_consolidate(int n_steps, float boost) {
  // Replay-style consolidation. We temporarily boost STDP amplitudes,
  // disable external drive (the network only sees its own internal
  // noise + the activity it was already carrying), and run for n_steps.
  // Co-firing patterns that had been built up during the awake training
  // are reinforced; tagged synapses capture additional weight,
  // consolidating recent learning. Models the slow-wave / REM replay
  // observed in hippocampus and cortex.
  const float saved_a_ltp = cfg_.stdp_a_ltp;
  const float saved_a_ltd = cfg_.stdp_a_ltd;
  const float saved_ach = cfg_.acetylcholine_level;
  cfg_.stdp_a_ltp *= boost;
  cfg_.stdp_a_ltd *= 0.7f;          // slightly less LTD during replay
  cfg_.acetylcholine_level *= 0.5f; // less attention-driven plasticity

  std::uniform_real_distribution<float> small_noise(0.0f, 0.05f);
  for (int s = 0; s < n_steps; ++s) {
    // Tiny baseline noise on every neuron so spontaneous activity has
    // somewhere to start; the structural connectome and chemistry then
    // do the rest.
    for (Neuron& nu : neurons_) {
      nu.input_acc += small_noise(rng_);
    }
    step();
  }

  cfg_.stdp_a_ltp = saved_a_ltp;
  cfg_.stdp_a_ltd = saved_a_ltd;
  cfg_.acetylcholine_level = saved_ach;
}

void Simulator::sleep_sws_replay(
    const std::vector<std::vector<float>>& sequence, int n_features,
    int present_per_pattern, int gap_steps, float boost) {
  // Slow-wave-sleep replay: each pattern in `sequence` is presented in
  // order, like the hippocampus replaying a recent trajectory in its
  // original temporal sequence. STDP windows that depend on causal
  // ordering get coherent input; the network consolidates the
  // *order-dependent* aspects of the experience, not just the
  // co-firing statistics that random replay reinforces. Silence gaps
  // between patterns let dynamics decay so successive patterns don't
  // contaminate each other's plasticity.
  if (sequence.empty() || n_features <= 0) return;
  const float saved_a_ltp = cfg_.stdp_a_ltp;
  const float saved_a_ltd = cfg_.stdp_a_ltd;
  const float saved_ach = cfg_.acetylcholine_level;
  cfg_.stdp_a_ltp *= boost;
  cfg_.stdp_a_ltd *= 0.6f;        // SWS suppresses LTD a bit
  cfg_.acetylcholine_level *= 0.3f;  // low ACh in slow-wave sleep

  std::vector<float> zero(n_features, 0.0f);
  for (const auto& pat : sequence) {
    for (int s = 0; s < present_per_pattern; ++s) {
      apply_input_pattern(
          pat.data(),
          std::min<int>(n_features, static_cast<int>(pat.size())));
      step();
    }
    for (int s = 0; s < gap_steps; ++s) {
      apply_input_pattern(zero.data(), n_features);
      step();
    }
  }

  cfg_.stdp_a_ltp = saved_a_ltp;
  cfg_.stdp_a_ltd = saved_a_ltd;
  cfg_.acetylcholine_level = saved_ach;
}

void Simulator::sleep_rem_replay(
    int n_steps, const std::vector<std::vector<float>>& patterns,
    int n_features, float boost) {
  // REM-style replay: random fragments at each step with very high
  // STDP gain and elevated attention. Encourages the recombination
  // and unusual associations characteristic of dreams. Implementation
  // is the noise-augmented pattern replay from sleep_replay_patterns
  // but with stronger boost and an acetylcholine *increase* rather
  // than decrease (REM is a high-ACh state).
  if (patterns.empty() || n_features <= 0) {
    sleep_consolidate(n_steps, boost);
    return;
  }
  const float saved_a_ltp = cfg_.stdp_a_ltp;
  const float saved_a_ltd = cfg_.stdp_a_ltd;
  const float saved_ach = cfg_.acetylcholine_level;
  cfg_.stdp_a_ltp *= boost;
  cfg_.stdp_a_ltd *= 0.5f;
  cfg_.acetylcholine_level *= 1.4f;

  std::uniform_int_distribution<int> pick(
      0, static_cast<int>(patterns.size()) - 1);
  std::uniform_real_distribution<float> bigger_noise(0.0f, 0.08f);
  for (int s = 0; s < n_steps; ++s) {
    const auto& pat = patterns[pick(rng_)];
    apply_input_pattern(pat.data(),
                        std::min<int>(n_features, static_cast<int>(pat.size())));
    for (Neuron& nu : neurons_) {
      nu.input_acc += bigger_noise(rng_);
    }
    step();
  }

  cfg_.stdp_a_ltp = saved_a_ltp;
  cfg_.stdp_a_ltd = saved_a_ltd;
  cfg_.acetylcholine_level = saved_ach;
}

void Simulator::sleep_replay_patterns(
    int n_steps, const std::vector<std::vector<float>>& patterns,
    int n_features, float boost) {
  // Pattern-replay variant of sleep consolidation. Each step the network
  // is driven with a pattern drawn at random from the supplied set --
  // rehearsing experiences that the demo just lived through. STDP is
  // boosted as in plain consolidate. No reward is broadcast; the
  // already-tagged synapses use the replay activity to capture more
  // weight via standard STDP and tag-and-capture rules.
  if (patterns.empty() || n_features <= 0) {
    sleep_consolidate(n_steps, boost);
    return;
  }
  const float saved_a_ltp = cfg_.stdp_a_ltp;
  const float saved_a_ltd = cfg_.stdp_a_ltd;
  const float saved_ach = cfg_.acetylcholine_level;
  cfg_.stdp_a_ltp *= boost;
  cfg_.stdp_a_ltd *= 0.7f;
  cfg_.acetylcholine_level *= 0.5f;

  std::uniform_int_distribution<int> pick(
      0, static_cast<int>(patterns.size()) - 1);
  std::uniform_real_distribution<float> small_noise(0.0f, 0.04f);

  for (int s = 0; s < n_steps; ++s) {
    const auto& pat = patterns[pick(rng_)];
    apply_input_pattern(pat.data(),
                        std::min<int>(n_features, static_cast<int>(pat.size())));
    for (Neuron& nu : neurons_) {
      nu.input_acc += small_noise(rng_);
    }
    step();
  }

  cfg_.stdp_a_ltp = saved_a_ltp;
  cfg_.stdp_a_ltd = saved_a_ltd;
  cfg_.acetylcholine_level = saved_ach;
}

namespace {

// Encode a (bx, by, bz) bin triple into a single int64. Each axis takes
// 24 bits with two's-complement sign-extension on decode, which is
// plenty for any grid this code base is going to run on.
constexpr int64_t kBinAxisMask = 0xFFFFFFLL;
inline int64_t pos_bin_key(int bx, int by, int bz) {
  return (static_cast<int64_t>(bx) & kBinAxisMask) |
         ((static_cast<int64_t>(by) & kBinAxisMask) << 24) |
         ((static_cast<int64_t>(bz) & kBinAxisMask) << 48);
}
inline std::array<int, 3> pos_bin_decode(int64_t key) {
  auto sx = [](int64_t v) {
    int x = static_cast<int>(v & kBinAxisMask);
    if (x & 0x800000) x |= ~static_cast<int>(kBinAxisMask);
    return x;
  };
  return {sx(key), sx(key >> 24), sx(key >> 48)};
}

}  // namespace

std::array<int, 3> Simulator::position_bin_for(uint32_t neuron_id) const {
  if (neuron_id == 0 || neuron_id > neurons_.size()) return {0, 0, 0};
  const Neuron& nu = neurons_[neuron_id - 1];
  const int R = std::max(1, cfg_.region_size);
  return {nu.soma.x / R, nu.soma.y / R, nu.soma.z / R};
}

void Simulator::refresh_position_features() {
  position_features_.clear();
  const int R = std::max(1, cfg_.region_size);

  // Determine the highest INPUT channel in use so tuning_curve is
  // sized consistently across bins.
  int max_input_channel = -1;
  for (const Neuron& nu : neurons_) {
    if (nu.role == NeuronRole::INPUT && nu.channel > max_input_channel) {
      max_input_channel = nu.channel;
    }
  }
  const int n_channels =
      (max_input_channel >= 0) ? max_input_channel + 1 : 0;

  for (const Neuron& nu : neurons_) {
    const int64_t key = pos_bin_key(nu.soma.x / R, nu.soma.y / R,
                                     nu.soma.z / R);
    auto& pf = position_features_[key];
    pf.n_neurons += 1;
    pf.mean_fire_rate_ema += nu.fire_rate_ema;
    pf.mean_activity_baseline += nu.activity_baseline;
    pf.mean_incoming_weight += nu.incoming_weight_sum;
    if (n_channels > 0 &&
        pf.tuning_curve.size() < static_cast<std::size_t>(n_channels)) {
      pf.tuning_curve.resize(n_channels, 0.0f);
    }
  }

  // Walk every INPUT neuron's outgoing synapses and add the synapse
  // weight to the target bin's tuning_curve entry for that channel.
  // This snapshots the connectome's labelled-line mapping per bin --
  // bins primarily wired from channel C will report C as their
  // dominant tuning. O(neurons * avg_outgoing).
  if (n_channels > 0) {
    for (const Neuron& pre : neurons_) {
      if (pre.role != NeuronRole::INPUT) continue;
      if (pre.channel < 0 || pre.channel >= n_channels) continue;
      for (const SynapseEdge& syn : pre.outgoing) {
        if (syn.target_neuron == 0 ||
            syn.target_neuron > neurons_.size()) continue;
        const Neuron& post = neurons_[syn.target_neuron - 1];
        const int64_t key = pos_bin_key(
            post.soma.x / R, post.soma.y / R, post.soma.z / R);
        auto it = position_features_.find(key);
        if (it == position_features_.end()) continue;
        if (it->second.tuning_curve.size() <
            static_cast<std::size_t>(n_channels)) {
          it->second.tuning_curve.resize(n_channels, 0.0f);
        }
        it->second.tuning_curve[pre.channel] += syn.weight;
      }
    }
  }

  for (auto& kv : position_features_) {
    PositionFeatures& pf = kv.second;
    if (pf.n_neurons > 0) {
      const float n = static_cast<float>(pf.n_neurons);
      pf.mean_fire_rate_ema /= n;
      pf.mean_activity_baseline /= n;
      pf.mean_incoming_weight /= n;
    }
  }
}

const PositionFeatures* Simulator::position_features_at(
    int bx, int by, int bz) const {
  auto it = position_features_.find(pos_bin_key(bx, by, bz));
  if (it == position_features_.end()) return nullptr;
  return &it->second;
}

bool Simulator::dump_position_features_csv(const char* path) const {
  std::ofstream f(path);
  if (!f) return false;

  // Tuning-curve width: max length across bins so every row has the
  // same column count.
  std::size_t n_channels = 0;
  for (const auto& kv : position_features_) {
    n_channels = std::max(n_channels, kv.second.tuning_curve.size());
  }

  f << "bin_x,bin_y,bin_z,n_neurons,mean_fire_rate_ema,"
       "mean_activity_baseline,mean_incoming_weight";
  for (std::size_t i = 0; i < n_channels; ++i) f << ",tuning_" << i;
  f << "\n";
  for (const auto& kv : position_features_) {
    const auto bin = pos_bin_decode(kv.first);
    const PositionFeatures& pf = kv.second;
    f << bin[0] << ',' << bin[1] << ',' << bin[2] << ','
      << pf.n_neurons << ',' << pf.mean_fire_rate_ema << ','
      << pf.mean_activity_baseline << ',' << pf.mean_incoming_weight;
    for (std::size_t i = 0; i < n_channels; ++i) {
      const float v = (i < pf.tuning_curve.size())
                          ? pf.tuning_curve[i]
                          : 0.0f;
      f << ',' << v;
    }
    f << '\n';
  }
  return f.good();
}

void Simulator::apply_position_prior(Neuron& nu) {
  // Soft cortical-map prior: a newborn neuron inherits the running
  // mean BCM activity_baseline of its bin neighbours. Linear scan is
  // O(n_neurons); fine for the population sizes the demos run with.
  // No-op when the bin is empty (the neuron keeps its default baseline).
  const int R = std::max(1, cfg_.region_size);
  const int bx = nu.soma.x / R;
  const int by = nu.soma.y / R;
  const int bz = nu.soma.z / R;
  int n = 0;
  float sum_baseline = 0.0f;
  for (const Neuron& other : neurons_) {
    if (other.id == nu.id) continue;
    if (other.soma.x / R != bx) continue;
    if (other.soma.y / R != by) continue;
    if (other.soma.z / R != bz) continue;
    ++n;
    sum_baseline += other.activity_baseline;
  }
  if (n > 0) {
    nu.activity_baseline = sum_baseline / static_cast<float>(n);
  }
}

bool Simulator::dump_csv(const char* prefix) const {
  // Write structural snapshot to three companion CSV files: voxels (the
  // 2-bit grid filtered to non-EMPTY cells), neurons (id + role +
  // polarity + soma + body-voxel count + n_branches) and synapses
  // (pre, post, position, weight, branch, conduction delay, permanent).
  // The Python visualisation scripts under `scripts/` read these.
  const std::string p = prefix;
  std::ofstream vf(p + "_voxels.csv");
  std::ofstream nf(p + "_neurons.csv");
  std::ofstream sf(p + "_synapses.csv");
  if (!vf || !nf || !sf) return false;

  vf << "x,y,z,state\n";
  for (int z = 0; z < grid_.Z(); ++z) {
    for (int y = 0; y < grid_.Y(); ++y) {
      for (int x = 0; x < grid_.X(); ++x) {
        const auto c = grid_.get(x, y, z);
        if (c == BrainGrid::EMPTY) continue;
        vf << x << ',' << y << ',' << z << ',' << static_cast<int>(c) << '\n';
      }
    }
  }

  nf << "id,role,polarity,channel,soma_x,soma_y,soma_z,n_branches,"
        "body_voxels,fire_rate_ema\n";
  for (const Neuron& nu : neurons_) {
    nf << nu.id << ',' << static_cast<int>(nu.role) << ','
       << static_cast<int>(nu.polarity) << ',' << nu.channel << ','
       << static_cast<int>(nu.soma.x) << ','
       << static_cast<int>(nu.soma.y) << ','
       << static_cast<int>(nu.soma.z) << ','
       << static_cast<int>(nu.n_branches) << ',' << nu.body.size() << ','
       << nu.fire_rate_ema << '\n';
  }

  sf << "pre,post,pos_x,pos_y,pos_z,weight,branch,conduction_delay,"
        "permanent,consolidation_tag\n";
  for (const Neuron& nu : neurons_) {
    for (const auto& syn : nu.outgoing) {
      sf << nu.id << ',' << syn.target_neuron << ','
         << static_cast<int>(syn.pos.x) << ','
         << static_cast<int>(syn.pos.y) << ','
         << static_cast<int>(syn.pos.z) << ',' << syn.weight << ','
         << static_cast<int>(syn.branch) << ',' << syn.conduction_delay
         << ',' << (syn.permanent ? 1 : 0) << ','
         << syn.consolidation_tag << '\n';
    }
  }
  return true;
}

// ---- Sleep: persistence layer --------------------------------------------
//
// Binary file layout (little-endian, host-aligned):
//   magic[4]            = "SNC2"
//   uint32 version      = 2
//   SimConfig cfg       (POD)
//   int step, vz_lo, vz_hi
//   uint64 grid_words   then raw front-buffer uint64s
//   uint64 energy_cells then raw float array
//   uint64 owner_cells  then raw uint32 array
//   uint64 n_neurons
//   for each neuron:
//     id, soma(int16x3), role, polarity, channel,
//     potential, input_acc, fire_rate_ema,
//     last_fire_step, fired_this_step (bool)
//     incoming_weight_sum
//     uint64 body_n then body voxels
//     uint64 incoming_n then floats
//     uint64 outgoing_n then for each:
//       target_neuron, pos(int16x3), weight,
//       last_active_step, eligibility,
//       conduction_delay,
//       delivered_count, caused_fire_count, last_delivery_step,
//       uint64 transit_n then SpikePackets
//
// Writes on save, reads on load. Per-neuron computation independence is
// fully preserved: nothing in this routine changes the simulator's
// behavioural model.
namespace {

template <class T>
void wpod(std::ofstream& f, const T& v) {
  f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <class T>
void rpod(std::ifstream& f, T& v) {
  f.read(reinterpret_cast<char*>(&v), sizeof(T));
}
template <class T>
void wvec(std::ofstream& f, const std::vector<T>& v) {
  uint64_t n = v.size();
  wpod(f, n);
  if (n) f.write(reinterpret_cast<const char*>(v.data()), n * sizeof(T));
}
template <class T>
void rvec(std::ifstream& f, std::vector<T>& v) {
  uint64_t n;
  rpod(f, n);
  v.resize(static_cast<std::size_t>(n));
  if (n) f.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));
}

}  // namespace

bool Simulator::save_state(const char* path) const {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  const char magic[4] = {'S', 'N', 'C', 'B'};  // SNCB = SNC11
  f.write(magic, 4);
  uint32_t version = 11;
  wpod(f, version);
  wpod(f, cfg_);
  wpod(f, step_);
  wpod(f, vz_lo_);
  wpod(f, vz_hi_);

  wvec(f, grid_.raw_front());
  wvec(f, energy_.raw_data());
  wvec(f, owner_);
  wvec(f, voxel_role_);            // Phase 1 morphology refactor

  uint64_t nn = neurons_.size();
  wpod(f, nn);
  for (const Neuron& nu : neurons_) {
    wpod(f, nu.id);
    wpod(f, nu.soma);
    wpod(f, nu.role);
    wpod(f, nu.polarity);
    wpod(f, nu.channel);
    wpod(f, nu.potential);
    wpod(f, nu.input_acc);
    wpod(f, nu.fire_rate_ema);
    wpod(f, nu.last_fire_step);
    uint8_t fired = nu.fired_this_step ? 1 : 0;
    wpod(f, fired);
    wpod(f, nu.incoming_weight_sum);
    wpod(f, nu.activity_baseline);
    wpod(f, nu.n_branches);
    wpod(f, nu.predicted_input);
    wpod(f, nu.excitability_bias);
    wvec(f, nu.branch_potential);
    wvec(f, nu.branch_threshold);
    wvec(f, nu.branch_passive_gain);
    wvec(f, nu.body);
    wvec(f, nu.incoming_queue);

    uint64_t outn = nu.outgoing.size();
    wpod(f, outn);
    for (const SynapseEdge& syn : nu.outgoing) {
      wpod(f, syn.target_neuron);
      wpod(f, syn.pos);
      wpod(f, syn.weight);
      wpod(f, syn.last_active_step);
      wpod(f, syn.eligibility);
      wpod(f, syn.consolidation_tag);
      uint8_t perm = syn.permanent ? 1 : 0;
      wpod(f, perm);
      wpod(f, syn.conduction_delay);
      wpod(f, syn.branch);
      wpod(f, syn.vesicle_state);
      wpod(f, syn.delivered_count);
      wpod(f, syn.caused_fire_count);
      wpod(f, syn.last_delivery_step);
      wpod(f, syn.eat_me_tag);
      wpod(f, syn.dont_eat_me);
      wvec(f, syn.transit);
    }
  }
  // Per-class engram membership tables. Variable-length: outer count
  // followed by each class's vector.
  uint64_t n_classes = engram_members_.size();
  wpod(f, n_classes);
  for (const auto& v : engram_members_) wvec(f, v);
  uint64_t n_regions = class_regions_.size();
  wpod(f, n_regions);
  for (const auto& r : class_regions_) wpod(f, r);
  return f.good();
}

bool Simulator::load_state(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  char magic[4];
  f.read(magic, 4);
  if (std::memcmp(magic, "SNCB", 4) != 0) return false;
  uint32_t version;
  rpod(f, version);
  if (version != 11) return false;

  rpod(f, cfg_);
  rpod(f, step_);
  rpod(f, vz_lo_);
  rpod(f, vz_hi_);

  // Reconstruct the grid + energy field with the saved dims.
  grid_ = BrainGrid(cfg_.X, cfg_.Y, cfg_.Z);
  energy_ = EnergyField(cfg_.X, cfg_.Y, cfg_.Z, cfg_.region_size,
                        cfg_.energy_max);
  // Pack P-lite: the delivery ring is transient state; loaded brains
  // resume with no events in flight (a brief delivery-gap in the
  // first few steps is harmless and matches `reset_dynamics`).
  for (auto& slot : delivery_ring_) slot.clear();

  rvec(f, grid_.raw_front());
  rvec(f, energy_.raw_data());
  rvec(f, owner_);
  rvec(f, voxel_role_);            // Phase 1 morphology refactor

  uint64_t nn;
  rpod(f, nn);
  neurons_.clear();
  neurons_.resize(static_cast<std::size_t>(nn));
  for (Neuron& nu : neurons_) {
    rpod(f, nu.id);
    rpod(f, nu.soma);
    rpod(f, nu.role);
    rpod(f, nu.polarity);
    rpod(f, nu.channel);
    rpod(f, nu.potential);
    rpod(f, nu.input_acc);
    rpod(f, nu.fire_rate_ema);
    rpod(f, nu.last_fire_step);
    uint8_t fired;
    rpod(f, fired);
    nu.fired_this_step = fired != 0;
    rpod(f, nu.incoming_weight_sum);
    rpod(f, nu.activity_baseline);
    rpod(f, nu.n_branches);
    rpod(f, nu.predicted_input);
    rpod(f, nu.excitability_bias);
    rvec(f, nu.branch_potential);
    rvec(f, nu.branch_threshold);
    rvec(f, nu.branch_passive_gain);
    rvec(f, nu.body);
    rvec(f, nu.incoming_queue);

    uint64_t outn;
    rpod(f, outn);
    nu.outgoing.resize(static_cast<std::size_t>(outn));
    for (SynapseEdge& syn : nu.outgoing) {
      rpod(f, syn.target_neuron);
      rpod(f, syn.pos);
      rpod(f, syn.weight);
      rpod(f, syn.last_active_step);
      rpod(f, syn.eligibility);
      rpod(f, syn.consolidation_tag);
      uint8_t perm;
      rpod(f, perm);
      syn.permanent = perm != 0;
      rpod(f, syn.conduction_delay);
      rpod(f, syn.branch);
      rpod(f, syn.vesicle_state);
      rpod(f, syn.delivered_count);
      rpod(f, syn.caused_fire_count);
      rpod(f, syn.last_delivery_step);
      rpod(f, syn.eat_me_tag);
      rpod(f, syn.dont_eat_me);
      rvec(f, syn.transit);
    }
  }
  uint64_t n_classes = 0;
  rpod(f, n_classes);
  engram_members_.assign(static_cast<std::size_t>(n_classes),
                          std::vector<uint32_t>{});
  for (auto& v : engram_members_) rvec(f, v);
  uint64_t n_regions = 0;
  rpod(f, n_regions);
  class_regions_.assign(static_cast<std::size_t>(n_regions), ClassRegion{});
  for (auto& r : class_regions_) rpod(f, r);
  return f.good();
}

}  // namespace snc
