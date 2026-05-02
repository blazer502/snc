#include "simulator.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
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
  auto new_lin = [&](int x, int y, int z) {
    return static_cast<std::size_t>(x) +
           static_cast<std::size_t>(y) * newX +
           static_cast<std::size_t>(z) * newX * newY;
  };
  for (int z = 0; z < cfg_.Z; ++z) {
    for (int y = 0; y < cfg_.Y; ++y) {
      for (int x = 0; x < cfg_.X; ++x) {
        new_owner[new_lin(x + dx, y + dy, z + dz)] = owner_[lin(x, y, z)];
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

  cfg_.X = newX;
  cfg_.Y = newY;
  cfg_.Z = newZ;
  grid_ = std::move(new_grid);
  energy_ = std::move(new_energy);
  owner_ = std::move(new_owner);
  vz_lo_ += dz;
  vz_hi_ += dz;
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

  // 5. Energy gradient: high near the VZ (where neurogenesis burns glucose),
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
                                 float weight, int conduction_delay) {
  if (pre_id == 0 || pre_id > neurons_.size()) return;
  if (post_id == 0 || post_id > neurons_.size()) return;
  Neuron& pre = neurons_[pre_id - 1];
  SynapseEdge edge;
  edge.target_neuron = post_id;
  edge.pos = pre.soma;  // unused for non-grid synapses
  edge.weight = weight;
  edge.last_active_step = step_;
  edge.last_delivery_step = step_ - 10000;
  edge.conduction_delay = std::max(1, conduction_delay);
  pre.outgoing.push_back(edge);
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
  // is preserved.
  const float lr = cfg_.reward_lr;
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
  const float lr = cfg_.reward_lr;
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
      float w = syn.weight + lr * r * syn.eligibility;
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

void Simulator::integrate_incoming_phase() {
  // Stage 1->2 boundary: every spike that the scheduler delivered into
  // a neuron's `incoming_queue` last cycle is summed into `input_acc`,
  // joining any direct external injection. The queue is then cleared.
  // Strictly per-neuron and embarrassingly parallel.
  const int nn = static_cast<int>(neurons_.size());
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nn; ++i) {
    Neuron& nu = neurons_[i];
    float sum = 0.0f;
    for (float v : nu.incoming_queue) sum += v;
    nu.input_acc += sum;
    nu.incoming_queue.clear();
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
    if (nu.role == NeuronRole::INPUT) {
      nu.potential = nu.input_acc;
    } else {
      nu.potential = nu.potential * cfg_.potential_decay + nu.input_acc;
    }
    nu.input_acc = 0.0f;

    if (nu.potential >= cfg_.fire_threshold) {
      nu.fired_this_step = true;
      nu.last_fire_step = step_;
      nu.potential = 0.0f;
    }
    nu.fire_rate_ema =
        nu.fire_rate_ema * (1.0f - cfg_.fire_rate_alpha) +
        (nu.fired_this_step ? 1.0f : 0.0f) * cfg_.fire_rate_alpha;
  }
  // Spike count for this step (post-parallel reduction).
  int spikes = 0;
  for (const Neuron& nu : neurons_) if (nu.fired_this_step) ++spikes;
  last_stats_.spikes = spikes;
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
  const float a_ltp = cfg_.stdp_a_ltp;
  const int nn = static_cast<int>(neurons_.size());
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nn; ++i) {
    Neuron& pre = neurons_[i];
    for (auto& syn : pre.outgoing) {
      // Continuous decay of the eligibility trace happens every step
      // regardless of events (matches molecular trace decay).
      syn.eligibility *= cfg_.eligibility_decay;

      if (syn.target_neuron == 0 || syn.target_neuron > neurons_.size())
        continue;
      const Neuron& post = neurons_[syn.target_neuron - 1];
      if (!post.fired_this_step) continue;
      const int dt = step_ - syn.last_delivery_step;
      if (dt <= 0 || dt > W) continue;

      const float kernel = std::exp(-static_cast<float>(dt) / tau);
      syn.weight += a_ltp * kernel;
      if (syn.weight > cfg_.weight_max) syn.weight = cfg_.weight_max;

      ++syn.caused_fire_count;
      syn.eligibility +=
          cfg_.eligibility_potentiation * kernel * (1.0f + post.fire_rate_ema);
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
      syn.weight *= correction;
      if (syn.weight > cfg_.weight_max) syn.weight = cfg_.weight_max;
      if (syn.weight < 0.0f) syn.weight = 0.0f;
    }
  }
}

void Simulator::fire_dispatch_phase() {
  // Stage 3: when a neuron fires, push a SpikePacket onto each outgoing
  // synapse's transit queue. The forwarding decision is deterministic
  // given local state: when the soma's region energy is below
  // `forward_min_energy`, only synapses whose own weight is at least
  // `forward_low_energy_floor` get to enqueue. This mirrors the fact that
  // ATP-starved axons still successfully transmit through the strongest
  // synaptic boutons (large vesicle pools), while weaker boutons fail.
  for (Neuron& nu : neurons_) {
    if (!nu.fired_this_step) continue;
    const float soma_e = energy_.energy_at(nu.soma.x, nu.soma.y, nu.soma.z);
    const bool starved = soma_e < cfg_.forward_min_energy;
    const float sign =
        (nu.polarity == NeuronPolarity::INHIBITORY) ? -1.0f : 1.0f;
    for (auto& syn : nu.outgoing) {
      if (starved && syn.weight < cfg_.forward_low_energy_floor) continue;
      SpikePacket pk;
      pk.magnitude = sign * syn.weight;
      pk.delay_remaining = syn.conduction_delay;
      syn.transit.push_back(pk);
    }
  }
}

void Simulator::scheduler_dispatch_phase() {
  // Stage 4: the scheduler advances every in-flight spike by one voxel
  // step and delivers any whose conduction delay has elapsed. Delivery
  // pushes the magnitude into the post neuron's incoming queue (queue 1
  // for the next cycle), updates synapse use accounting and pays the
  // synapse-use energy cost from the synapse's local region.
  for (Neuron& pre : neurons_) {
    for (auto& syn : pre.outgoing) {
      if (syn.transit.empty()) continue;
      // Advance every packet by one step.
      for (auto& pk : syn.transit) --pk.delay_remaining;
      // Deliver any that have arrived.
      auto write = syn.transit.begin();
      for (auto read = syn.transit.begin(); read != syn.transit.end(); ++read) {
        if (read->delay_remaining > 0) {
          if (write != read) *write = *read;
          ++write;
          continue;
        }
        if (syn.target_neuron > 0 && syn.target_neuron <= neurons_.size()) {
          Neuron& post = neurons_[syn.target_neuron - 1];
          post.incoming_queue.push_back(read->magnitude);

          // LTD half of STDP: if the post fired *before* this delivery
          // arrived, the spike is too late to be causal. The same NMDA /
          // CaMKII / AMPA pathway that mediates LTP runs in reverse and
          // removes receptors, weakening the synapse. Strictly local --
          // we look only at the post's last_fire_step, which the synapse
          // already needs to know about its own target.
          const int dt = step_ - post.last_fire_step;
          if (dt > 0 && dt <= cfg_.stdp_window) {
            const float kernel =
                std::exp(-static_cast<float>(dt) / cfg_.stdp_tau);
            syn.weight -= cfg_.stdp_a_ltd * kernel;
            if (syn.weight < 0.0f) syn.weight = 0.0f;
          }
        }
        syn.last_delivery_step = step_;
        syn.last_active_step = step_;
        ++syn.delivered_count;
        // Synapse use depletes a tiny amount of local energy.
        float& e = energy_.energy_at(syn.pos.x, syn.pos.y, syn.pos.z);
        e -= cfg_.synapse_use_cost;
        if (e < 0.0f) e = 0.0f;
      }
      syn.transit.erase(write, syn.transit.end());
    }
  }
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
              const bool retracted = syn.weight < cfg_.spine_retraction_floor;
              const bool ancient = (step_ - syn.last_active_step) >
                                   cfg_.prune_inactive_steps;
              if (!retracted && !ancient) return false;
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
      for (int k = 0; k < 6; ++k) {
        const int nx = v.x + kNeighbours[k][0];
        const int ny = v.y + kNeighbours[k][1];
        const int nz = v.z + kNeighbours[k][2];
        if (!grid_.in_bounds(nx, ny, nz)) continue;

        const auto c = grid_.get(nx, ny, nz);
        if (c != BrainGrid::NEURON) continue;  // BLOCKED / SYNAPSE / EMPTY skip

        const uint32_t other = owner_[lin(nx, ny, nz)];
        if (other == 0 || other == pre.id) continue;

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
  fire_dispatch_phase();
  scheduler_dispatch_phase();
  homeostatic_phase();
  pruning_phase();
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
  const char magic[4] = {'S', 'N', 'C', '2'};
  f.write(magic, 4);
  uint32_t version = 2;
  wpod(f, version);
  wpod(f, cfg_);
  wpod(f, step_);
  wpod(f, vz_lo_);
  wpod(f, vz_hi_);

  wvec(f, grid_.raw_front());
  wvec(f, energy_.raw_data());
  wvec(f, owner_);

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
      wpod(f, syn.conduction_delay);
      wpod(f, syn.delivered_count);
      wpod(f, syn.caused_fire_count);
      wpod(f, syn.last_delivery_step);
      wvec(f, syn.transit);
    }
  }
  return f.good();
}

bool Simulator::load_state(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  char magic[4];
  f.read(magic, 4);
  if (std::memcmp(magic, "SNC2", 4) != 0) return false;
  uint32_t version;
  rpod(f, version);
  if (version != 2) return false;

  rpod(f, cfg_);
  rpod(f, step_);
  rpod(f, vz_lo_);
  rpod(f, vz_hi_);

  // Reconstruct the grid + energy field with the saved dims.
  grid_ = BrainGrid(cfg_.X, cfg_.Y, cfg_.Z);
  energy_ = EnergyField(cfg_.X, cfg_.Y, cfg_.Z, cfg_.region_size,
                        cfg_.energy_max);

  rvec(f, grid_.raw_front());
  rvec(f, energy_.raw_data());
  rvec(f, owner_);

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
      rpod(f, syn.conduction_delay);
      rpod(f, syn.delivered_count);
      rpod(f, syn.caused_fire_count);
      rpod(f, syn.last_delivery_step);
      rvec(f, syn.transit);
    }
  }
  return f.good();
}

}  // namespace snc
