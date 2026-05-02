// Demo driver. Two seeding modes are supported:
//
//   snc_demo                       -> staggered cortical-area schedule
//   snc_demo random [steps] [n]    -> uniform random seed, no schedule
//
// In schedule mode the (x, y) plane is divided into four cortical areas with
// different developmental clocks. Each area independently runs through a
// fetal -> infant -> child -> adult sequence, but the areas enter each phase
// at different absolute steps (motor first, language last). This mirrors the
// fact that primary motor / sensory cortices mature ahead of higher-order
// association and language regions in real cortical development.
//
// Two outputs are produced:
//   - stdout: per-area neuron / synapse counts at sampled steps
//   - dev_curve.csv: per-step deltas (sprouts, synapses_formed, synapses_pruned,
//     spikes, total_synapses) for plotting the synaptogenesis vs. pruning
//     curves.

#include "simulator.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct AreaSchedule {
  const char* name;
  snc::BoundingBox box;
  int fetal_start;        // step when this area starts neurogenesis
  int infant_start;       // step when birth rate drops to maintenance
  int adult_start;        // step when birth stops entirely
  int fetal_birth_rate;   // births per step during fetal
  int infant_birth_rate;  // births per step during infant
  int prune_inactive_at_adult;
  float synapse_form_prob_at_adult;
};

// Four cortical areas, one per (x, y) quadrant. Motor / sensory mature first
// (lower fetal_start), language matures last. The numbers are picked for
// demo readability, not to match any real timeline in absolute units.
constexpr AreaSchedule kAreas[] = {
    {"motor",     {  2, 30,  2, 30},   0, 100, 220, 3, 1, 80,  0.30f},
    {"sensory",   { 34, 62,  2, 30},  30, 130, 260, 3, 1, 80,  0.30f},
    {"associate", {  2, 30, 34, 62},  90, 200, 330, 3, 1, 120, 0.40f},
    {"language",  { 34, 62, 34, 62}, 150, 260, 400, 3, 1, 160, 0.45f},
};
constexpr int kNumAreas = sizeof(kAreas) / sizeof(kAreas[0]);

const char* phase_name(const AreaSchedule& a, int step) {
  if (step < a.fetal_start) return "pre";
  if (step < a.infant_start) return "fetal";
  if (step < a.adult_start) return "infant";
  return "adult";
}

int birth_rate_for(const AreaSchedule& a, int step) {
  if (step < a.fetal_start) return 0;
  if (step < a.infant_start) return a.fetal_birth_rate;
  if (step < a.adult_start) return a.infant_birth_rate;
  return 0;
}

bool in_box(const snc::Voxel& v, const snc::BoundingBox& b) {
  return v.x >= b.x_lo && v.x <= b.x_hi &&
         v.y >= b.y_lo && v.y <= b.y_hi;
}

struct AreaMetrics {
  int neurons = 0;
  int synapses = 0;
  int body_voxels = 0;
};

std::vector<AreaMetrics> compute_area_metrics(const snc::Simulator& sim) {
  std::vector<AreaMetrics> m(kNumAreas);
  for (const auto& nu : sim.neurons()) {
    int idx = -1;
    for (int i = 0; i < kNumAreas; ++i) {
      if (in_box(nu.soma, kAreas[i].box)) { idx = i; break; }
    }
    if (idx < 0) continue;
    m[idx].neurons += 1;
    m[idx].synapses += static_cast<int>(nu.outgoing.size());
    m[idx].body_voxels += static_cast<int>(nu.body.size());
  }
  return m;
}

// The whole adult-pruning ramp is piecewise linear: aggressive pruning kicks
// in once *all* areas have entered their adult phase. This is intentionally
// simple -- the per-area schedule already gives different maturation curves
// even with a single global pruning policy.
void update_global_pruning(snc::Simulator& sim, int step) {
  bool any_pre_adult = false;
  for (const auto& a : kAreas) {
    if (step < a.adult_start) { any_pre_adult = true; break; }
  }
  auto& cfg = sim.mutable_config();
  if (any_pre_adult) {
    cfg.prune_inactive_steps = 400;
    cfg.weight_decay = 0.997f;
  } else {
    cfg.prune_inactive_steps = 80;
    cfg.weight_decay = 0.992f;
  }
}

}  // namespace

int main(int argc, char** argv) {
  snc::SimConfig cfg;
  cfg.X = 64;
  cfg.Y = 64;
  cfg.Z = 64;
  cfg.region_size = 8;

  std::string mode = "schedule";
  int total_steps = 600;
  int n_neurons = 80;

  if (argc > 1) mode = argv[1];
  if (argc > 2) total_steps = std::atoi(argv[2]);
  if (argc > 3) n_neurons = std::atoi(argv[3]);

  snc::Simulator sim(cfg);

  if (mode == "random") {
    sim.seed_neurons(n_neurons);
    std::printf("uniform random seed: %zu neurons in %dx%dx%d\n",
                sim.neuron_count(), cfg.X, cfg.Y, cfg.Z);
  } else {
    // Place only the radial-glia scaffold and energy gradient -- the four
    // areas grow their own populations entirely via birth_neurons.
    snc::FetalSeed f;
    f.vz_neurons = 0;
    f.migrating_neurons = 0;
    f.cortical_plate_neurons = 0;
    sim.seed_fetal(f);
    std::printf("scaffold-only seed in %dx%dx%d (areas grow their own "
                "neurons)\n", cfg.X, cfg.Y, cfg.Z);
  }

  std::FILE* csv = std::fopen("dev_curve.csv", "w");
  if (csv) {
    std::fprintf(csv,
                 "step,sprouts,synapses_formed,synapses_pruned,spikes,"
                 "total_neurons,total_synapses,structural_neurons,"
                 "max_blob_size,occupancy,grid_x,grid_y,grid_z,"
                 "motor_n,motor_s,sensory_n,sensory_s,"
                 "associate_n,associate_s,language_n,language_s\n");
  }
  // Sample the structural-neuron count every `structural_sample_every`
  // steps. Flood-fill is O(volume); too frequent and it dominates the
  // step time. 25 steps is a good middle-ground for plotting purposes.
  constexpr int kStructuralSampleEvery = 25;
  int last_structural_count = 0;
  int last_max_blob = 0;
  float last_occupancy = 0.0f;

  std::printf("\nstep  motor(n/s)  sensory(n/s)  associate(n/s)  language(n/s)"
              "  formed/pruned\n");

  std::mt19937 noise_rng(42);
  std::uniform_real_distribution<float> noise(0.0f, 0.2f);

  const auto t0 = std::chrono::steady_clock::now();

  for (int s = 0; s < total_steps; ++s) {
    if (mode == "schedule") {
      update_global_pruning(sim, s);
      // Per-area neurogenesis on its own clock.
      for (const auto& a : kAreas) {
        const int rate = birth_rate_for(a, s);
        if (rate > 0) sim.birth_neurons(rate, &a.box);
      }
    }

    // External drive: a few first neurons strongly, others at low noise.
    const std::size_t driven =
        sim.neuron_count() < 5 ? sim.neuron_count() : std::size_t{5};
    for (std::size_t id = 1; id <= sim.neuron_count(); ++id) {
      const float amount = (id <= driven) ? 0.6f : noise(noise_rng);
      sim.inject_input(static_cast<uint32_t>(id), amount);
    }

    sim.step();

    // Periodically sample the (expensive) structural-neuron flood-fill
    // count and reuse the last sample for the in-between rows.
    if (s % kStructuralSampleEvery == 0 || s == total_steps - 1) {
      auto sizes = sim.structural_neuron_sizes();
      last_structural_count = static_cast<int>(sizes.size());
      last_max_blob = 0;
      long long sum = 0;
      for (int v : sizes) {
        if (v > last_max_blob) last_max_blob = v;
        sum += v;
      }
      const auto& g = sim.grid();
      const long long volume = 1LL * g.X() * g.Y() * g.Z();
      last_occupancy = volume > 0 ? float(sum) / float(volume) : 0.0f;
    }

    const auto& st = sim.last_stats();
    if (csv && (mode == "schedule")) {
      const auto m = compute_area_metrics(sim);
      const auto& g = sim.grid();
      std::fprintf(csv,
                   "%d,%d,%d,%d,%d,%zu,%zu,%d,%d,%.4f,%d,%d,%d,"
                   "%d,%d,%d,%d,%d,%d,%d,%d\n",
                   s, st.sprouts, st.synapses_formed, st.synapses_pruned,
                   st.spikes,
                   sim.neuron_count(), sim.total_synapses(),
                   last_structural_count, last_max_blob, last_occupancy,
                   g.X(), g.Y(), g.Z(),
                   m[0].neurons, m[0].synapses,
                   m[1].neurons, m[1].synapses,
                   m[2].neurons, m[2].synapses,
                   m[3].neurons, m[3].synapses);
    }

    if (s % 40 == 0 || s == total_steps - 1) {
      if (mode == "schedule") {
        const auto m = compute_area_metrics(sim);
        std::printf("%4d  %4d/%-4d   %4d/%-4d     %4d/%-4d       %4d/%-4d "
                    "       %4d/%-4d\n",
                    s,
                    m[0].neurons, m[0].synapses,
                    m[1].neurons, m[1].synapses,
                    m[2].neurons, m[2].synapses,
                    m[3].neurons, m[3].synapses,
                    st.synapses_formed, st.synapses_pruned);
      } else {
        std::printf("%4d  total: %zu neurons %zu synapses; "
                    "formed=%d pruned=%d\n",
                    s, sim.neuron_count(), sim.total_synapses(),
                    st.synapses_formed, st.synapses_pruned);
      }
    }
  }

  if (csv) std::fclose(csv);

  const auto t1 = std::chrono::steady_clock::now();
  const double dt = std::chrono::duration<double>(t1 - t0).count();
  std::printf("\nran %d steps in %.3fs (%.1f steps/sec)\n",
              total_steps, dt, total_steps / dt);
  std::printf("wrote per-step metrics to dev_curve.csv\n");

  return 0;
}
