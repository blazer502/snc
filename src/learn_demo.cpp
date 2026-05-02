// Toy training demo: pattern association via reward-modulated plasticity.
//
// Task: there are 4 input features and 4 output classes. For each trial a
// one-hot input pattern (target class) is applied; the network must learn to
// fire its `target`-th OUTPUT neuron most strongly. The teaching signal is a
// per-class reward vector -- +1 for the target class, -1 for the others --
// broadcast at the end of each trial. Each synapse uses its own eligibility
// trace and the reward at its post-synaptic channel; per-neuron computation
// stays strictly local.
//
// Layout:
//   INPUT neurons placed at z=2     (bottom of the cortex, sensory band)
//   OUTPUT neurons placed at z=Z-3  (top, motor / decision band)
//   Bulk neurons grown by seed_fetal in between, providing intermediate
//   connectivity that the reward signal sculpts.

#include "simulator.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int kFeatures = 4;
constexpr int kClasses = 4;

int argmax_n(const float* v, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i) if (v[i] > v[best]) best = i;
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  snc::SimConfig cfg;
  cfg.X = 32;
  cfg.Y = 32;
  cfg.Z = 32;
  cfg.region_size = 8;
  cfg.fire_threshold = 0.8f;
  cfg.synapse_form_prob = 0.7f;
  cfg.weight_decay = 0.998f;
  cfg.weight_potentiation = 0.0008f;  // near-zero Hebbian so reward signal
                                      //   is the dominant teacher
  cfg.prune_inactive_steps = 1500;
  cfg.prune_weight_floor = 0.005f;
  cfg.eligibility_decay = 0.88f;
  cfg.eligibility_potentiation = 0.7f;
  cfg.reward_lr = 0.08f;
  cfg.input_drive_strength = 2.0f;
  cfg.initial_weight = 0.35f;
  cfg.weight_max = 1.5f;
  cfg.sprout_attempts = 8;

  int growth_steps = (argc > 1) ? std::atoi(argv[1]) : 700;
  int trials = (argc > 2) ? std::atoi(argv[2]) : 1500;

  snc::Simulator sim(cfg);

  // 1. Seed the bulk of the cortex with seed_fetal but no CP / migrating
  //    neurons -- everything will be VZ-band internal neurons that bridge
  //    the (later-placed) INPUT and OUTPUT layers.
  snc::FetalSeed seed;
  seed.vz_neurons = 80;
  seed.migrating_neurons = 0;
  seed.cortical_plate_neurons = 0;
  seed.vz_thickness = cfg.Z - 4;       // VZ covers most of the cortex so
  seed.radial_glia_density = 0.0f;     //   sprouting can fill the volume
  sim.seed_fetal(seed);

  // 2. Place INPUT neurons in a small (x, y) cluster at z = 2.
  std::vector<uint32_t> input_ids;
  for (int i = 0; i < kFeatures; ++i) {
    const int x = 6 + 6 * (i % 2);
    const int y = 6 + 6 * (i / 2);
    const uint32_t id = sim.add_neuron_at(x, y, 2);
    if (id == 0) {
      std::fprintf(stderr, "failed to place INPUT %d\n", i);
      return 1;
    }
    sim.set_role(id, snc::NeuronRole::INPUT, i);
    input_ids.push_back(id);
  }

  // 3. Place OUTPUT neurons in a similar cluster at z = Z - 3.
  std::vector<uint32_t> output_ids;
  for (int i = 0; i < kClasses; ++i) {
    const int x = 18 + 6 * (i % 2);
    const int y = 18 + 6 * (i / 2);
    const uint32_t id = sim.add_neuron_at(x, y, cfg.Z - 3);
    if (id == 0) {
      std::fprintf(stderr, "failed to place OUTPUT %d\n", i);
      return 1;
    }
    sim.set_role(id, snc::NeuronRole::OUTPUT, i);
    output_ids.push_back(id);
  }

  std::printf("placed %zu INPUT (z=2) and %zu OUTPUT (z=%d) neurons\n",
              input_ids.size(), output_ids.size(), cfg.Z - 3);

  // 4. Pre-training growth phase. We drive INPUT neurons with low random
  //    activity and inject general noise to internal neurons so the network
  //    sprouts a connectome that spans inputs to outputs before training.
  std::mt19937 rng(0xC0FFEE);
  std::uniform_real_distribution<float> noise(0.0f, 0.35f);

  std::printf("[grow] %d steps of spontaneous activity\n", growth_steps);
  for (int s = 0; s < growth_steps; ++s) {
    float pat[kFeatures];
    for (int i = 0; i < kFeatures; ++i) pat[i] = noise(rng);
    sim.apply_input_pattern(pat, kFeatures);
    for (std::size_t id = 1; id <= sim.neuron_count(); ++id) {
      sim.inject_input(static_cast<uint32_t>(id), noise(rng) * 0.3f);
    }
    sim.step();
  }
  std::printf("[grow] done. neurons=%zu synapses=%zu\n",
              sim.neuron_count(), sim.total_synapses());

  // 5. Training loop with per-class reward.
  std::uniform_int_distribution<int> target_dist(0, kClasses - 1);
  std::uniform_real_distribution<float> output_explore(0.0f, 0.06f);
  constexpr int present_steps = 20;
  constexpr int rest_steps = 4;

  int correct_recent = 0;
  const int recent_window = 50;
  std::vector<int> recent_results;
  recent_results.reserve(recent_window);

  std::printf("\n[train] %d trials\n", trials);
  std::printf("trial  target  pred  out0   out1   out2   out3   acc(last %d)\n",
              recent_window);
  const auto t0 = std::chrono::steady_clock::now();

  for (int t = 0; t < trials; ++t) {
    const int target = target_dist(rng);
    float pattern[kFeatures] = {0, 0, 0, 0};
    pattern[target] = 1.0f;

    sim.clear_eligibility();

    // Output exploration noise: only during training, lets non-dominant
    // OUTPUT neurons occasionally fire so their pathways can be credited
    // when correct.
    for (int s = 0; s < present_steps; ++s) {
      sim.apply_input_pattern(pattern, kFeatures);
      for (uint32_t oid : output_ids) {
        sim.inject_input(oid, output_explore(rng));
      }
      // Light internal noise to keep the network active.
      for (std::size_t id = kFeatures + kClasses + 1;
           id <= sim.neuron_count(); ++id) {
        sim.inject_input(static_cast<uint32_t>(id), noise(rng) * 0.25f);
      }
      sim.step();
    }

    float out[kClasses] = {0, 0, 0, 0};
    sim.read_output(out, kClasses);
    const int pred = argmax_n(out, kClasses);
    const bool match = (pred == target);

    // Per-class reward: +1 to the correct channel, -1 to the others.
    // INTERNAL synapses get a small positive reward proportional to overall
    // accuracy of the recent decision (rough credit for "good route").
    float rewards[kClasses];
    for (int c = 0; c < kClasses; ++c) rewards[c] = (c == target) ? 1.0f : -1.0f;
    const float internal_reward = match ? 0.2f : -0.05f;
    sim.apply_reward_per_class(rewards, kClasses, internal_reward);

    for (int s = 0; s < rest_steps; ++s) sim.step();

    if (recent_results.size() >= static_cast<std::size_t>(recent_window)) {
      if (recent_results.front()) --correct_recent;
      recent_results.erase(recent_results.begin());
    }
    recent_results.push_back(match ? 1 : 0);
    if (match) ++correct_recent;

    if (t % 25 == 0 || t == trials - 1) {
      std::printf("%5d  %5d   %4d  %5.3f  %5.3f  %5.3f  %5.3f  %5.1f%%\n",
                  t, target, pred,
                  out[0], out[1], out[2], out[3],
                  100.0f * correct_recent /
                      std::max(1, int(recent_results.size())));
    }
  }

  const auto t1 = std::chrono::steady_clock::now();
  const double dt = std::chrono::duration<double>(t1 - t0).count();
  std::printf("[train] done in %.2fs (%.1f trials/sec). "
              "final synapses=%zu\n",
              dt, trials / dt, sim.total_synapses());

  // 6. Test sweep: present each pattern with no noise, no learning, and
  //    print the output activations.
  std::printf("\n[test] no-noise evaluation:\n");
  std::printf("target  pred  out0   out1   out2   out3   correct?\n");
  int test_correct = 0;
  for (int target = 0; target < kClasses; ++target) {
    float pattern[kFeatures] = {0, 0, 0, 0};
    pattern[target] = 1.0f;
    for (int s = 0; s < present_steps * 2; ++s) {
      sim.apply_input_pattern(pattern, kFeatures);
      sim.step();
    }
    float out[kClasses] = {0, 0, 0, 0};
    sim.read_output(out, kClasses);
    const int pred = argmax_n(out, kClasses);
    const bool ok = (pred == target);
    if (ok) ++test_correct;
    std::printf("%5d   %4d  %5.3f  %5.3f  %5.3f  %5.3f   %s\n",
                target, pred, out[0], out[1], out[2], out[3],
                ok ? "yes" : "no");
  }
  std::printf("[test] %d / %d correct\n", test_correct, kClasses);

  return 0;
}
