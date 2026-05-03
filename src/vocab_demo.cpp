// Multi-word vocabulary demo: scaling the embodied learning loop from
// the 2-word mom/dad task to a small 4-word vocabulary (mom / dad /
// ball / dog). Tests whether the architecture generalises beyond
// binary classification.
//
// Architecture (per-word repeating motif):
//   - 4 sensory INPUT channels per word (16 total, channels 0..15)
//   - 1 motor OUTPUT per word (4 total)
//   - 1 self-perception INPUT per word (channels 16..19)
//   - All motor outputs are multi-compartment: priors on branch 0
//     (heterogeneous innate priors with low threshold + passive leak),
//     bulk on branch 1 (high threshold + no leak; isolated)
//   - Lateral inhibition: each motor excites a dedicated PV interneuron
//     that silences the other motors (winner-take-all)
//
// Curriculum: babble -> imitation -> pairing -> solo -> continuous eval
// (same shape as mom/dad demo, four classes instead of two).

#include "simulator.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kClasses = 4;
constexpr int kFeatPerClass = 4;
constexpr int kExtFeatures = kClasses * kFeatPerClass;       // 16
constexpr int kEffFeatures = kClasses;                       // 4 self-perception
constexpr int kAllFeatures = kExtFeatures + kEffFeatures;    // 20
constexpr float kDecideMargin = 0.04f;

const char* kWords[kClasses] = {"mom", "dad", "ball", "dog"};

struct Scene {
  float pattern[kExtFeatures];
  int label;
};

std::vector<Scene> build_scenes() {
  // For each word, generate 5 distinct "looks": each scene activates a
  // subset of that word's 4 features. This forces the network to learn
  // the feature-class mapping rather than memorising one fingerprint.
  std::vector<Scene> out;
  for (int c = 0; c < kClasses; ++c) {
    static const int kFeatureMasks[5][kFeatPerClass] = {
        {1, 1, 0, 0},
        {1, 0, 1, 0},
        {0, 1, 1, 0},
        {1, 0, 0, 1},
        {0, 1, 0, 1},
    };
    for (const auto& mask : kFeatureMasks) {
      Scene s;
      std::memset(s.pattern, 0, sizeof(s.pattern));
      for (int i = 0; i < kFeatPerClass; ++i) {
        s.pattern[c * kFeatPerClass + i] = static_cast<float>(mask[i]);
      }
      s.label = c;
      out.push_back(s);
    }
  }
  return out;
}

int argmax_n(const float* v, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i) if (v[i] > v[best]) best = i;
  return best;
}

const char* utter(const float* rates) {
  // Single-winner readout. Returns "..." if the leading two outputs are
  // too close to call.
  int top = argmax_n(rates, kClasses);
  float top_v = rates[top];
  float second = 0.0f;
  for (int i = 0; i < kClasses; ++i) {
    if (i != top && rates[i] > second) second = rates[i];
  }
  if (top_v < 0.05f) return "...";
  if (top_v - second < kDecideMargin) return "...";
  return kWords[top];
}

void compose_pattern(float* out, const Scene* scene, int say_class) {
  for (int i = 0; i < kAllFeatures; ++i) out[i] = 0.0f;
  if (scene) {
    for (int i = 0; i < kExtFeatures; ++i) out[i] = scene->pattern[i];
  }
  if (say_class >= 0 && say_class < kClasses) {
    out[kExtFeatures + say_class] = 1.0f;
  }
}

}  // namespace

int main(int argc, char** argv) {
  snc::SimConfig cfg;
  cfg.X = 48;
  cfg.Y = 48;
  cfg.Z = 48;
  cfg.region_size = 8;
  cfg.fire_threshold = 0.45f;
  cfg.synapse_form_prob = 0.55f;
  cfg.weight_max = 1.5f;
  cfg.initial_weight = 0.3f;
  cfg.input_drive_strength = 1.4f;
  cfg.eligibility_decay = 0.9f;
  cfg.eligibility_potentiation = 0.5f;
  cfg.reward_lr = 0.05f;
  cfg.aversive_amplification = 2.5f;
  cfg.stdp_a_ltp = 0.018f;
  cfg.stdp_a_ltd = 0.012f;
  cfg.stdp_window = 14;
  cfg.stdp_tau = 6.0f;
  cfg.spine_retraction_floor = 0.008f;
  cfg.prune_inactive_steps = 4000;

  // New stability mechanisms exercised here: a small refractory period
  // (~3 steps) prevents any motor neuron from firing every single step
  // and forming a self-sustaining attractor; mild stochastic vesicle
  // release adds biological baseline noise that further breaks
  // deterministic loops.
  cfg.refractory_steps = 3;
  cfg.release_probability = 0.85f;
  cfg.weight_potentiation = 0.0f;
  cfg.homeostatic_rate = 0.0f;
  cfg.heterosynaptic_damp = 0.0f;
  cfg.bcm_baseline_alpha = 0.0f;

  // Multi-compartment integration: keep cfg defaults compatible with
  // single-compartment cells (full passive transmission, no spike) so
  // inhibitors / self / bulk neurons still receive their synaptic
  // input. Multi-compartment motors then *override* per-branch.
  cfg.dendritic_threshold = 1.0e9f;          // single-compartment: no spike
  cfg.dendritic_spike_amplitude = 1.0f;
  cfg.dendritic_passive_gain = 1.0f;         // single-compartment: full leak
  cfg.dendritic_decay = 0.0f;                // no temporal accumulation for legacy
  cfg.synaptogenesis_default_branch = 1;

  // Default curriculum lengths. The small-vocabulary task is sensitive
  // to long imitate stages: extended forced-prime firing on a 4-class
  // network builds up enough recurrent activity to produce a "stuck
  // motor" attractor. Empirically, imitate <= 20 trials keeps the
  // network out of the attractor and yields ~95% probe accuracy.
  int babble_trials = (argc > 1) ? std::atoi(argv[1]) : 80;
  int imitate_trials = (argc > 2) ? std::atoi(argv[2]) : 20;
  int pair_trials = (argc > 3) ? std::atoi(argv[3]) : 240;
  int solo_trials = (argc > 4) ? std::atoi(argv[4]) : 320;

  snc::Simulator sim(cfg);

  // -------- Anatomy ----------------------------------------------------
  snc::FetalSeed seed;
  seed.vz_neurons = 240;
  seed.migrating_neurons = 0;            // keep upper-z lanes free for
  seed.cortical_plate_neurons = 0;       //   the hand-installed motor /
                                         //   self-perception clusters
  seed.vz_thickness = cfg.Z - 5;
  // No radial-glia scaffold so the hand-installed motor / self-perception
  // / inhibitor positions are guaranteed empty -- glia would otherwise
  // BLOCK random (x, y) columns through all z.
  seed.radial_glia_density = 0.0f;
  seed.frac_pv = 0.14f;
  seed.frac_sst = 0.04f;
  seed.frac_vip = 0.02f;
  seed.brainstem_neurons = 14;
  seed.thalamic_relay_neurons = 20;
  seed.aversive_nucleus_neurons = 6;
  sim.seed_fetal(seed);

  // External sensory inputs (channels 0..15).
  std::vector<uint32_t> ext_in;
  for (int c = 0; c < kClasses; ++c) {
    for (int f = 0; f < kFeatPerClass; ++f) {
      const int channel = c * kFeatPerClass + f;
      const int x = 4 + f * 4;
      const int y = 4 + c * 10;
      const uint32_t id = sim.add_neuron_at(x, y, 2);
      if (!id) { std::fprintf(stderr, "ext input %d failed\n", channel);
                 return 1; }
      sim.set_role(id, snc::NeuronRole::INPUT, channel);
      sim.set_polarity(id, snc::NeuronPolarity::EXCITATORY);
      ext_in.push_back(id);
    }
  }

  // Motor outputs (one per word) + self-perception inputs.
  std::vector<uint32_t> motors;
  std::vector<uint32_t> selfs;
  std::vector<uint32_t> inhibitors;
  for (int c = 0; c < kClasses; ++c) {
    const int xm = 6 + c * 8;
    const int ym = cfg.Y / 2;
    const uint32_t m = sim.add_neuron_at(xm, ym, cfg.Z - 3);
    if (!m) { std::fprintf(stderr, "motor %d failed\n", c); return 1; }
    sim.set_role(m, snc::NeuronRole::OUTPUT, c);
    sim.set_polarity(m, snc::NeuronPolarity::EXCITATORY);
    sim.set_branches(m, 2);
    // Branch 0 (innate priors): low threshold so two priors at 0.45
    // each (sum 0.9) crosses; modest passive leak so a single prior
    // still nudges the soma. Branch 1 (sprouted bulk): unreachable
    // threshold + zero leak so background noise is structurally
    // isolated from the soma.
    sim.set_branch_threshold(m, 0, 0.8f);
    sim.set_branch_passive_gain(m, 0, 0.3f);
    sim.set_branch_threshold(m, 1, 1.0e9f);
    sim.set_branch_passive_gain(m, 1, 0.0f);
    motors.push_back(m);

    // Self-perception input for this word (channel kExtFeatures + c).
    // Multi-compartment with the same isolation trick as the motors:
    // branch 0 receives the legitimate efference-copy wire from motor;
    // branch 1 absorbs whatever sprouting the bulk grows toward this
    // cell and is structurally isolated from the soma.
    const uint32_t s = sim.add_neuron_at(xm, ym + 4, cfg.Z - 4);
    if (!s) { std::fprintf(stderr, "self %d failed\n", c); return 1; }
    sim.set_role(s, snc::NeuronRole::INPUT, kExtFeatures + c);
    sim.set_polarity(s, snc::NeuronPolarity::EXCITATORY);
    sim.set_branches(s, 2);
    sim.set_branch_threshold(s, 0, 1.0e9f);  // pure passive on branch 0
    sim.set_branch_passive_gain(s, 0, 1.0f);
    sim.set_branch_threshold(s, 1, 1.0e9f);  // sprouting branch muted
    sim.set_branch_passive_gain(s, 1, 0.0f);
    selfs.push_back(s);

    // Lateral-inhibition interneuron for this word (PV-style).
    // Same multi-compartment isolation: real PV cells in cortex
    // receive their drive on a few selective dendrites; uncontrolled
    // bulk sprouting onto the same compartment would let baseline
    // activity tonically silence the network.
    const uint32_t inh = sim.add_neuron_at(xm + 1, ym, cfg.Z - 4);
    sim.set_polarity(inh, snc::NeuronPolarity::INHIBITORY);
    sim.set_branches(inh, 2);
    sim.set_branch_threshold(inh, 0, 1.0e9f);
    sim.set_branch_passive_gain(inh, 0, 1.0f);
    sim.set_branch_threshold(inh, 1, 1.0e9f);
    sim.set_branch_passive_gain(inh, 1, 0.0f);
    inhibitors.push_back(inh);
  }

  // -------- Wiring ----------------------------------------------------

  // Innate priors: each word's 4 sensory inputs prime its motor neuron's
  // branch 0 with a strong weight. Two priors at 0.55 sum to 1.1, well
  // above branch 0's threshold of 0.8 -> reliable dendritic spike on
  // any 2-feature scene. They are installed with `innate_tag = 1.0` so
  // the consolidation-tag protection shields them from spine retraction
  // during stages where their input channel happens to be silent (real
  // cortex spares labelled-line connections from microglial pruning).
  for (int c = 0; c < kClasses; ++c) {
    for (int f = 0; f < kFeatPerClass; ++f) {
      sim.install_synapse(ext_in[c * kFeatPerClass + f],
                          motors[c], 0.55f, 4, /*branch=*/0,
                          /*innate_tag=*/1.0f);
    }
  }

  // Efference copy: each motor projects to its own self-perception input.
  // Innate-tagged so the link survives quiet stages.
  for (int c = 0; c < kClasses; ++c) {
    sim.install_synapse(motors[c], selfs[c], 1.4f, 1, /*branch=*/0,
                        /*innate_tag=*/1.0f);
  }

  // Reverse self -> motor (initially weak; babble strengthens via STDP).
  // Tagged so it survives but with a slightly lower tag so it can still
  // potentially be re-shaped if the brain decides the wiring should
  // change (defensive, not a hard requirement).
  for (int c = 0; c < kClasses; ++c) {
    sim.install_synapse(selfs[c], motors[c], 0.25f, 2, /*branch=*/1,
                        /*innate_tag=*/0.7f);
  }

  // Lateral inhibition: each motor excites its inhibitor; each
  // inhibitor silences every other motor. Cortical winner-take-all.
  // Tagged because these are part of the labelled-line architecture.
  for (int c = 0; c < kClasses; ++c) {
    sim.install_synapse(motors[c], inhibitors[c], 0.7f, 1, /*branch=*/0,
                        /*innate_tag=*/1.0f);
    for (int j = 0; j < kClasses; ++j) {
      if (j == c) continue;
      sim.install_synapse(inhibitors[c], motors[j], 1.4f, 1, /*branch=*/0,
                          /*innate_tag=*/1.0f);
    }
  }

  // Sparse sensory -> bulk seeds for plasticity to evolve a parallel
  // route through the cortex.
  std::mt19937 rng(0xFEEDC0DE);
  std::uniform_int_distribution<int> bulk_pick(
      kExtFeatures + kClasses * 3 + 1,
      static_cast<int>(sim.neuron_count()));
  for (uint32_t in_id : ext_in) {
    for (int k = 0; k < 3; ++k) {
      sim.install_synapse(in_id, bulk_pick(rng), 0.2f, 4);
    }
  }

  std::printf("anatomy: %zu neurons total -- 16 ext INPUT, 4 OUTPUT, "
              "4 self-percept INPUT, 4 lateral-inh interneurons\n",
              sim.neuron_count());

  // ---------------- Curriculum -----------------------------------------
  std::uniform_real_distribution<float> noise(0.0f, 0.12f);
  std::uniform_int_distribution<int> coin(0, kClasses - 1);
  const auto scenes = build_scenes();
  std::uniform_int_distribution<int> scene_pick(
      0, static_cast<int>(scenes.size()) - 1);

  std::vector<bool> skip_noise(sim.neuron_count() + 1, false);
  for (uint32_t id : ext_in)    skip_noise[id] = true;
  for (uint32_t id : motors)    skip_noise[id] = true;
  for (uint32_t id : selfs)     skip_noise[id] = true;
  for (uint32_t id : inhibitors) skip_noise[id] = true;

  auto inject_internal_noise = [&]() {
    for (std::size_t id = 1; id <= sim.neuron_count(); ++id) {
      if (id < skip_noise.size() && skip_noise[id]) continue;
      sim.inject_input(static_cast<uint32_t>(id), noise(rng));
    }
  };

  auto read_motors = [&](float* rates) {
    sim.read_output(rates, kClasses);
  };

  // STAGE 1 -- babble
  std::printf("\n[stage 1: babble] %d trials\n", babble_trials);
  for (int t = 0; t < babble_trials; ++t) {
    const int target = coin(rng);
    const uint32_t motor_id = motors[target];
    for (int s = 0; s < 14; ++s) {
      sim.inject_input(motor_id, 1.6f);
      inject_internal_noise();
      sim.step();
    }
    for (int s = 0; s < 4; ++s) sim.step();
  }

  // STAGE 2 -- imitation
  std::printf("[stage 2: imitate] %d trials\n", imitate_trials);
  std::vector<std::vector<float>> recent_patterns;
  recent_patterns.reserve(64);
  int imitate_correct = 0;
  for (int t = 0; t < imitate_trials; ++t) {
    const int target = coin(rng);
    sim.clear_eligibility();
    float pat[kAllFeatures];
    compose_pattern(pat, /*scene=*/nullptr, /*say=*/target);
    for (int s = 0; s < 16; ++s) {
      sim.apply_input_pattern(pat, kAllFeatures);
      const float prime = 0.5f * (1.0f - 0.5f * t / float(imitate_trials));
      sim.inject_input(motors[target], prime);
      inject_internal_noise();
      sim.step();
    }
    float rates[kClasses]; read_motors(rates);
    const char* said = utter(rates);
    const bool match = std::strcmp(said, kWords[target]) == 0;
    if (match) ++imitate_correct;

    float rewards[kClasses];
    for (int c = 0; c < kClasses; ++c) {
      rewards[c] = (c == target) ? 1.0f : -0.6f;
    }
    sim.apply_reward_per_class(rewards, kClasses, match ? 0.1f : -0.05f);
    for (int s = 0; s < 4; ++s) sim.step();

    if (recent_patterns.size() < 64) {
      recent_patterns.emplace_back(pat, pat + kAllFeatures);
    }
    if (t % 60 == 0 || t == imitate_trials - 1) {
      std::printf("  imitate t=%4d  shown=%s said=%s\n",
                  t, kWords[target], said);
    }
  }
  std::printf("  imitate accuracy: %d / %d (%.1f%%)\n",
              imitate_correct, imitate_trials,
              100.0f * imitate_correct / imitate_trials);

  // STAGE 3 -- pairing
  std::printf("[stage 3: pair] %d trials\n", pair_trials);
  int pair_correct = 0;
  for (int t = 0; t < pair_trials; ++t) {
    const Scene& scene = scenes[scene_pick(rng)];
    sim.clear_eligibility();
    float pat[kAllFeatures];
    compose_pattern(pat, &scene, /*say=*/scene.label);
    for (int s = 0; s < 18; ++s) {
      sim.apply_input_pattern(pat, kAllFeatures);
      const float prime = 0.3f * (1.0f - float(t) / float(pair_trials));
      sim.inject_input(motors[scene.label], prime);
      inject_internal_noise();
      sim.step();
    }
    float rates[kClasses]; read_motors(rates);
    const char* said = utter(rates);
    const bool match = std::strcmp(said, kWords[scene.label]) == 0;
    if (match) ++pair_correct;

    float rewards[kClasses];
    for (int c = 0; c < kClasses; ++c) {
      rewards[c] = (c == scene.label) ? 1.0f : -0.7f;
    }
    sim.apply_reward_per_class(rewards, kClasses, match ? 0.1f : -0.05f);
    for (int s = 0; s < 4; ++s) sim.step();

    if (recent_patterns.size() < 64) {
      recent_patterns.emplace_back(pat, pat + kAllFeatures);
    }
    if (t % 60 == 0 || t == pair_trials - 1) {
      std::printf("  pair    t=%4d  shown=%s said=%s\n",
                  t, kWords[scene.label], said);
    }
  }
  std::printf("  pair accuracy: %d / %d (%.1f%%)\n",
              pair_correct, pair_trials,
              100.0f * pair_correct / pair_trials);

  // STAGE 4 -- solo
  std::printf("[stage 4: solo] %d trials\n", solo_trials);
  int solo_correct = 0;
  for (int t = 0; t < solo_trials; ++t) {
    const Scene& scene = scenes[scene_pick(rng)];
    sim.clear_eligibility();
    float pat[kAllFeatures];
    compose_pattern(pat, &scene, /*say=*/-1);
    for (int s = 0; s < 20; ++s) {
      sim.apply_input_pattern(pat, kAllFeatures);
      inject_internal_noise();
      sim.step();
    }
    float rates[kClasses]; read_motors(rates);
    const char* said = utter(rates);
    const bool match = std::strcmp(said, kWords[scene.label]) == 0;
    if (match) ++solo_correct;

    float rewards[kClasses];
    for (int c = 0; c < kClasses; ++c) {
      rewards[c] = (c == scene.label) ? 1.0f : -1.0f;
    }
    sim.apply_reward_per_class(rewards, kClasses, match ? 0.15f : -0.05f);
    if (!match) {
      const float top = *std::max_element(rates, rates + kClasses);
      float second = 0;
      for (int c = 0; c < kClasses; ++c) if (rates[c] != top) second = std::max(second, rates[c]);
      const float confidence = top - second;
      if (confidence > 0.1f) sim.apply_aversive(confidence);
    }
    for (int s = 0; s < 4; ++s) sim.step();

    if (t % 60 == 0 || t == solo_trials - 1) {
      std::printf("  solo    t=%4d  shown=%s said=%s\n",
                  t, kWords[scene.label], said);
    }
  }
  std::printf("  solo accuracy: %d / %d (%.1f%%)\n",
              solo_correct, solo_trials,
              100.0f * solo_correct / solo_trials);

  // STAGE 5 -- continuous evaluation (no discrete test).
  std::printf("\n[stage 5: continuous-eval] rolling accuracy is the "
              "developmental milestone\n");
  constexpr int continuous_trials = 60;
  int rolling_correct = 0;
  for (int t = 0; t < continuous_trials; ++t) {
    const Scene& scene = scenes[scene_pick(rng)];
    sim.clear_eligibility();
    float pat[kAllFeatures];
    compose_pattern(pat, &scene, /*say=*/-1);
    for (int s = 0; s < 18; ++s) {
      sim.apply_input_pattern(pat, kAllFeatures);
      inject_internal_noise();
      sim.step();
    }
    float rates[kClasses]; read_motors(rates);
    const char* said = utter(rates);
    const bool match = std::strcmp(said, kWords[scene.label]) == 0;
    if (match) ++rolling_correct;

    float rewards[kClasses];
    for (int c = 0; c < kClasses; ++c) {
      rewards[c] = (c == scene.label) ? 0.4f : -0.3f;
    }
    sim.apply_reward_per_class(rewards, kClasses, 0.0f);
    for (int s = 0; s < 3; ++s) sim.step();

    if ((t + 1) % 10 == 0) {
      std::printf("  walk t=%2d  shown=%-4s said=%-4s rolling=%d/%d\n",
                  t, kWords[scene.label], said,
                  rolling_correct, t + 1);
    }
  }
  std::printf("[continuous-eval] %d / %d (%.1f%%)\n",
              rolling_correct, continuous_trials,
              100.0f * rolling_correct / continuous_trials);

  // Probe each canonical scene without learning + noise.
  std::printf("\n[probe] no-noise canonical patterns:\n");
  std::printf("shown  said   rates(mom/dad/ball/dog)\n");
  int probe_correct = 0;
  for (const Scene& scene : scenes) {
    // Active reset between scenes: zero every neuron's transient state
    // so the previous scene's saturated activity doesn't leak through
    // a self-sustaining attractor. Structural weights / tags / the
    // grid are preserved -- only the chemistry is reset.
    sim.reset_dynamics();
    float zero[kAllFeatures] = {0};
    for (int s = 0; s < 30; ++s) {
      sim.apply_input_pattern(zero, kAllFeatures);
      sim.step();
    }
    float pat[kAllFeatures];
    compose_pattern(pat, &scene, /*say=*/-1);
    for (int s = 0; s < 36; ++s) {
      sim.apply_input_pattern(pat, kAllFeatures);
      sim.step();
    }
    float rates[kClasses]; read_motors(rates);
    const char* said = utter(rates);
    const bool ok = std::strcmp(said, kWords[scene.label]) == 0;
    if (ok) ++probe_correct;
    std::printf("%4s   %4s   (%.2f, %.2f, %.2f, %.2f)\n",
                kWords[scene.label], said,
                rates[0], rates[1], rates[2], rates[3]);
  }
  std::printf("[probe] %d / %zu correct\n", probe_correct, scenes.size());

  // Sleep replay + save.
  std::printf("\n[sleep] consolidating recent patterns\n");
  sim.sleep_replay_patterns(200, recent_patterns, kAllFeatures, 1.5f);
  sim.save_state("vocab_brain.snc");
  std::printf("[sleep] saved.\n");

  return 0;
}
