// Embodied "mom" / "dad" demo with infant-style developmental curriculum.
//
// Instead of dumping random examples + reward at the network from step 0,
// we walk it through five stages that mirror how a human infant acquires
// a first word. Each stage exercises a *different* part of the loop and
// progressively scaffolds the next:
//
//   1. BABBLE
//      The caregiver isn't talking yet. We force the motor outputs to
//      fire at random ("mama mama dada dada"). Each motor firing pushes
//      a copy of itself into the self-perception inputs (efference copy);
//      STDP forms the first reverse self -> motor links so the baby
//      later "knows" how to make the sound it is hearing.
//
//   2. IMITATION
//      Caregiver says the word: we drive the self-perception input
//      channel directly (the baby hears "mom"). At the same time we
//      gently prime the corresponding motor neuron (chorus). The baby
//      hears + co-fires the motor; STDP strengthens the audio -> motor
//      bridge that babbling started. Reward on match.
//
//   3. PAIRING
//      Caregiver shows mom AND says "mom" together. External sensory
//      pattern + self-perception drive together; motor primed weakly.
//      The sensory -> motor priors get reinforced under the
//      audio-supported signal.
//
//   4. SOLO
//      Caregiver only shows mom (no auditory prompt). Baby must
//      produce the sound on its own. Reward on match. This is the
//      first time the network is on its own with no priming.
//
//   5. TEST
//      Each canonical scene is presented once with no reward and no
//      priming. Accuracy is reported.
//
// After the final stage the brain state is consolidated through a sleep
// replay that re-presents the recently-seen patterns and then saved to
// disk -- the next session can resume from this connectome.

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

constexpr int kExtFeatures = 8;        // 0..3 mom, 4..7 dad
constexpr int kEffFeatures = 2;        // 8 self-mom, 9 self-dad
constexpr int kAllFeatures = kExtFeatures + kEffFeatures;
constexpr int kClasses     = 2;        // 0 = mom, 1 = dad

struct Scene {
  float pattern[kExtFeatures];
  int label;
};

std::vector<Scene> build_scenes() {
  static const float kMomPatterns[][kExtFeatures] = {
      {1, 1, 0, 0, 0, 0, 0, 0},
      {1, 0, 1, 0, 0, 0, 0, 0},
      {0, 1, 1, 0, 0, 0, 0, 0},
      {1, 0, 0, 1, 0, 0, 0, 0},
      {0, 1, 0, 1, 0, 0, 0, 0},
      {1, 1, 1, 0, 0, 0, 0, 0},
  };
  static const float kDadPatterns[][kExtFeatures] = {
      {0, 0, 0, 0, 1, 1, 0, 0},
      {0, 0, 0, 0, 1, 0, 1, 0},
      {0, 0, 0, 0, 0, 1, 1, 0},
      {0, 0, 0, 0, 1, 0, 0, 1},
      {0, 0, 0, 0, 0, 1, 0, 1},
      {0, 0, 0, 0, 1, 1, 1, 0},
  };
  std::vector<Scene> out;
  for (const auto& p : kMomPatterns) {
    Scene s; std::memcpy(s.pattern, p, sizeof(p)); s.label = 0;
    out.push_back(s);
  }
  for (const auto& p : kDadPatterns) {
    Scene s; std::memcpy(s.pattern, p, sizeof(p)); s.label = 1;
    out.push_back(s);
  }
  return out;
}

const char* utter(float mom_rate, float dad_rate) {
  if (mom_rate < 0.05f && dad_rate < 0.05f) return "...";
  return mom_rate >= dad_rate ? "mom" : "dad";
}

// Build a 10-channel feature vector. `scene` may be null (no sensory),
// `say_class` may be -1 (no caregiver audio).
void compose_pattern(float* out10, const Scene* scene, int say_class) {
  for (int i = 0; i < kAllFeatures; ++i) out10[i] = 0.0f;
  if (scene) {
    for (int i = 0; i < kExtFeatures; ++i) out10[i] = scene->pattern[i];
  }
  if (say_class == 0) out10[kExtFeatures + 0] = 1.0f;       // self-mom drive
  if (say_class == 1) out10[kExtFeatures + 1] = 1.0f;       // self-dad drive
}

}  // namespace

int main(int argc, char** argv) {
  // ------------------------------------------------------------------
  //                              CONFIG
  // ------------------------------------------------------------------
  snc::SimConfig cfg;
  cfg.X = 32;
  cfg.Y = 32;
  cfg.Z = 32;
  cfg.region_size = 8;
  cfg.fire_threshold = 0.45f;
  cfg.synapse_form_prob = 0.55f;
  cfg.weight_max = 1.5f;
  cfg.initial_weight = 0.3f;
  cfg.input_drive_strength = 1.4f;
  cfg.eligibility_decay = 0.9f;
  cfg.eligibility_potentiation = 0.5f;
  cfg.reward_lr = 0.06f;
  cfg.stdp_a_ltp = 0.018f;
  cfg.stdp_a_ltd = 0.012f;          // LTP > LTD so co-firing learning
                                    //   dominates over anti-causal erosion
  cfg.stdp_window = 14;
  cfg.stdp_tau = 6.0f;
  cfg.spine_retraction_floor = 0.008f;
  cfg.prune_inactive_steps = 4000;
  cfg.weight_potentiation = 0.0f;

  // Plasticity stabilisers kept very gentle for this small task.
  cfg.homeostatic_rate = 0.0f;
  cfg.heterosynaptic_damp = 0.0f;
  cfg.bcm_baseline_alpha = 0.0f;

  // Multi-compartment outputs: priors on branch 0, sprouted plasticity
  // on branch 1. Threshold low enough that a single sufficiently-grown
  // self -> motor synapse (weight ~ 0.5) can drive a dendritic spike.
  cfg.dendritic_threshold = 0.45f;
  cfg.dendritic_spike_amplitude = 1.0f;
  cfg.dendritic_passive_gain = 0.0f;
  cfg.dendritic_decay = 0.0f;
  cfg.synaptogenesis_default_branch = 1;

  int babble_trials = (argc > 1) ? std::atoi(argv[1]) : 100;
  int imitate_trials = (argc > 2) ? std::atoi(argv[2]) : 200;
  int pair_trials = (argc > 3) ? std::atoi(argv[3]) : 200;
  int solo_trials = (argc > 4) ? std::atoi(argv[4]) : 300;
  const char* save_path = (argc > 5) ? argv[5] : "mom_dad_brain.snc";

  snc::Simulator sim(cfg);

  // ------------------------------------------------------------------
  //                               ANATOMY
  // ------------------------------------------------------------------
  snc::FetalSeed seed;
  seed.vz_neurons = 50;
  seed.migrating_neurons = 0;
  seed.cortical_plate_neurons = 0;
  seed.vz_thickness = cfg.Z - 4;
  seed.radial_glia_density = 0.02f;
  sim.seed_fetal(seed);
  sim.randomize_polarity(0.2f);

  std::vector<uint32_t> ext_in;
  for (int i = 0; i < kExtFeatures; ++i) {
    const int x = 6 + 4 * (i % 4);
    const int y = (i < 4) ? 6 : 18;
    const uint32_t id = sim.add_neuron_at(x, y, 2);
    if (!id) { std::fprintf(stderr, "ext input %d\n", i); return 1; }
    sim.set_role(id, snc::NeuronRole::INPUT, i);
    sim.set_polarity(id, snc::NeuronPolarity::EXCITATORY);
    ext_in.push_back(id);
  }

  uint32_t mom_out = sim.add_neuron_at(10, 12, cfg.Z - 3);
  uint32_t dad_out = sim.add_neuron_at(22, 12, cfg.Z - 3);
  sim.set_role(mom_out, snc::NeuronRole::OUTPUT, 0);
  sim.set_role(dad_out, snc::NeuronRole::OUTPUT, 1);
  sim.set_polarity(mom_out, snc::NeuronPolarity::EXCITATORY);
  sim.set_polarity(dad_out, snc::NeuronPolarity::EXCITATORY);
  sim.set_branches(mom_out, 2);
  sim.set_branches(dad_out, 2);

  uint32_t self_mom = sim.add_neuron_at(10, 16, cfg.Z - 4);
  uint32_t self_dad = sim.add_neuron_at(22, 16, cfg.Z - 4);
  sim.set_role(self_mom, snc::NeuronRole::INPUT, 8);
  sim.set_role(self_dad, snc::NeuronRole::INPUT, 9);
  sim.set_polarity(self_mom, snc::NeuronPolarity::EXCITATORY);
  sim.set_polarity(self_dad, snc::NeuronPolarity::EXCITATORY);

  // ------------------------------------------------------------------
  //                               WIRING
  // ------------------------------------------------------------------

  // Innate priors: external sensory features wire to their motor output's
  // labelled-line dendrite (branch 0). When a coherent set of priors fires
  // they cross the dendritic threshold and the soma fires.
  for (int i = 0; i < 4; ++i) {
    sim.install_synapse(ext_in[i], mom_out, 0.55f, 2, /*branch=*/0);
  }
  for (int i = 4; i < 8; ++i) {
    sim.install_synapse(ext_in[i], dad_out, 0.55f, 2, /*branch=*/0);
  }

  // Efference copy: motor -> self-perception, low delay. The baby hears
  // its own voice as soon as the motor fires.
  sim.install_synapse(mom_out, self_mom, 1.4f, 1, /*branch=*/0);
  sim.install_synapse(dad_out, self_dad, 1.4f, 1, /*branch=*/0);

  // Reverse self -> motor links (initially weak). These are what BABBLE
  // strengthens via Hebbian STDP, and what IMITATION uses to convert a
  // heard syllable into a motor production.
  sim.install_synapse(self_mom, mom_out, 0.25f, 2, /*branch=*/1);
  sim.install_synapse(self_dad, dad_out, 0.25f, 2, /*branch=*/1);

  // Random sparse sensory -> bulk wires give STDP a substrate to evolve
  // a parallel sensory-bulk-motor route in PAIRING and SOLO.
  std::mt19937 rng(0xBEEF);
  std::uniform_int_distribution<int> bulk_pick(
      kExtFeatures + kClasses + kEffFeatures + 1,
      static_cast<int>(sim.neuron_count()));
  for (uint32_t in_id : ext_in) {
    for (int k = 0; k < 3; ++k) {
      sim.install_synapse(in_id, bulk_pick(rng), 0.25f, 4);
    }
  }

  std::printf("anatomy: %zu neurons. priors / efference / reverse "
              "self->motor are pre-wired; the curriculum walks the rest.\n",
              sim.neuron_count());

  // ------------------------------------------------------------------
  //                          CURRICULUM HELPERS
  // ------------------------------------------------------------------
  std::uniform_real_distribution<float> noise(0.0f, 0.15f);
  std::uniform_int_distribution<int> coin(0, 1);
  const auto scenes = build_scenes();
  std::uniform_int_distribution<int> scene_pick(
      0, static_cast<int>(scenes.size()) - 1);

  auto inject_internal_noise = [&]() {
    const std::size_t bulk_start =
        kExtFeatures + kClasses + kEffFeatures + 1;
    for (std::size_t id = bulk_start; id <= sim.neuron_count(); ++id) {
      sim.inject_input(static_cast<uint32_t>(id), noise(rng));
    }
  };

  auto read_motor = [&](float& mr, float& dr) {
    float out[kClasses] = {0, 0};
    sim.read_output(out, kClasses);
    mr = out[0]; dr = out[1];
  };

  // ------------------------------------------------------------------
  //                       STAGE 1 -- BABBLE
  // ------------------------------------------------------------------
  std::printf("\n[stage 1: babble] %d trials -- random motor firings, "
              "STDP forms self -> motor reverse links\n", babble_trials);
  constexpr int babble_present = 14;
  for (int t = 0; t < babble_trials; ++t) {
    const int target = coin(rng);
    const uint32_t motor_id = (target == 0) ? mom_out : dad_out;
    for (int s = 0; s < babble_present; ++s) {
      // Babbling drive: force the chosen motor neuron to fire by injecting
      // strong external input directly into the soma. The efference copy
      // back to the self-perception input then activates STDP on every
      // self -> motor synapse currently in place.
      sim.inject_input(motor_id, 1.5f);
      inject_internal_noise();
      sim.step();
    }
    // Brief silent gap so activity decays before next babble.
    for (int s = 0; s < 4; ++s) sim.step();
  }

  // ------------------------------------------------------------------
  //                       STAGE 2 -- IMITATION
  // ------------------------------------------------------------------
  std::printf("[stage 2: imitate] %d trials -- caregiver speaks, baby "
              "echoes; reward on match\n", imitate_trials);
  constexpr int imitate_present = 16;
  std::vector<std::vector<float>> recent_patterns;
  recent_patterns.reserve(64);

  int imitate_correct = 0;
  for (int t = 0; t < imitate_trials; ++t) {
    const int target = coin(rng);
    const uint32_t motor_id = (target == 0) ? mom_out : dad_out;

    sim.clear_eligibility();

    float pat[kAllFeatures];
    compose_pattern(pat, /*scene=*/nullptr, /*say=*/target);
    for (int s = 0; s < imitate_present; ++s) {
      // Caregiver's voice activates the self-perception input directly.
      sim.apply_input_pattern(pat, kAllFeatures);
      // Gentle motor priming so the baby's motor co-fires and STDP can
      // capture the audio -> motor bridge. Decreases over the stage.
      const float prime = 0.45f * (1.0f - 0.5f * t / float(imitate_trials));
      sim.inject_input(motor_id, prime);
      inject_internal_noise();
      sim.step();
    }
    float mr, dr; read_motor(mr, dr);
    const char* said = utter(mr, dr);
    const char* shown = (target == 0) ? "mom" : "dad";
    const bool match = std::strcmp(said, shown) == 0;
    if (match) ++imitate_correct;

    float rewards[kClasses];
    rewards[0] = (target == 0) ? 1.0f : -0.6f;
    rewards[1] = (target == 1) ? 1.0f : -0.6f;
    sim.apply_reward_per_class(rewards, kClasses, match ? 0.1f : -0.05f);

    for (int s = 0; s < 4; ++s) sim.step();

    if (recent_patterns.size() < 64) {
      recent_patterns.emplace_back(pat, pat + kAllFeatures);
    }

    if (t % 40 == 0 || t == imitate_trials - 1) {
      std::printf("  imitate t=%4d  shown=%s said=%s motor=(%.2f, %.2f) "
                  "match=%d\n", t, shown, said, mr, dr,
                  match ? 1 : 0);
    }
  }
  std::printf("  imitate accuracy: %d / %d (%.1f%%)\n",
              imitate_correct, imitate_trials,
              100.0f * imitate_correct / imitate_trials);

  // ------------------------------------------------------------------
  //                       STAGE 3 -- PAIRING
  // ------------------------------------------------------------------
  std::printf("[stage 3: pair] %d trials -- caregiver shows + says\n",
              pair_trials);
  constexpr int pair_present = 18;
  int pair_correct = 0;
  for (int t = 0; t < pair_trials; ++t) {
    const Scene& scene = scenes[scene_pick(rng)];
    const uint32_t motor_id = (scene.label == 0) ? mom_out : dad_out;

    sim.clear_eligibility();

    float pat[kAllFeatures];
    compose_pattern(pat, &scene, /*say=*/scene.label);
    for (int s = 0; s < pair_present; ++s) {
      sim.apply_input_pattern(pat, kAllFeatures);
      const float prime = 0.25f * (1.0f - float(t) / float(pair_trials));
      sim.inject_input(motor_id, prime);
      inject_internal_noise();
      sim.step();
    }
    float mr, dr; read_motor(mr, dr);
    const char* said = utter(mr, dr);
    const char* shown = (scene.label == 0) ? "mom" : "dad";
    const bool match = std::strcmp(said, shown) == 0;
    if (match) ++pair_correct;

    float rewards[kClasses];
    rewards[0] = (scene.label == 0) ? 1.0f : -0.7f;
    rewards[1] = (scene.label == 1) ? 1.0f : -0.7f;
    sim.apply_reward_per_class(rewards, kClasses, match ? 0.1f : -0.05f);

    for (int s = 0; s < 4; ++s) sim.step();

    if (recent_patterns.size() < 64) {
      recent_patterns.emplace_back(pat, pat + kAllFeatures);
    }

    if (t % 40 == 0 || t == pair_trials - 1) {
      std::printf("  pair    t=%4d  shown=%s said=%s motor=(%.2f, %.2f) "
                  "match=%d\n", t, shown, said, mr, dr,
                  match ? 1 : 0);
    }
  }
  std::printf("  pair accuracy: %d / %d (%.1f%%)\n",
              pair_correct, pair_trials,
              100.0f * pair_correct / pair_trials);

  // ------------------------------------------------------------------
  //                       STAGE 4 -- SOLO
  // ------------------------------------------------------------------
  std::printf("[stage 4: solo] %d trials -- caregiver only shows; "
              "baby produces voice on its own\n", solo_trials);
  constexpr int solo_present = 20;
  int solo_correct = 0;
  for (int t = 0; t < solo_trials; ++t) {
    const Scene& scene = scenes[scene_pick(rng)];

    sim.clear_eligibility();

    float pat[kAllFeatures];
    compose_pattern(pat, &scene, /*say=*/-1);          // sensory only
    for (int s = 0; s < solo_present; ++s) {
      sim.apply_input_pattern(pat, kAllFeatures);
      inject_internal_noise();
      sim.step();
    }
    float mr, dr; read_motor(mr, dr);
    const char* said = utter(mr, dr);
    const char* shown = (scene.label == 0) ? "mom" : "dad";
    const bool match = std::strcmp(said, shown) == 0;
    if (match) ++solo_correct;

    float rewards[kClasses];
    rewards[0] = (scene.label == 0) ? 1.0f : -1.0f;
    rewards[1] = (scene.label == 1) ? 1.0f : -1.0f;
    sim.apply_reward_per_class(rewards, kClasses, match ? 0.15f : -0.05f);

    // Confidence-modulated aversive learning: when the network was
    // *confidently wrong*, fire an extra aversive signal proportional
    // to the confidence. Bounded credulity -- a confident error costs
    // more than an uncertain one, so the network learns to *doubt*
    // patterns that have produced confident misclassifications.
    if (!match) {
      const float confidence = std::fabs(mr - dr);
      if (confidence > 0.1f) {
        sim.apply_aversive(confidence);
      }
    }

    for (int s = 0; s < 4; ++s) sim.step();

    if (t % 40 == 0 || t == solo_trials - 1) {
      std::printf("  solo    t=%4d  shown=%s said=%s motor=(%.2f, %.2f) "
                  "match=%d\n", t, shown, said, mr, dr,
                  match ? 1 : 0);
    }
  }
  std::printf("  solo accuracy: %d / %d (%.1f%%)\n",
              solo_correct, solo_trials,
              100.0f * solo_correct / solo_trials);

  // ------------------------------------------------------------------
  //                       STAGE 5 -- TEST
  // ------------------------------------------------------------------
  std::printf("\n[stage 5: test] no learning; canonical scenes only\n");
  std::printf("shown  said    motor=(mom, dad)\n");
  int test_correct = 0;
  for (const Scene& scene : scenes) {
    // Silent gap so previous activity decays.
    float zero[kAllFeatures] = {0};
    for (int s = 0; s < 30; ++s) {
      sim.apply_input_pattern(zero, kAllFeatures);
      sim.step();
    }
    float pat[kAllFeatures];
    compose_pattern(pat, &scene, /*say=*/-1);
    for (int s = 0; s < solo_present * 2; ++s) {
      sim.apply_input_pattern(pat, kAllFeatures);
      sim.step();
    }
    float mr, dr; read_motor(mr, dr);
    const char* said = utter(mr, dr);
    const char* shown = (scene.label == 0) ? "mom" : "dad";
    const bool ok = std::strcmp(said, shown) == 0;
    if (ok) ++test_correct;
    std::printf("%5s  %5s   (%.3f, %.3f)\n", shown, said, mr, dr);
  }
  std::printf("[test] %d / %zu correct\n", test_correct, scenes.size());

  // ------------------------------------------------------------------
  //                       SLEEP CONSOLIDATION
  // ------------------------------------------------------------------
  std::printf("\n[sleep] replaying %zu recent patterns over 200 steps\n",
              recent_patterns.size());
  sim.sleep_replay_patterns(200, recent_patterns, kAllFeatures, 1.5f);

  if (sim.save_state(save_path)) {
    std::printf("[sleep] saved brain state to %s\n", save_path);
  } else {
    std::printf("[sleep] FAILED to save\n");
    return 1;
  }
  {
    snc::Simulator wake(snc::SimConfig{});
    if (!wake.load_state(save_path)) {
      std::printf("[wake] failed to load\n");
      return 1;
    }
    std::printf("[wake] reloaded: %zu neurons, %zu synapses, step=%d\n",
                wake.neuron_count(), wake.total_synapses(),
                wake.current_step());
  }

  return 0;
}
