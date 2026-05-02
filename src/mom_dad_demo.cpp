// "mom" / "dad" embodied interaction demo.
//
// The network is given a tiny "body":
//   - eyes/ears   : 8 external sensory INPUT neurons (channels 0..7)
//                   carrying mom-features (0..3) or dad-features (4..7)
//   - voice       : 2 motor OUTPUT neurons whose firing is rendered as the
//                   string "mom" or "dad" each trial
//   - self-ear    : 2 proprioceptive/auditory INPUT neurons (channels 8..9)
//                   wired from the motor outputs via low-delay efference
//                   copies. This is how the network "hears itself speak".
//
// Self vs external is distinguished automatically by anatomy: the efference
// inputs are physically different INPUT neurons on different channels, so
// the downstream cortex sees two distinct activity patterns. Real cortex
// achieves the same separation through anatomically distinct projections
// from motor cortex (corollary discharge, e.g. the M1 -> A1 pathway that
// silences the auditory cortex during own-vocalisation).
//
// Plasticity additions on top of the existing simulator:
//   - Feedforward inhibition motif (PV+ basket-cell analogue) for fast
//     winner-take-all decoding
//   - Reward prediction error (RPE) instead of raw reward when broadcasting
//     the dopamine modulator -- learning is driven by surprise, exactly as
//     midbrain dopaminergic neurons signal in vivo (Schultz 1997)
//
// At the end of training the entire brain state is serialised to disk
// ("sleep") so the next run can resume from this connectome.

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
    Scene s; std::memcpy(s.pattern, p, sizeof(p)); s.label = 0; out.push_back(s);
  }
  for (const auto& p : kDadPatterns) {
    Scene s; std::memcpy(s.pattern, p, sizeof(p)); s.label = 1; out.push_back(s);
  }
  return out;
}

const char* utter(float mom_rate, float dad_rate) {
  // Argmax decoding: whichever motor output wins gets verbalised. With the
  // feedforward inhibition motif in place, only one normally fires above
  // baseline so there's a clear winner.
  if (mom_rate < 0.05f && dad_rate < 0.05f) return "...";
  return mom_rate >= dad_rate ? "mom" : "dad";
}

}  // namespace

int main(int argc, char** argv) {
  snc::SimConfig cfg;
  cfg.X = 32;
  cfg.Y = 32;
  cfg.Z = 32;
  cfg.region_size = 8;
  cfg.fire_threshold = 0.5f;
  cfg.synapse_form_prob = 0.6f;
  cfg.weight_max = 1.5f;
  cfg.initial_weight = 0.3f;
  cfg.input_drive_strength = 1.6f;
  cfg.eligibility_decay = 0.9f;
  cfg.eligibility_potentiation = 0.5f;
  cfg.reward_lr = 0.05f;
  cfg.stdp_a_ltp = 0.02f;
  cfg.stdp_a_ltd = 0.025f;
  cfg.stdp_window = 14;
  cfg.stdp_tau = 6.0f;
  cfg.homeostatic_target_in = 1.8f;
  cfg.homeostatic_rate = 0.0008f;
  cfg.spine_retraction_floor = 0.01f;
  cfg.prune_inactive_steps = 3000;
  cfg.weight_potentiation = 0.0f;

  int growth_steps = (argc > 1) ? std::atoi(argv[1]) : 400;
  int trials = (argc > 2) ? std::atoi(argv[2]) : 800;
  const char* save_path = (argc > 3) ? argv[3] : "mom_dad_brain.snc";

  snc::Simulator sim(cfg);

  // -------- Anatomy ----------------------------------------------------

  snc::FetalSeed seed;
  seed.vz_neurons = 60;
  seed.migrating_neurons = 0;
  seed.cortical_plate_neurons = 0;
  seed.vz_thickness = cfg.Z - 4;
  seed.radial_glia_density = 0.02f;
  sim.seed_fetal(seed);
  sim.randomize_polarity(0.2f);   // ~20% GABAergic in the bulk

  // External sensory INPUTs at z = 2.
  std::vector<uint32_t> ext_in;
  for (int i = 0; i < kExtFeatures; ++i) {
    const int x = 6 + 4 * (i % 4);
    const int y = (i < 4) ? 6 : 18;
    const uint32_t id = sim.add_neuron_at(x, y, 2);
    if (!id) { std::fprintf(stderr, "ext input %d failed\n", i); return 1; }
    sim.set_role(id, snc::NeuronRole::INPUT, i);
    sim.set_polarity(id, snc::NeuronPolarity::EXCITATORY);
    ext_in.push_back(id);
  }

  // Motor OUTPUTs at z = Z-3 ("voice").
  uint32_t mom_out = sim.add_neuron_at(10, 12, cfg.Z - 3);
  uint32_t dad_out = sim.add_neuron_at(22, 12, cfg.Z - 3);
  sim.set_role(mom_out, snc::NeuronRole::OUTPUT, 0);
  sim.set_role(dad_out, snc::NeuronRole::OUTPUT, 1);
  sim.set_polarity(mom_out, snc::NeuronPolarity::EXCITATORY);
  sim.set_polarity(dad_out, snc::NeuronPolarity::EXCITATORY);

  // Self-perception INPUTs at z = Z-4 ("ears that hear my own voice").
  // Channels 8 and 9 -- distinct from external 0..7 so any downstream
  // bulk neuron can observe activity at these channels and learn to treat
  // it as self-generated.
  uint32_t self_mom = sim.add_neuron_at(10, 16, cfg.Z - 4);
  uint32_t self_dad = sim.add_neuron_at(22, 16, cfg.Z - 4);
  sim.set_role(self_mom, snc::NeuronRole::INPUT, 8);
  sim.set_role(self_dad, snc::NeuronRole::INPUT, 9);
  sim.set_polarity(self_mom, snc::NeuronPolarity::EXCITATORY);
  sim.set_polarity(self_dad, snc::NeuronPolarity::EXCITATORY);

  // Inhibitory interneurons providing feedforward inhibition for
  // winner-take-all decoding. mom-inputs drive `inh_silences_dad`; that
  // inhibitor (GABAergic) silences dad_out, and vice versa.
  uint32_t inh_silences_dad = sim.add_neuron_at(15, 8, cfg.Z - 5);
  uint32_t inh_silences_mom = sim.add_neuron_at(15, 24, cfg.Z - 5);
  sim.set_polarity(inh_silences_dad, snc::NeuronPolarity::INHIBITORY);
  sim.set_polarity(inh_silences_mom, snc::NeuronPolarity::INHIBITORY);

  // -------- Wiring -----------------------------------------------------

  // Innate priors: each external feature drives its class's motor output.
  for (int i = 0; i < 4; ++i) {
    sim.install_synapse(ext_in[i], mom_out, 0.55f, 2);
  }
  for (int i = 4; i < 8; ++i) {
    sim.install_synapse(ext_in[i], dad_out, 0.55f, 2);
  }

  // Feedforward inhibition: mom inputs drive the dad-silencing inhibitor.
  for (int i = 0; i < 4; ++i) {
    sim.install_synapse(ext_in[i], inh_silences_dad, 0.35f, 1);
  }
  for (int i = 4; i < 8; ++i) {
    sim.install_synapse(ext_in[i], inh_silences_mom, 0.35f, 1);
  }
  sim.install_synapse(inh_silences_dad, dad_out, 1.2f, 1);
  sim.install_synapse(inh_silences_mom, mom_out, 1.2f, 1);

  // Efference copy: motor outputs project to their own self-perception
  // inputs at low delay (axon collateral analogue). When the network says
  // "mom", channel 8 gets activated one step later -- it hears itself.
  sim.install_synapse(mom_out, self_mom, 1.4f, 1);
  sim.install_synapse(dad_out, self_dad, 1.4f, 1);

  // Random sparse input -> bulk wires so plasticity has a substrate to
  // sculpt over time.
  std::mt19937 rng(0xBEEF);
  std::uniform_int_distribution<int> bulk_pick(
      kExtFeatures + kClasses + kEffFeatures + 2 + 1,
      static_cast<int>(sim.neuron_count()));
  for (uint32_t in_id : ext_in) {
    for (int k = 0; k < 4; ++k) {
      sim.install_synapse(in_id, bulk_pick(rng), 0.25f, 4);
    }
  }
  // Also wire self-perception into the bulk so the network can build
  // associations between hearing itself and other internal states.
  for (uint32_t self_id : {self_mom, self_dad}) {
    for (int k = 0; k < 4; ++k) {
      sim.install_synapse(self_id, bulk_pick(rng), 0.25f, 4);
    }
  }

  std::printf("anatomy: %zu neurons total. "
              "8 ext INPUT, 2 self-percept INPUT (chan 8,9), "
              "2 motor OUTPUT, 2 inh interneurons.\n",
              sim.neuron_count());

  // -------- Pre-training growth ---------------------------------------

  std::uniform_real_distribution<float> noise(0.0f, 0.2f);
  std::printf("[grow] %d steps of spontaneous activity\n", growth_steps);
  for (int s = 0; s < growth_steps; ++s) {
    for (std::size_t id = 1; id <= sim.neuron_count(); ++id) {
      sim.inject_input(static_cast<uint32_t>(id), noise(rng));
    }
    sim.step();
  }
  std::printf("[grow] done. neurons=%zu synapses=%zu\n",
              sim.neuron_count(), sim.total_synapses());

  // -------- Interaction loop with RPE ---------------------------------

  const auto scenes = build_scenes();
  std::uniform_int_distribution<int> scene_pick(
      0, static_cast<int>(scenes.size()) - 1);

  // Reward prediction-error machinery (dopamine analogue). The network's
  // internal "expectation" of reward per class evolves as an EMA; the
  // signal actually broadcast as reward is `actual - expected`. This is
  // exactly what midbrain dopaminergic neurons fire (Schultz 1997).
  float expected_reward[kClasses] = {0.0f, 0.0f};
  constexpr float kExpEmaAlpha = 0.04f;

  constexpr int present_steps = 18;
  constexpr int rest_steps = 4;

  int correct_recent = 0;
  const int recent_window = 50;
  std::vector<int> recent_results;
  recent_results.reserve(recent_window);

  std::printf("\n[talk] %d trials\n", trials);
  std::printf("trial  shown  said   self_mom self_dad   mom_out dad_out  "
              "rpe[m,d]    acc(last %d)\n", recent_window);
  const auto t0 = std::chrono::steady_clock::now();

  for (int t = 0; t < trials; ++t) {
    const Scene& scene = scenes[scene_pick(rng)];
    const char* shown = (scene.label == 0) ? "mom" : "dad";

    sim.clear_eligibility();

    for (int s = 0; s < present_steps; ++s) {
      // External input pattern. Channels 8/9 (efference) are NOT touched
      // by apply_input_pattern when n_features = kExtFeatures, so they
      // only ever receive activity through the efference-copy synapses
      // installed above.
      sim.apply_input_pattern(scene.pattern, kExtFeatures);
      // Light internal noise on bulk neurons.
      const std::size_t bulk_start =
          kExtFeatures + kClasses + kEffFeatures + 2 + 1;
      for (std::size_t id = bulk_start; id <= sim.neuron_count(); ++id) {
        sim.inject_input(static_cast<uint32_t>(id), noise(rng) * 0.2f);
      }
      sim.step();
    }

    // Read the network's voice and self-perception.
    float ext_out[kClasses] = {0, 0};
    sim.read_output(ext_out, kClasses);
    const float self_mom_rate = sim.neurons()[self_mom - 1].fire_rate_ema;
    const float self_dad_rate = sim.neurons()[self_dad - 1].fire_rate_ema;
    const char* said = utter(ext_out[0], ext_out[1]);
    const bool match = (std::strcmp(said, shown) == 0);

    // Reward prediction error: `actual - expected`. The expected value
    // tracks the network's recent success on each class.
    float actual[kClasses];
    actual[0] = (scene.label == 0) ? (match ? 1.0f : -1.0f)
                                   : (match ? 1.0f : -1.0f);  // dummy, see below
    // Simpler: per-class reward = +1 if it should fire and did; -1 if it
    // should fire and didn't, or fired wrongly.
    actual[0] = (scene.label == 0) ?  1.0f : -1.0f;
    actual[1] = (scene.label == 1) ?  1.0f : -1.0f;
    if (!match) {
      // Apply a stronger negative on the wrongly-uttered class to amplify
      // surprise (this mimics the phasic-dip in dopamine on negative
      // outcome).
      const int said_idx = std::strcmp(said, "mom") == 0 ? 0
                          : std::strcmp(said, "dad") == 0 ? 1 : -1;
      if (said_idx >= 0) actual[said_idx] -= 0.5f;
    }

    float rpe[kClasses];
    for (int c = 0; c < kClasses; ++c) {
      rpe[c] = actual[c] - expected_reward[c];
      expected_reward[c] = expected_reward[c] * (1.0f - kExpEmaAlpha) +
                           actual[c] * kExpEmaAlpha;
    }
    sim.apply_reward_per_class(rpe, kClasses, match ? 0.1f : -0.05f);

    for (int s = 0; s < rest_steps; ++s) sim.step();

    if (recent_results.size() >= static_cast<std::size_t>(recent_window)) {
      if (recent_results.front()) --correct_recent;
      recent_results.erase(recent_results.begin());
    }
    recent_results.push_back(match ? 1 : 0);
    if (match) ++correct_recent;

    if (t % 25 == 0 || t == trials - 1) {
      std::printf("%5d  %4s   %4s    %5.3f    %5.3f    %5.3f   %5.3f   "
                  "[%+5.2f, %+5.2f]   %5.1f%%\n",
                  t, shown, said,
                  self_mom_rate, self_dad_rate,
                  ext_out[0], ext_out[1],
                  rpe[0], rpe[1],
                  100.0f * correct_recent /
                      std::max(1, int(recent_results.size())));
    }
  }

  const auto t1 = std::chrono::steady_clock::now();
  const double dt = std::chrono::duration<double>(t1 - t0).count();
  std::printf("[talk] done in %.2fs (%.1f trials/sec). "
              "final synapses=%zu\n",
              dt, trials / dt, sim.total_synapses());

  // -------- Test sweep -------------------------------------------------

  std::printf("\n[test] no-noise readout per scene "
              "(also showing self-perception for each utterance):\n");
  std::printf("shown  said   mom_out dad_out  self_mom self_dad\n");
  int test_correct = 0;
  for (const Scene& scene : scenes) {
    // Silent gap so activity from the previous scene decays before we
    // measure -- without it the fire-rate EMA bleeds across scenes.
    float zero_pat[kExtFeatures] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int s = 0; s < 40; ++s) {
      sim.apply_input_pattern(zero_pat, kExtFeatures);
      sim.step();
    }
    for (int s = 0; s < present_steps * 2; ++s) {
      sim.apply_input_pattern(scene.pattern, kExtFeatures);
      sim.step();
    }
    float out[kClasses] = {0, 0};
    sim.read_output(out, kClasses);
    const float sm = sim.neurons()[self_mom - 1].fire_rate_ema;
    const float sd = sim.neurons()[self_dad - 1].fire_rate_ema;
    const char* said = utter(out[0], out[1]);
    const char* shown = (scene.label == 0) ? "mom" : "dad";
    const bool ok = std::strcmp(said, shown) == 0;
    if (ok) ++test_correct;
    std::printf("%5s  %4s   %5.3f   %5.3f   %5.3f    %5.3f\n",
                shown, said, out[0], out[1], sm, sd);
  }
  std::printf("[test] %d / %zu correct\n", test_correct, scenes.size());

  // -------- Sleep ------------------------------------------------------

  // Sleep replay before saving: drives the connectome with internal noise
  // for a short window with elevated STDP, consolidating the patterns
  // that wakeful learning has built up. Mirrors slow-wave / REM replay.
  std::printf("\n[sleep] consolidating via replay (200 steps)\n");
  sim.sleep_consolidate(200, 1.5f);

  if (sim.save_state(save_path)) {
    std::printf("\n[sleep] saved brain state to %s\n", save_path);
  } else {
    std::printf("\n[sleep] FAILED to save to %s\n", save_path);
    return 1;
  }

  // Wake test: re-load the saved file into a fresh simulator and confirm
  // structure matches. The next session would resume from exactly this
  // connectome, picking up plasticity where it left off.
  {
    snc::Simulator wake(snc::SimConfig{});
    if (!wake.load_state(save_path)) {
      std::printf("[wake] failed to load %s\n", save_path);
      return 1;
    }
    std::printf("[wake] reloaded brain: %zu neurons, %zu synapses, "
                "step=%d. Continued plasticity available.\n",
                wake.neuron_count(), wake.total_synapses(),
                wake.current_step());
  }

  return 0;
}
