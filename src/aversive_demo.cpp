// Innate-aversive learning demo.
//
// "The baby touches the kettle once and learns not to."
//
// The network has four sensory channels and one motor output ("touch
// the object"). Channel 3 carries an innately aversive cue (heat, pain,
// sharp edge -- in real biology, nociceptor signals). The wiring is
// genetic, not learned:
//
//   inputs[0..2] -> motor             (excitatory priors: see object,
//                                      reach for it)
//   inputs[3]    -> amygdala          (strong, fast: pain detector)
//   amygdala     -> motor             (inhibitory_SST: silences the
//                                      reach when amygdala fires)
//
// On top of this innate scaffold, plasticity happens: every trial in
// which the amygdala fires *and* the motor output fires (the baby
// touched the danger before the inhibitory pathway could fully gate
// it) triggers a global aversive plasticity event. After enough such
// trials, the synapses that contributed to the unwanted reach decay,
// and the network learns to *stop* reaching when its own amygdala says
// "no" -- without an external supervisor labelling the trials.
//
// What this demo demonstrates:
//   - The simulator's `apply_aversive` is a generic mechanism the demo
//     can drive from any local observation. We use the firing of an
//     anatomically-defined amygdala neuron, but the same hook can be
//     wired to any innate "this is bad" detector.
//   - Bounded credulity: the learning is *self-supervised* through the
//     innate aversive circuit. The network builds a value judgement
//     ("don't touch hot things") from its own internal signal, not
//     from a teacher saying which trials are right or wrong.

#include "simulator.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kFeatures = 4;       // 0..2 normal, 3 = innate danger
constexpr int kClasses  = 1;       // single motor output
constexpr float kMotorFireThresh = 0.05f;

struct Pattern {
  float v[kFeatures];
  bool danger;
};

std::vector<Pattern> build_patterns() {
  return {
    // Safe patterns -- "see toy", "see ball", combinations.
    {{1, 0, 0, 0}, false},
    {{0, 1, 0, 0}, false},
    {{1, 1, 0, 0}, false},
    {{0, 0, 1, 0}, false},
    {{1, 0, 1, 0}, false},
    {{0, 1, 1, 0}, false},
    // Danger patterns -- channel 3 active.
    {{1, 0, 0, 1}, true},
    {{0, 1, 0, 1}, true},
    {{1, 1, 0, 1}, true},
    {{0, 0, 1, 1}, true},
    {{1, 0, 1, 1}, true},
  };
}

}  // namespace

int main(int argc, char** argv) {
  snc::SimConfig cfg;
  cfg.X = 32;
  cfg.Y = 32;
  cfg.Z = 32;
  cfg.region_size = 8;
  cfg.fire_threshold = 0.45f;
  cfg.weight_max = 1.5f;
  cfg.initial_weight = 0.3f;
  cfg.input_drive_strength = 1.4f;
  cfg.eligibility_decay = 0.9f;
  cfg.eligibility_potentiation = 0.6f;
  cfg.reward_lr = 0.05f;
  cfg.aversive_amplification = 3.0f;   // strong negativity bias
  cfg.stdp_a_ltp = 0.02f;
  cfg.stdp_a_ltd = 0.02f;
  cfg.spine_retraction_floor = 0.01f;
  cfg.prune_inactive_steps = 4000;
  cfg.weight_potentiation = 0.0f;
  cfg.homeostatic_rate = 0.0f;
  cfg.heterosynaptic_damp = 0.0f;
  cfg.bcm_baseline_alpha = 0.0f;
  // Active dendrites with biologically-grounded temporal integration.
  // Default cfg values cover branch 1 (the synaptogenesis-default
  // branch); the demo then *overrides* the innate-prior branch 0 of
  // each special neuron via set_branch_threshold / passive_gain so
  // the labelled-line and bulk-noise compartments behave differently:
  //
  //   branch 0 (priors) -- threshold low (1.0), passive_gain 0.3,
  //                         decay 0.5: a single weak prior accumulates
  //                         over a few sustained steps and eventually
  //                         passes the spike threshold OR drives the
  //                         soma sub-threshold; both paths produce
  //                         firing for safe single-feature patterns.
  //   branch 1 (sprouted) -- threshold inf (no spike), passive_gain 0
  //                         (no leak): bulk noise is structurally
  //                         isolated from the soma, can never produce
  //                         spurious firing.
  cfg.dendritic_threshold = 1.0e9f;     // default: never spike
  cfg.dendritic_spike_amplitude = 1.0f;
  cfg.dendritic_passive_gain = 0.0f;    // default: no leak
  cfg.dendritic_decay = 0.5f;            // mild temporal integration
  cfg.synaptogenesis_default_branch = 1;

  int trials = (argc > 1) ? std::atoi(argv[1]) : 200;

  snc::Simulator sim(cfg);

  // Tiny fetal seed -- this demo doesn't need a big bulk; the wiring is
  // mostly hand-installed.
  snc::FetalSeed seed;
  seed.vz_neurons = 24;
  seed.vz_thickness = cfg.Z - 5;
  seed.brainstem_neurons = 4;
  seed.thalamic_relay_neurons = 6;
  seed.aversive_nucleus_neurons = 2;
  sim.seed_fetal(seed);

  // Sensory inputs at the cortical floor.
  std::vector<uint32_t> inputs;
  for (int i = 0; i < kFeatures; ++i) {
    const uint32_t id = sim.add_neuron_at(8 + 4 * i, 16, 2);
    if (!id) { std::fprintf(stderr, "input %d failed\n", i); return 1; }
    sim.set_role(id, snc::NeuronRole::INPUT, i);
    sim.set_polarity(id, snc::NeuronPolarity::EXCITATORY);
    inputs.push_back(id);
  }

  // Motor output. Two dendritic branches: priors (and the inhibitory
  // amygdala wire) on branch 0; sprouted bulk on branch 1.
  const uint32_t motor = sim.add_neuron_at(16, 16, cfg.Z - 3);
  sim.set_role(motor, snc::NeuronRole::OUTPUT, 0);
  sim.set_polarity(motor, snc::NeuronPolarity::EXCITATORY);
  sim.set_branches(motor, 2);
  // Branch 0 (priors): low threshold, modest passive leak, temporal
  // integration. Even single-feature safe patterns build up activity.
  sim.set_branch_threshold(motor, 0, 1.0f);
  sim.set_branch_passive_gain(motor, 0, 0.3f);
  // Branch 1 (sprouted bulk): unreachable threshold, zero leak so
  // background activity never reaches the soma.
  sim.set_branch_threshold(motor, 1, 1.0e9f);
  sim.set_branch_passive_gain(motor, 1, 0.0f);

  // Amygdala-analogue: GABAergic SST cell sitting between sensors and
  // the motor output. Fires whenever the danger channel is active.
  // Two branches as well -- the pain afferent on branch 0, sprouted
  // bulk on branch 1.
  const uint32_t amyg = sim.add_neuron_at(16, 16, cfg.Z / 2);
  sim.set_polarity(amyg, snc::NeuronPolarity::INHIBITORY_SST);
  sim.set_branches(amyg, 2);
  sim.set_branch_threshold(amyg, 0, 0.8f);     // pain afferent -> spike
  sim.set_branch_passive_gain(amyg, 0, 0.0f);  // no leak from this branch
  sim.set_branch_threshold(amyg, 1, 1.0e9f);   // bulk noise never spikes
  sim.set_branch_passive_gain(amyg, 1, 0.0f);

  // Innate wiring (the "DNA" prior for this circuit):
  //   1. Safe-feature inputs prime the motor output (see ball -> reach).
  //      Each prior has weight 0.45, well below branch 0's threshold of
  //      1.0; sustained delivery + dendritic_decay 0.5 lets a single
  //      prior asymptote to ~0.9 -- just below threshold so the cell
  //      contributes via the passive path. Two priors firing summed
  //      together comfortably crosses threshold for an NMDA-style spike.
  for (int i = 0; i < 3; ++i) {
    sim.install_synapse(inputs[i], motor, 0.45f, 4, /*branch=*/0);
  }
  //   2. Pain channel drives the amygdala, fast and strong.
  sim.install_synapse(inputs[3], amyg, 1.0f, 1, /*branch=*/0);
  //   3. Amygdala silences the motor (inhibitory wire onto branch 0).
  sim.install_synapse(amyg, motor, 2.0f, 1, /*branch=*/0);

  std::printf("anatomy: %zu neurons (incl. 4 sensors, 1 motor, "
              "1 amygdala-SST). innate wiring locks in the reflex; "
              "plasticity tunes the rest.\n", sim.neuron_count());

  // Build a "do not drive with bulk noise" set so the inputs / motor /
  // amygdala only respond to their proper afferents.
  std::vector<bool> skip_noise(sim.neuron_count() + 1, false);
  for (uint32_t id : inputs) skip_noise[id] = true;
  skip_noise[motor] = true;
  skip_noise[amyg] = true;

  // ------------------------------------------------------------------
  //                            INTERACTION
  // ------------------------------------------------------------------
  std::mt19937 rng(0xCAFEBABE);
  std::uniform_real_distribution<float> noise(0.0f, 0.04f);
  const auto patterns = build_patterns();
  std::uniform_int_distribution<int> pat_pick(
      0, static_cast<int>(patterns.size()) - 1);

  constexpr int present_steps = 16;
  constexpr int rest_steps = 4;
  const int recent_window = 30;
  std::vector<int> rolling;
  rolling.reserve(recent_window);
  int rolling_correct = 0;

  std::printf("\ntrial  pattern_danger  amyg_fired  motor_fired  "
              "outcome           rolling-acc(%d)\n", recent_window);
  const auto t0 = std::chrono::steady_clock::now();

  for (int t = 0; t < trials; ++t) {
    const Pattern& p = patterns[pat_pick(rng)];
    sim.clear_eligibility();

    bool amyg_fired = false;
    for (int s = 0; s < present_steps; ++s) {
      sim.apply_input_pattern(p.v, kFeatures);
      // Light internal noise so the bulk has spontaneous activity.
      // Skip the inputs / motor / amygdala -- they should only be
      // driven by their proper afferents, not by bulk noise.
      for (std::size_t id = 1; id <= sim.neuron_count(); ++id) {
        if (id < skip_noise.size() && skip_noise[id]) continue;
        sim.inject_input(static_cast<uint32_t>(id), noise(rng));
      }
      sim.step();
      if (sim.neurons()[amyg - 1].fired_this_step) amyg_fired = true;
    }

    float out[kClasses] = {0};
    sim.read_output(out, kClasses);
    const bool motor_fired = out[0] > kMotorFireThresh;

    // Self-supervised valuation:
    //   - Amygdala fired AND motor fired   -> "we touched something hot"
    //                                          -> aversive plasticity
    //                                             (intensity scales with
    //                                              how strongly motor fired)
    //   - Amygdala did not fire AND motor   -> safe action: gentle reward
    //   - Amygdala fired AND motor did not  -> we correctly avoided
    //                                          -> small reward (the
    //                                              avoidance pathway
    //                                              should be reinforced)
    //   - Both silent                       -> nothing to learn
    if (amyg_fired && motor_fired) {
      sim.apply_aversive(out[0]);
    } else if (!amyg_fired && motor_fired) {
      float r[kClasses] = { 0.4f };
      sim.apply_reward_per_class(r, kClasses, 0.05f);
    } else if (amyg_fired && !motor_fired) {
      float r[kClasses] = { 0.25f };
      sim.apply_reward_per_class(r, kClasses, 0.05f);
    }

    for (int s = 0; s < rest_steps; ++s) sim.step();

    const bool correct =
        (p.danger && !motor_fired) || (!p.danger && motor_fired);
    if (rolling.size() >= static_cast<std::size_t>(recent_window)) {
      if (rolling.front()) --rolling_correct;
      rolling.erase(rolling.begin());
    }
    rolling.push_back(correct ? 1 : 0);
    if (correct) ++rolling_correct;

    if (t % 10 == 0 || t == trials - 1) {
      const char* outcome =
          (amyg_fired && motor_fired)    ? "TOUCHED-DANGER (aversive)"
        : (!amyg_fired && motor_fired)   ? "safe-reach (reward)"
        : (amyg_fired && !motor_fired)   ? "avoided (reward)"
                                          : "no-action";
      std::printf("%5d        %d           %d            %d         "
                  "%-26s %d/%zu\n",
                  t, p.danger ? 1 : 0, amyg_fired ? 1 : 0,
                  motor_fired ? 1 : 0, outcome,
                  rolling_correct, rolling.size());
    }
  }

  const auto t1 = std::chrono::steady_clock::now();
  const double dt = std::chrono::duration<double>(t1 - t0).count();
  std::printf("ran %d trials in %.2fs (%.1f trials/sec). "
              "final synapses=%zu structural-neurons=%d\n",
              trials, dt, trials / dt, sim.total_synapses(),
              sim.count_structural_neurons());

  // Probe: present each canonical pattern with no learning, no noise.
  std::printf("\n[probe] no-noise sweep:\n");
  std::printf("  pattern (danger?)  motor_rate  amyg_fired_during  "
              "decision\n");
  int probe_correct = 0;
  for (const Pattern& p : patterns) {
    float zero[kFeatures] = {0};
    for (int s = 0; s < 25; ++s) {
      sim.apply_input_pattern(zero, kFeatures);
      sim.step();
    }
    bool amyg_during = false;
    for (int s = 0; s < present_steps * 2; ++s) {
      sim.apply_input_pattern(p.v, kFeatures);
      sim.step();
      if (sim.neurons()[amyg - 1].fired_this_step) amyg_during = true;
    }
    float out[kClasses] = {0};
    sim.read_output(out, kClasses);
    const bool motor_fired = out[0] > kMotorFireThresh;
    const bool correct =
        (p.danger && !motor_fired) || (!p.danger && motor_fired);
    if (correct) ++probe_correct;
    std::printf("  (%d %d %d %d) %-7s   %5.3f       %d                 %s\n",
                int(p.v[0]), int(p.v[1]), int(p.v[2]), int(p.v[3]),
                p.danger ? "DANGER" : "safe",
                out[0], amyg_during ? 1 : 0,
                motor_fired ? "REACH" : "withhold");
  }
  std::printf("[probe] %d / %zu correct\n",
              probe_correct, patterns.size());

  return 0;
}
