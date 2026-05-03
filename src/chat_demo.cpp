// snc_chat -- conversational teaching interface for the simulator.
//
// A small REPL that reads commands from stdin and presents stimuli /
// records responses one episode at a time. Claude Code (or a human)
// pipes a teaching script in, observes the network's responses on
// stdout, and decides what to teach next. The brain state persists
// across commands within one session and can be saved / reloaded
// across sessions.
//
// Built-in vocabulary (6 concepts):
//   mom, dad, hi, bye, yes, no
// Each concept owns four sensory feature channels and one motor output;
// efference copies wire each motor back to a self-perception input so
// the network "hears" itself speak.
//
// Commands (one per line):
//   help                   -- list commands
//   concepts               -- list available concepts
//   babble <N>             -- N random motor firings (no reward)
//   show <concept>         -- present sensory pattern, observe
//   teach <concept>        -- present + prime matching motor
//   correct                -- last episode's target was right; reward
//   wrong                  -- last episode's response was wrong; aversive
//   status                 -- print brain stats
//   save <path>            -- persist brain
//   load <path>            -- reload brain (replaces current state)
//   quit                   -- exit
//
// Typical teaching loop:
//   echo "babble 30
//   teach mom
//   correct
//   show mom
//   correct" | snc_chat

#include "simulator.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kClasses = 6;
constexpr int kFeatPerClass = 4;
constexpr int kExtFeatures = kClasses * kFeatPerClass;       // 24
constexpr int kEffFeatures = kClasses;                       // 6
constexpr int kAllFeatures = kExtFeatures + kEffFeatures;    // 30

const char* kWords[kClasses] = {
    "mom", "dad", "hi", "bye", "yes", "no"
};

int word_index(const std::string& w) {
  for (int i = 0; i < kClasses; ++i) {
    if (w == kWords[i]) return i;
  }
  return -1;
}

// Build a one-hot-ish stimulus pattern: feature[c*4 .. c*4+3] = (1,1,0,0).
// The 2-of-4 sub-pattern guarantees the prior dendritic-spike threshold
// (0.8) is comfortably exceeded.
void make_pattern(int c, float* out) {
  for (int i = 0; i < kAllFeatures; ++i) out[i] = 0.0f;
  if (c < 0 || c >= kClasses) return;
  out[c * kFeatPerClass + 0] = 1.0f;
  out[c * kFeatPerClass + 1] = 1.0f;
}

int argmax_of(const float* v, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i) if (v[i] > v[best]) best = i;
  return best;
}

const char* utter(const float* rates) {
  int top = argmax_of(rates, kClasses);
  if (rates[top] < 0.05f) return "...";
  float second = 0.0f;
  for (int i = 0; i < kClasses; ++i) {
    if (i != top && rates[i] > second) second = rates[i];
  }
  if (rates[top] - second < 0.04f) return "...";
  return kWords[top];
}

snc::SimConfig make_config() {
  snc::SimConfig cfg;
  cfg.X = 48; cfg.Y = 48; cfg.Z = 48;
  cfg.region_size = 8;
  cfg.fire_threshold = 0.45f;
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
  cfg.weight_potentiation = 0.0f;
  cfg.refractory_steps = 3;
  cfg.release_probability = 0.85f;
  // Homeostatic / heterosynaptic / BCM kept off here. The chat brain
  // is small (one cluster of priors per concept) and the homeostatic
  // pull toward target_in would shrink the labelled-line priors faster
  // than STDP can rebuild them under the chat-volume of activity.
  cfg.homeostatic_rate = 0.0f;
  cfg.heterosynaptic_damp = 0.0f;
  cfg.bcm_baseline_alpha = 0.0f;
  cfg.dendritic_threshold = 1.0e9f;
  cfg.dendritic_passive_gain = 1.0f;
  cfg.dendritic_decay = 0.0f;
  cfg.synaptogenesis_default_branch = 1;
  return cfg;
}

struct Brain {
  snc::Simulator sim;
  std::vector<uint32_t> ext_in;
  std::vector<uint32_t> motors;
  std::vector<uint32_t> selfs;
  std::vector<uint32_t> inhibitors;
  std::vector<bool> skip_noise;
  std::mt19937 rng;
  // Last episode bookkeeping (for `correct` / `wrong`).
  int last_target = -1;
  int last_said = -1;
  bool last_match = false;
  float last_rates[kClasses] = {0};

  explicit Brain(snc::SimConfig cfg)
      : sim(cfg), rng(0xCC0DE) {}
};

// Place neurons + install innate priors. Mirrors vocab_demo but with
// kClasses=6.
void build_anatomy(Brain& b) {
  snc::Simulator& sim = b.sim;
  const auto& cfg = sim.energy().rX();  // dummy use
  (void)cfg;
  const int Z = sim.grid().Z();
  const int Y = sim.grid().Y();

  // Sparse fetal seed (no glia, no migrating, no CP -- chat builds its
  // bulk by sprouting during babble).
  snc::FetalSeed seed;
  seed.vz_neurons = 100;
  seed.migrating_neurons = 0;
  seed.cortical_plate_neurons = 0;
  seed.vz_thickness = Z - 5;
  seed.radial_glia_density = 0.0f;
  seed.frac_pv = 0.14f;
  seed.frac_sst = 0.04f;
  seed.frac_vip = 0.02f;
  seed.brainstem_neurons = 8;
  seed.thalamic_relay_neurons = 12;
  seed.aversive_nucleus_neurons = 4;
  sim.seed_fetal(seed);

  // External sensory inputs: 24 channels, arranged in 6 rows of 4.
  b.ext_in.reserve(kExtFeatures);
  for (int c = 0; c < kClasses; ++c) {
    for (int f = 0; f < kFeatPerClass; ++f) {
      const int channel = c * kFeatPerClass + f;
      const int x = 4 + f * 4;
      const int y = 4 + c * 6;
      const uint32_t id = sim.add_neuron_at(x, y, 2);
      if (!id) {
        std::fprintf(stderr, "ext input %d failed at (%d,%d,2)\n",
                     channel, x, y);
        std::exit(1);
      }
      sim.set_role(id, snc::NeuronRole::INPUT, channel);
      sim.set_polarity(id, snc::NeuronPolarity::EXCITATORY);
      b.ext_in.push_back(id);
    }
  }

  // Motor outputs + self-perception + lateral inhibitors.
  b.motors.reserve(kClasses);
  b.selfs.reserve(kClasses);
  b.inhibitors.reserve(kClasses);
  for (int c = 0; c < kClasses; ++c) {
    const int xm = 6 + c * 6;
    const int ym = Y / 2;
    const uint32_t m = sim.add_neuron_at(xm, ym, Z - 3);
    sim.set_role(m, snc::NeuronRole::OUTPUT, c);
    sim.set_polarity(m, snc::NeuronPolarity::EXCITATORY);
    sim.set_branches(m, 2);
    sim.set_branch_threshold(m, 0, 0.8f);
    sim.set_branch_passive_gain(m, 0, 0.3f);
    sim.set_branch_threshold(m, 1, 1.0e9f);
    sim.set_branch_passive_gain(m, 1, 0.0f);
    b.motors.push_back(m);

    const uint32_t s = sim.add_neuron_at(xm, ym + 4, Z - 4);
    sim.set_role(s, snc::NeuronRole::INPUT, kExtFeatures + c);
    sim.set_polarity(s, snc::NeuronPolarity::EXCITATORY);
    sim.set_branches(s, 2);
    sim.set_branch_threshold(s, 0, 1.0e9f);
    sim.set_branch_passive_gain(s, 0, 1.0f);
    sim.set_branch_threshold(s, 1, 1.0e9f);
    sim.set_branch_passive_gain(s, 1, 0.0f);
    b.selfs.push_back(s);

    const uint32_t inh = sim.add_neuron_at(xm + 1, ym, Z - 4);
    sim.set_polarity(inh, snc::NeuronPolarity::INHIBITORY);
    sim.set_branches(inh, 2);
    sim.set_branch_threshold(inh, 0, 1.0e9f);
    sim.set_branch_passive_gain(inh, 0, 1.0f);
    sim.set_branch_threshold(inh, 1, 1.0e9f);
    sim.set_branch_passive_gain(inh, 1, 0.0f);
    b.inhibitors.push_back(inh);
  }

  // Innate priors / efference / lateral inhibition (all permanent).
  for (int c = 0; c < kClasses; ++c) {
    for (int f = 0; f < kFeatPerClass; ++f) {
      sim.install_synapse(b.ext_in[c * kFeatPerClass + f],
                          b.motors[c], 0.55f, 4, 0, 1.0f);
    }
  }
  for (int c = 0; c < kClasses; ++c) {
    sim.install_synapse(b.motors[c], b.selfs[c], 1.4f, 1, 0, 1.0f);
    sim.install_synapse(b.selfs[c], b.motors[c], 0.25f, 2, 1, 0.7f);
  }
  for (int c = 0; c < kClasses; ++c) {
    sim.install_synapse(b.motors[c], b.inhibitors[c], 0.7f, 1, 0, 1.0f);
    for (int j = 0; j < kClasses; ++j) {
      if (j == c) continue;
      sim.install_synapse(b.inhibitors[c], b.motors[j], 1.4f, 1, 0, 1.0f);
    }
  }

  // Skip-set so noise injection bypasses the labelled-line cells.
  b.skip_noise.assign(sim.neuron_count() + 1, false);
  for (uint32_t id : b.ext_in)    b.skip_noise[id] = true;
  for (uint32_t id : b.motors)    b.skip_noise[id] = true;
  for (uint32_t id : b.selfs)     b.skip_noise[id] = true;
  for (uint32_t id : b.inhibitors) b.skip_noise[id] = true;
}

void inject_internal_noise(Brain& b) {
  static thread_local std::uniform_real_distribution<float> noise(0.0f, 0.1f);
  for (std::size_t id = 1; id <= b.sim.neuron_count(); ++id) {
    if (id < b.skip_noise.size() && b.skip_noise[id]) continue;
    b.sim.inject_input(static_cast<uint32_t>(id), noise(b.rng));
  }
}

void run_present(Brain& b, const float* pattern, int prime_target,
                  float prime_strength, int n_steps) {
  for (int s = 0; s < n_steps; ++s) {
    b.sim.apply_input_pattern(pattern, kAllFeatures);
    if (prime_target >= 0 && prime_target < kClasses && prime_strength > 0) {
      b.sim.inject_input(b.motors[prime_target], prime_strength);
    }
    inject_internal_noise(b);
    b.sim.step();
  }
}

// Commands --------------------------------------------------------------

void cmd_help() {
  std::printf(
      "commands:\n"
      "  help                   list commands\n"
      "  concepts               list available concepts\n"
      "  babble <N>             N random motor firings (no reward)\n"
      "  show <concept>         present sensory pattern, observe\n"
      "  teach <concept>        present + prime matching motor\n"
      "  correct                last response was right; reward\n"
      "  wrong                  last response was wrong; aversive\n"
      "  status                 brain stats\n"
      "  save <path>            persist brain\n"
      "  load <path>            reload brain (replaces state)\n"
      "  quit                   exit\n");
}

void cmd_concepts() {
  std::printf("concepts (%d):\n", kClasses);
  for (int i = 0; i < kClasses; ++i) {
    std::printf("  %s\n", kWords[i]);
  }
}

void cmd_babble(Brain& b, int n) {
  std::uniform_int_distribution<int> coin(0, kClasses - 1);
  for (int t = 0; t < n; ++t) {
    const int target = coin(b.rng);
    for (int s = 0; s < 12; ++s) {
      b.sim.inject_input(b.motors[target], 1.6f);
      inject_internal_noise(b);
      b.sim.step();
    }
    for (int s = 0; s < 4; ++s) b.sim.step();
  }
  std::printf("[babble] %d trials done. step=%d, synapses=%zu\n",
              n, b.sim.current_step(), b.sim.total_synapses());
}

void cmd_show(Brain& b, const std::string& concept) {
  const int c = word_index(concept);
  if (c < 0) { std::printf("unknown concept '%s'\n", concept.c_str()); return; }
  b.sim.clear_eligibility();
  b.sim.reset_dynamics();
  // Silent gap so the previous response decays.
  float zero[kAllFeatures] = {0};
  for (int s = 0; s < 25; ++s) {
    b.sim.apply_input_pattern(zero, kAllFeatures);
    b.sim.step();
  }
  float pat[kAllFeatures];
  make_pattern(c, pat);
  // 36 present steps so the prior delivery (delay 4) has time to
  // propagate, the dendritic spike to fire, and fire_rate_ema to
  // accumulate above the utterance threshold despite the refractory
  // period (motor can fire at most every 4 steps -> ema cap ~ 0.25).
  run_present(b, pat, /*prime=*/-1, 0.0f, 36);
  float rates[kClasses];
  b.sim.read_output(rates, kClasses);
  const char* said = utter(rates);
  const int said_idx = word_index(said);
  b.last_target = c;
  b.last_said = said_idx;
  b.last_match = (said_idx == c);
  std::memcpy(b.last_rates, rates, sizeof(rates));
  std::printf("[show] shown=%s  said=%s  rates=", concept.c_str(), said);
  for (int i = 0; i < kClasses; ++i) std::printf(" %s:%.2f", kWords[i], rates[i]);
  std::printf("\n");
}

void cmd_teach(Brain& b, const std::string& concept) {
  const int c = word_index(concept);
  if (c < 0) { std::printf("unknown concept '%s'\n", concept.c_str()); return; }
  b.sim.clear_eligibility();
  b.sim.reset_dynamics();
  float zero[kAllFeatures] = {0};
  for (int s = 0; s < 20; ++s) {
    b.sim.apply_input_pattern(zero, kAllFeatures);
    b.sim.step();
  }
  float pat[kAllFeatures];
  make_pattern(c, pat);
  run_present(b, pat, /*prime=*/c, 0.4f, 16);
  float rates[kClasses];
  b.sim.read_output(rates, kClasses);
  const char* said = utter(rates);
  const int said_idx = word_index(said);
  b.last_target = c;
  b.last_said = said_idx;
  b.last_match = (said_idx == c);
  std::memcpy(b.last_rates, rates, sizeof(rates));
  std::printf("[teach] target=%s  said=%s%s\n", concept.c_str(), said,
              b.last_match ? "  (match)" : "");
}

void cmd_correct(Brain& b) {
  if (b.last_target < 0) {
    std::printf("[correct] no last episode\n");
    return;
  }
  float rewards[kClasses];
  for (int c = 0; c < kClasses; ++c) {
    rewards[c] = (c == b.last_target) ? 1.0f : -0.5f;
  }
  b.sim.apply_reward_per_class(rewards, kClasses, 0.1f);
  for (int s = 0; s < 4; ++s) b.sim.step();
  std::printf("[correct] +reward applied to '%s'\n",
              kWords[b.last_target]);
}

void cmd_wrong(Brain& b) {
  if (b.last_target < 0) {
    std::printf("[wrong] no last episode\n");
    return;
  }
  float rewards[kClasses];
  for (int c = 0; c < kClasses; ++c) {
    rewards[c] = (c == b.last_target) ? 0.5f : -1.0f;
  }
  b.sim.apply_reward_per_class(rewards, kClasses, -0.05f);
  // Aversive on wrong-but-confident.
  if (b.last_said >= 0 && b.last_said != b.last_target) {
    const float top = b.last_rates[b.last_said];
    float second = 0;
    for (int c = 0; c < kClasses; ++c) {
      if (c != b.last_said && b.last_rates[c] > second)
        second = b.last_rates[c];
    }
    const float confidence = top - second;
    if (confidence > 0.1f) b.sim.apply_aversive(confidence);
  }
  for (int s = 0; s < 4; ++s) b.sim.step();
  std::printf("[wrong] -reward + aversive applied; expected %s\n",
              kWords[b.last_target]);
}

void cmd_status(Brain& b) {
  b.sim.refresh_position_features();
  std::printf("[status] step=%d  neurons=%zu  synapses=%zu  "
              "structural-blobs=%d  bins=%zu\n",
              b.sim.current_step(), b.sim.neuron_count(),
              b.sim.total_synapses(),
              b.sim.count_structural_neurons(),
              b.sim.position_bin_count());
}

bool process_line(Brain& b, const std::string& raw) {
  std::string line = raw;
  // Trim leading whitespace.
  while (!line.empty() && std::isspace(static_cast<unsigned char>(line.front())))
    line.erase(line.begin());
  if (line.empty() || line[0] == '#') return true;
  std::istringstream is(line);
  std::string cmd;
  is >> cmd;
  if (cmd == "help")        cmd_help();
  else if (cmd == "concepts") cmd_concepts();
  else if (cmd == "babble") {
    int n = 10; is >> n; cmd_babble(b, n);
  }
  else if (cmd == "show") {
    std::string c; is >> c; cmd_show(b, c);
  }
  else if (cmd == "teach") {
    std::string c; is >> c; cmd_teach(b, c);
  }
  else if (cmd == "correct") cmd_correct(b);
  else if (cmd == "wrong")   cmd_wrong(b);
  else if (cmd == "status")  cmd_status(b);
  else if (cmd == "save") {
    std::string p; is >> p;
    if (p.empty()) p = "chat_brain.snc";
    std::printf("[save] %s -> %s\n", p.c_str(),
                b.sim.save_state(p.c_str()) ? "ok" : "FAILED");
  }
  else if (cmd == "load") {
    std::string p; is >> p;
    if (p.empty()) p = "chat_brain.snc";
    std::printf("[load] %s -> %s\n", p.c_str(),
                b.sim.load_state(p.c_str()) ? "ok" : "FAILED");
  }
  else if (cmd == "quit" || cmd == "exit") return false;
  else std::printf("unknown command '%s' (try 'help')\n", cmd.c_str());
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Brain b{make_config()};

  if (argc > 1 && std::string(argv[1]) == "--load") {
    if (argc < 3) {
      std::fprintf(stderr, "usage: snc_chat [--load <path>]\n");
      return 1;
    }
    if (!b.sim.load_state(argv[2])) {
      std::fprintf(stderr, "load %s failed\n", argv[2]);
      return 1;
    }
    std::printf("[ready] loaded brain from %s\n", argv[2]);
    // Note: skip_noise mapping needs the role/polarity already in the
    // loaded neurons. Reconstruct it by scanning roles. Inhibitor cells
    // are detected by polarity != EXCITATORY (PV/SST/VIP).
    b.skip_noise.assign(b.sim.neuron_count() + 1, false);
    for (const auto& nu : b.sim.neurons()) {
      const bool inhibitory =
          nu.polarity == snc::NeuronPolarity::INHIBITORY ||
          nu.polarity == snc::NeuronPolarity::INHIBITORY_SST ||
          nu.polarity == snc::NeuronPolarity::INHIBITORY_VIP;
      const bool special =
          nu.role == snc::NeuronRole::INPUT ||
          nu.role == snc::NeuronRole::OUTPUT ||
          inhibitory;
      if (special && nu.id < b.skip_noise.size())
        b.skip_noise[nu.id] = true;
    }
  } else {
    build_anatomy(b);
    std::printf("[ready] new brain. %d concepts: ", kClasses);
    for (int i = 0; i < kClasses; ++i) {
      std::printf("%s%s", kWords[i], (i + 1 == kClasses) ? "\n" : " ");
    }
    std::printf("[ready] type 'help' for commands.\n");
  }

  std::string line;
  while (std::getline(std::cin, line)) {
    if (!process_line(b, line)) break;
  }
  return 0;
}
