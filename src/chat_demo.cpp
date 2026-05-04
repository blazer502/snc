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
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// Optional persistent log: every line `say()` prints to stdout is also
// written here when set, so a session transcript can be reviewed later.
// Opened from main() if --log <path> is on the command line, or
// automatically to chat_session.log if not specified.
std::FILE* g_log = nullptr;

void say(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stdout, fmt, ap);
  va_end(ap);
  if (g_log) {
    va_start(ap, fmt);
    std::vfprintf(g_log, fmt, ap);
    va_end(ap);
    std::fflush(g_log);
  }
}

void log_input(const std::string& line) {
  // Mirror user-typed commands into the log so the transcript is
  // round-trippable. stdin is not echoed to stdout, so stdout alone
  // would only show the responses.
  if (!g_log) return;
  std::fprintf(g_log, "> %s\n", line.c_str());
  std::fflush(g_log);
}

// 20-word vocabulary drawn from early-childhood acquisition corpora
// (CHILDES, MacArthur-Bates CDI top words). The original 12 toddler
// words + 4 number basics + 4 colour basics. Semantic groups:
//   people     : mom / dad / baby
//   objects    : ball / dog / cat
//   greetings  : hi / bye
//   response   : yes / no
//   action     : more / stop
//   numbers    : one / two / three / four
//   colours    : red / green / blue / yellow
constexpr int kClasses = 20;
constexpr int kFeatPerClass = 4;
constexpr int kLabelFeatures = kClasses * kFeatPerClass;     // 80
constexpr int kSelfFeatures = kClasses;                      // 20
constexpr int kImgRows = 4;
constexpr int kImgCols = 4;
constexpr int kImageFeatures = kImgRows * kImgCols;          // 16

// Channel layout (input neurons):
//   [0   .. 79 ]  label sensory features      (kLabelFeatures = 80)
//   [80  .. 99 ]  efference / self-perception (kSelfFeatures  = 20)
//   [100 .. 115]  retinotopic image pixels    (kImageFeatures = 16)
//   [116 .. 123]  cochlear bins (Pack 26-A)   (kCochleaBins   = 8)
constexpr int kImgChannelStart = kLabelFeatures + kSelfFeatures;  // 100
constexpr int kExtFeatures = kLabelFeatures;                  // 80
constexpr int kEffFeatures = kSelfFeatures;                   // 20

// Pack 26-A.tune.lite (retry post-Phase-1): minimal cochlear pathway.
// 8 cochlea bins -> 8 A1 cells, DIRECT (no CN/IC/MGN intermediates),
// permanent labelled-line weight 0.55 delay 13. A1 -> motor PLASTIC
// at weight 0.0 (no ghost-weight homeostatic drag). With Phase 1
// AXON x DENDRITE synaptogenesis, organic A1->motor contacts won't
// form spuriously; only the install_synapse-created edges are active.
constexpr int kCochleaBins = 8;
constexpr int kCochleaChannelStart =
    kLabelFeatures + kSelfFeatures + kImageFeatures;          // 116
constexpr int kAllFeatures =
    kLabelFeatures + kSelfFeatures + kImageFeatures + kCochleaBins;  // 124

constexpr int kA1Neurons     = 8;
constexpr int kA1ChannelBase = 9400;

// Pack 26-B.tune.lite: minimal visual pathway (Hubel & Wiesel 1962).
// 8 V1 simple cells with 4-pixel orientation-tuned receptive fields:
// 4 horizontal-row detectors + 4 vertical-column detectors. The
// 16-pixel retina (existing image_in) projects 4 cells onto each V1
// cell at weight 0.15 -- 3 of 4 receptive-field pixels co-firing
// crosses fire_threshold=0.45 (orientation selectivity). V1 -> motor
// is plastic at weight 0.0 (no homeostatic drag); STDP grows the
// connections that fire coincident with motor activity during shows.
constexpr int kV1Neurons     = 8;
constexpr int kV1ChannelBase = 9500;

// Pack 26-C.tune.lite: motor speech via premotor sequencer + 5
// articulators (Tourville & Guenther 2011 DIVA-model template,
// simplified). Each word's motor cell fires its dedicated premotor,
// which in turn fires the 1-2 articulators that produce the word's
// dominant phonemes. Output: a temporal trajectory of articulator
// fire rates per word -- the brain produces a motor program, not
// just an argmax-of-current.
constexpr int kArticulators = 5;
constexpr int kArticulatorChannelBase = 9600;
constexpr int kPremotorChannelBase    = 9700;
// Articulator indices: 0 jaw, 1 tongue_tip, 2 tongue_body, 3 lips,
// 4 glottis. Two articulators per word for richer programs.
constexpr int kArticulatorPattern[kClasses][2] = {
    /* mom    /mɑm/   */ {3, 0},   // lips, jaw
    /* dad    /dæd/   */ {1, 0},   // tongue_tip, jaw
    /* baby   /beɪbi/ */ {3, 0},
    /* ball   /bɔl/   */ {3, 1},
    /* dog    /dɔɡ/   */ {1, 2},
    /* cat    /kæt/   */ {2, 1},
    /* hi     /haɪ/   */ {4, 0},   // glottis, jaw
    /* bye    /baɪ/   */ {3, 0},
    /* yes    /jɛs/   */ {2, 1},
    /* no     /noʊ/   */ {1, 0},
    /* more   /mɔr/   */ {3, 1},
    /* stop   /stɑp/  */ {1, 3},
    /* one    /wʌn/   */ {3, 1},
    /* two    /tu/    */ {1, 0},
    /* three  /θri/   */ {1, 1},
    /* four   /fɔr/   */ {3, 1},
    /* red    /rɛd/   */ {1, 0},
    /* green  /ɡrin/  */ {2, 1},
    /* blue   /blu/   */ {3, 1},
    /* yellow /jɛloʊ/ */ {2, 1},
};
const char* kArticulatorName[kArticulators] = {
    "jaw", "tng_tip", "tng_bod", "lips", "glott"
};

// Pack 26-C-full: closed-loop articulator -> cochlea. Each
// articulator's firing drives 1-2 cochlear bins corresponding to its
// rough phonemic frequency profile, so the brain hears its own
// articulator activations as sound (corollary discharge / efference
// copy). With 8-bin cochlea (200..4000 Hz log-frequency):
//   jaw       low vowel formant F1                 -> bins 0, 1
//   tongue_tip high consonant fricative noise      -> bins 6, 7
//   tongue_body mid formant                        -> bins 3, 4
//   lips      bilabial low                         -> bins 1, 2
//   glottis   voicing fundamental                  -> bin 0
constexpr int kArticulatorCochleaBin[kArticulators][2] = {
    /* jaw       */ {0, 1},
    /* tongue_tip*/ {6, 7},
    /* tongue_bod*/ {3, 4},
    /* lips      */ {1, 2},
    /* glottis   */ {0, -1},   // -1 = no second bin
};
// Per-V1-cell receptive field over the 16-pixel retina:
//   v1[0..3] = horizontal row detectors (rows 0..3)
//   v1[4..7] = vertical column detectors (cols 0..3)
constexpr int kV1RFSize = 4;
constexpr int kV1RF[kV1Neurons][kV1RFSize] = {
    /* v1[0]  row 0 */ { 0,  1,  2,  3},
    /* v1[1]  row 1 */ { 4,  5,  6,  7},
    /* v1[2]  row 2 */ { 8,  9, 10, 11},
    /* v1[3]  row 3 */ {12, 13, 14, 15},
    /* v1[4]  col 0 */ { 0,  4,  8, 12},
    /* v1[5]  col 1 */ { 1,  5,  9, 13},
    /* v1[6]  col 2 */ { 2,  6, 10, 14},
    /* v1[7]  col 3 */ { 3,  7, 11, 15},
};

constexpr int kAcousticOnsetSteps  = 4;
constexpr int kAcousticVowelSteps  = 8;
constexpr int kAcousticOffsetSteps = 4;
constexpr int kAcousticDuration =
    kAcousticOnsetSteps + kAcousticVowelSteps + kAcousticOffsetSteps;  // 16

// Peterson-Barney 1952 vowel formant frequencies (Hz). One formant
// triple per word; values approximate the dominant vowel of the
// English production. Numbers and colour words use plausible approxes
// from CMU dict / IPA charts.
struct WordFormants { float f1, f2, f3; };
constexpr WordFormants kFormants[kClasses] = {
    /* mom    /ɑ/   */ {730.f, 1090.f, 2440.f},
    /* dad    /æ/   */ {660.f, 1720.f, 2410.f},
    /* baby   /eɪ/  */ {550.f, 1900.f, 2500.f},
    /* ball   /ɔ/   */ {570.f,  840.f, 2410.f},
    /* dog    /ɔ/   */ {570.f,  870.f, 2400.f},
    /* cat    /æ/   */ {660.f, 1720.f, 2410.f},
    /* hi     /aɪ/  */ {660.f, 1200.f, 2550.f},
    /* bye    /aɪ/  */ {660.f, 1200.f, 2550.f},
    /* yes    /ɛ/   */ {530.f, 1840.f, 2480.f},
    /* no     /oʊ/  */ {570.f,  840.f, 2410.f},
    /* more   /ɔr/  */ {570.f,  840.f, 1700.f},
    /* stop   /ɑ/   */ {730.f, 1090.f, 2440.f},
    /* one    /wʌn/ */ {600.f,  900.f, 2500.f},
    /* two    /tu/  */ {300.f,  870.f, 2240.f},
    /* three  /θri/ */ {270.f, 2300.f, 3010.f},
    /* four   /fɔr/ */ {570.f,  840.f, 1700.f},
    /* red    /ɛd/  */ {530.f, 1840.f, 2480.f},
    /* green  /in/  */ {270.f, 2300.f, 3010.f},
    /* blue   /lu/  */ {300.f,  870.f, 2240.f},
    /* yellow /ɛlo/ */ {530.f, 1840.f, 2400.f},
};

const char* kWords[kClasses] = {
    "mom", "dad", "baby", "ball", "dog", "cat",
    "hi", "bye", "yes", "no", "more", "stop",
    "one", "two", "three", "four",
    "red", "green", "blue", "yellow"
};

// Hand-designed 4x4 retinal patterns -- one per concept. Each
// pattern lights up 4 pixels; some pixels are shared between
// concepts on purpose (visual generalisation should sit on top of
// labels). Indexed (row, col) -> pixel = row*kImgCols + col.
//
//   mom : top-left 2x2     dad : top-right 2x2
//   baby: top centre       ball: centre 2x2
//   dog : bot-left 2x2     cat : bot-right 2x2
//   hi  : left edge        bye : right edge
//   yes : top edge         no  : bottom edge
//   more: 4 corners        stop: main diagonal
//   one  : T-shape         two  : two horizontal pairs
//   three: zigzag          four : anti-diagonal-cross
constexpr int kImageBits[kClasses][4] = {
    /* mom  */ {0,  1,  4,  5},
    /* dad  */ {2,  3,  6,  7},
    /* baby */ {1,  2,  5,  6},
    /* ball */ {5,  6,  9, 10},
    /* dog  */ {8,  9, 12, 13},
    /* cat  */ {10, 11, 14, 15},
    /* hi   */ {0,  4,  8, 12},
    /* bye  */ {3,  7, 11, 15},
    /* yes  */ {0,  1,  2,  3},
    /* no   */ {12, 13, 14, 15},
    /* more */ {0,  3, 12, 15},
    /* stop */ {0,  5, 10, 15},
    /* one   */ {1,  5,  6, 10},
    /* two   */ {1,  2, 13, 14},
    /* three */ {0,  6,  9, 15},
    /* four  */ {3,  5, 10, 12},
    /* red    */ {2,  5,  8, 11},
    /* green  */ {1,  4, 11, 14},
    /* blue   */ {0,  7,  8, 15},
    /* yellow */ {3,  6,  9, 12},
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

// Familiarity classification for the conversational protocol. Looks
// only at the readout rates (no comparison to the shown stimulus --
// the brain doesn't know what was shown, only what its own circuits
// produced). Three levels mirroring known/guess/unknown in human
// word recognition:
//   KNOW    : confident recall; top >= 0.10 AND top - second >= 0.04
//   GUESS   : weak recall; top >= 0.04 AND top - second >= 0.02
//   UNKNOWN : nothing meaningful won the readout
// Returns the corresponding [tag] literal.
const char* familiarity_tag(const float* rates) {
  int top = argmax_of(rates, kClasses);
  float second = 0.0f;
  for (int i = 0; i < kClasses; ++i) {
    if (i != top && rates[i] > second) second = rates[i];
  }
  const float t = rates[top];
  const float c = t - second;
  if (t >= 0.10f && c >= 0.04f) return "[know]";
  if (t >= 0.04f && c >= 0.02f) return "[guess]";
  return "[unknown]";
}

snc::SimConfig make_config() {
  snc::SimConfig cfg;
  // Bigger volume: 12 motor / inhibitor / self clusters and 48 ext
  // inputs need more (x, y) lanes than the 4-word demo.
  cfg.X = 64; cfg.Y = 64; cfg.Z = 48;
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
  // Pack P-lite v2: parallel event dispatch via per-target buckets.
  // N=4 splits delivery work across 4 OpenMP threads while preserving
  // determinism (each bucket owns a disjoint set of post-synaptic
  // neurons). At chat-vocab scale the gain is small but the path is
  // exercised so larger demos can crank N higher.
  cfg.event_dispatch_buckets = 4;
  return cfg;
}

struct Brain {
  snc::Simulator sim;
  std::vector<uint32_t> ext_in;
  std::vector<uint32_t> motors;
  std::vector<uint32_t> selfs;
  std::vector<uint32_t> inhibitors;
  std::vector<uint32_t> image_in;  // 16 retinotopic pixel input neurons
  // Pack 26-A.tune.lite: cochlear pathway.
  std::vector<uint32_t> cochlea;            // 8 INPUT bins
  std::vector<uint32_t> a1;                 // 8 INTERNAL primary auditory
  // Pack 26-B.tune.lite: visual pathway.
  std::vector<uint32_t> v1;                 // 8 INTERNAL V1 simple cells
  // Pack 26-C.tune.lite: motor speech.
  std::vector<uint32_t> articulators;       // 5 INTERNAL articulator cells
  std::vector<uint32_t> premotors;          // kClasses INTERNAL premotor cells
  std::vector<bool> skip_noise;
  std::mt19937 rng;
  // Last episode bookkeeping (for `correct` / `wrong`).
  int last_target = -1;
  int last_said = -1;
  bool last_match = false;
  float last_rates[kClasses] = {0};
  // Neurons whose excitability_bias was raised by the most recent
  // teach episode. Reset to 1.0 by `correct` / `wrong` so the bias
  // does not leak into the next teach.
  std::vector<uint32_t> last_biased;
  // Rolling tallies across show / teach episodes.
  int total_shows = 0;
  int correct_shows = 0;
  int total_teaches = 0;
  int correct_teaches = 0;
  // Current developmental session count -- used to gate auto-grow on
  // load and propagated to the .meta sidecar on save / quit.
  int session_count = 1;

  explicit Brain(snc::SimConfig cfg)
      : sim(cfg), rng(0xCC0DE) {}
};

// Developmental volume-stage table. Anchored to the pre-adolescent
// peak (~1300 cm^3) of human cortical volume. The session count
// (stored in <brain>.snc.meta) decides which stage the network is in;
// at each load, if the brain is smaller than the target stage's
// dimensions we grow_volume to catch up.
struct DevStage {
  int x, y, z;
  int newborns_per_session;
  const char* name;
};
constexpr DevStage kStages[] = {
    { 64,  64, 48,  7, "toddler"        },  // sessions 0-29   (~200 cm^3)
    { 96,  96, 64, 15, "early-child"    },  // sessions 30-59  (~590 cm^3)
    {112, 112, 80, 25, "middle-child"   },  // sessions 60-99  (~1000 cm^3)
    {128, 128, 96, 40, "preadolescent"  },  // sessions 100+   (~1500 cm^3)
};
constexpr int kNumStages = sizeof(kStages) / sizeof(kStages[0]);

int stage_for_session(int session) {
  if (session < 30)  return 0;
  if (session < 60)  return 1;
  if (session < 100) return 2;
  return 3;
}

struct ChatMeta {
  int session_count = 0;
};

ChatMeta load_meta(const std::string& brain_path) {
  ChatMeta m;
  std::ifstream f(brain_path + ".meta");
  if (f) f >> m.session_count;
  return m;
}

void save_meta(const std::string& brain_path, const ChatMeta& m) {
  std::ofstream f(brain_path + ".meta");
  if (f) f << m.session_count << "\n";
}

// Rebuild the chat-side index vectors (motors / selfs / ext_in /
// skip_noise) by scanning every neuron's role / channel / polarity.
// Used both after `build_anatomy` and after `load_state`, since the
// .snc save format has roles / channels but not the demo-specific
// vectors.
void rebuild_index(Brain& b) {
  b.motors.assign(kClasses, 0);
  b.selfs.assign(kClasses, 0);
  b.ext_in.assign(kLabelFeatures, 0);
  b.image_in.assign(kImageFeatures, 0);
  b.cochlea.assign(kCochleaBins, 0);
  b.a1.assign(kA1Neurons, 0);
  b.v1.assign(kV1Neurons, 0);
  b.articulators.assign(kArticulators, 0);
  b.premotors.assign(kClasses, 0);
  b.inhibitors.clear();
  for (const auto& nu : b.sim.neurons()) {
    if (nu.role == snc::NeuronRole::INPUT) {
      const int ch = nu.channel;
      if (ch >= 0 && ch < kLabelFeatures) {
        b.ext_in[ch] = nu.id;
      } else if (ch >= kLabelFeatures &&
                 ch < kLabelFeatures + kSelfFeatures) {
        b.selfs[ch - kLabelFeatures] = nu.id;
      } else if (ch >= kImgChannelStart &&
                 ch < kImgChannelStart + kImageFeatures) {
        b.image_in[ch - kImgChannelStart] = nu.id;
      } else if (ch >= kCochleaChannelStart &&
                 ch < kCochleaChannelStart + kCochleaBins) {
        b.cochlea[ch - kCochleaChannelStart] = nu.id;
      }
    } else if (nu.role == snc::NeuronRole::OUTPUT) {
      if (nu.channel >= 0 && nu.channel < kClasses) {
        b.motors[nu.channel] = nu.id;
      }
    } else if (nu.role == snc::NeuronRole::INTERNAL) {
      const int ch = nu.channel;
      if (ch >= kA1ChannelBase && ch < kA1ChannelBase + kA1Neurons) {
        b.a1[ch - kA1ChannelBase] = nu.id;
      } else if (ch >= kV1ChannelBase &&
                 ch < kV1ChannelBase + kV1Neurons) {
        b.v1[ch - kV1ChannelBase] = nu.id;
      } else if (ch >= kArticulatorChannelBase &&
                 ch < kArticulatorChannelBase + kArticulators) {
        b.articulators[ch - kArticulatorChannelBase] = nu.id;
      } else if (ch >= kPremotorChannelBase &&
                 ch < kPremotorChannelBase + kClasses) {
        b.premotors[ch - kPremotorChannelBase] = nu.id;
      }
    }
  }
  // skip_noise: every labelled-line cell + every inhibitory cell
  // (the lateral PV inh cells plus the ~20% GABAergic fraction
  // randomize_polarity assigned in the bulk).
  b.skip_noise.assign(b.sim.neuron_count() + 1, false);
  for (uint32_t id : b.ext_in)   if (id < b.skip_noise.size()) b.skip_noise[id] = true;
  for (uint32_t id : b.selfs)    if (id < b.skip_noise.size()) b.skip_noise[id] = true;
  for (uint32_t id : b.motors)   if (id < b.skip_noise.size()) b.skip_noise[id] = true;
  for (uint32_t id : b.image_in) if (id < b.skip_noise.size()) b.skip_noise[id] = true;
  for (uint32_t id : b.cochlea)      if (id < b.skip_noise.size()) b.skip_noise[id] = true;
  for (uint32_t id : b.a1)           if (id < b.skip_noise.size()) b.skip_noise[id] = true;
  for (uint32_t id : b.v1)           if (id < b.skip_noise.size()) b.skip_noise[id] = true;
  for (uint32_t id : b.articulators) if (id < b.skip_noise.size()) b.skip_noise[id] = true;
  for (uint32_t id : b.premotors)    if (id < b.skip_noise.size()) b.skip_noise[id] = true;
  for (const auto& nu : b.sim.neurons()) {
    const bool inh = nu.polarity != snc::NeuronPolarity::EXCITATORY;
    if (inh && nu.id < b.skip_noise.size())
      b.skip_noise[nu.id] = true;
  }
}

// Place neurons + install innate priors. Mirrors vocab_demo but with
// kClasses=12 and a 64x64x48 grid.
void build_anatomy(Brain& b) {
  snc::Simulator& sim = b.sim;
  const int Z = sim.grid().Z();
  const int Y = sim.grid().Y();

  // Sparse fetal seed (no glia, no migrating, no CP -- chat builds its
  // bulk by sprouting during babble).
  snc::FetalSeed seed;
  seed.vz_neurons = 160;
  seed.migrating_neurons = 0;
  seed.cortical_plate_neurons = 0;
  seed.vz_thickness = Z - 5;
  seed.radial_glia_density = 0.0f;
  seed.frac_pv = 0.14f;
  seed.frac_sst = 0.04f;
  seed.frac_vip = 0.02f;
  seed.brainstem_neurons = 12;
  seed.thalamic_relay_neurons = 16;
  seed.aversive_nucleus_neurons = 6;
  sim.seed_fetal(seed);

  // External sensory (label) inputs: kLabelFeatures channels = kClasses
  // rows of kFeatPerClass columns at the cortical floor (z=2). At 16
  // classes the row stride is 3 (was 5 at 12 classes) to fit y < Y.
  b.ext_in.reserve(kLabelFeatures);
  for (int c = 0; c < kClasses; ++c) {
    for (int f = 0; f < kFeatPerClass; ++f) {
      const int channel = c * kFeatPerClass + f;
      const int x = 3 + f * 4;
      const int y = 3 + c * 3;
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

  // Retinotopic image inputs: 4x4 grid of "ganglion-cell" neurons
  // placed on a separate cortical patch (z=4). Channels assigned in
  // row-major order starting at kImgChannelStart so the existing
  // label and self-perception channels stay where they are.
  b.image_in.reserve(kImageFeatures);
  for (int r = 0; r < kImgRows; ++r) {
    for (int c = 0; c < kImgCols; ++c) {
      const int channel = kImgChannelStart + r * kImgCols + c;
      const int x = 32 + c * 5;     // separate from label cluster
      const int y = 4 + r * 5;
      const uint32_t id = sim.add_neuron_at(x, y, 4);
      if (!id) {
        std::fprintf(stderr, "image input pixel (%d,%d) failed at (%d,%d,4)\n",
                     r, c, x, y);
        std::exit(1);
      }
      sim.set_role(id, snc::NeuronRole::INPUT, channel);
      sim.set_polarity(id, snc::NeuronPolarity::EXCITATORY);
      b.image_in.push_back(id);
    }
  }

  // Motor outputs + self-perception + lateral inhibitors. With 20
  // clusters, lay them out as a 2-row x 10-col grid in (x, y) at
  // stride 6 to fit X = 64.
  b.motors.reserve(kClasses);
  b.selfs.reserve(kClasses);
  b.inhibitors.reserve(kClasses);
  for (int c = 0; c < kClasses; ++c) {
    const int col = c % 10;
    const int row = c / 10;          // 0 or 1
    const int xm = 4 + col * 6;
    const int ym = (Y / 2 - 8) + row * 16;
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

    // Per-class engram niche: a sphere centered in the cortical
    // column under this motor. Subsequent promote_engram calls
    // bias internal-neuron candidates toward this niche, so each
    // class's engram lives in its own physical region (analogue of
    // distinct cortical columns / category-selective patches like
    // FFA, PPA). Reduces cross-word interference at scale.
    sim.set_engram_region(c, xm, ym, Z / 2, /*radius=*/9);
  }

  // Pack 26-A.tune.lite (post-Phase-1): minimal cochlear pathway.
  // 8 cochlea bins -> 8 A1 cells, direct cochlea -> A1 (no CN/IC/MGN
  // intermediates). Tonotopic strip at y=Y-3 keeps the auditory
  // architecture clear of label / image / motor columns.
  const int aud_y = Y - 3;
  const int aud_x0 = 16;
  auto place_or_nearby = [&](int x, int y, int z) -> uint32_t {
    if (uint32_t id = sim.add_neuron_at(x, y, z); id) return id;
    for (int r = 1; r <= 4; ++r) {
      for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
          if (uint32_t id = sim.add_neuron_at(x + dx, y + dy, z); id)
            return id;
        }
      }
    }
    return 0;
  };
  // 8 cochlea bins, x stride 4, z=0.
  b.cochlea.reserve(kCochleaBins);
  for (int i = 0; i < kCochleaBins; ++i) {
    const int x = aud_x0 + i * 4;
    const uint32_t id = place_or_nearby(x, aud_y, 0);
    if (!id) { std::fprintf(stderr, "cochlea %d failed near (%d,%d,0)\n",
                             i, x, aud_y); std::exit(1); }
    sim.set_role(id, snc::NeuronRole::INPUT, kCochleaChannelStart + i);
    sim.set_polarity(id, snc::NeuronPolarity::EXCITATORY);
    b.cochlea.push_back(id);
  }
  // 8 A1 INTERNAL cells, x stride 4, z=8.
  b.a1.reserve(kA1Neurons);
  for (int i = 0; i < kA1Neurons; ++i) {
    const int x = aud_x0 + i * 4;
    const uint32_t id = place_or_nearby(x, aud_y, 8);
    if (!id) { std::fprintf(stderr, "a1 %d failed near (%d,%d,8)\n",
                             i, x, aud_y); std::exit(1); }
    sim.set_role(id, snc::NeuronRole::INTERNAL, kA1ChannelBase + i);
    sim.set_polarity(id, snc::NeuronPolarity::EXCITATORY);
    b.a1.push_back(id);
  }
  // Cochlea -> A1: 1:1 tonotopic, weight 0.55, delay 13. Permanent
  // labelled-line. install_synapse bypasses the morphology grid so
  // Phase 1's AXON x DENDRITE rule does not apply -- the edge is
  // installed directly.
  for (int i = 0; i < kCochleaBins; ++i) {
    sim.install_synapse(b.cochlea[i], b.a1[i], 0.55f, 13, 0, 1.0f);
  }
  // A1 -> motor: PLASTIC at weight 0.0 (no homeostatic drag) on
  // branch 1. STDP grows the A1 cells' connections to the motors
  // whose teach episodes their firing pattern correlates with.
  // Phase 1 prevents spurious organic A1 -> motor synapses, so the
  // STDP-shaped install_synapse edges are the only auditory readout
  // path. Pack ZZ's post.role == OUTPUT guard preserves the strong
  // ones from microglial elimination once they grow.
  for (int i = 0; i < kA1Neurons; ++i) {
    for (int c = 0; c < kClasses; ++c) {
      sim.install_synapse(b.a1[i], b.motors[c], 0.0f, 4, 1, 0.0f);
    }
  }

  // Pack 26-B.tune.lite (post-Phase-1): minimal visual pathway.
  // 8 V1 simple cells with 4-pixel orientation-tuned receptive
  // fields (Hubel & Wiesel 1962). Placed in their own y strip
  // separate from cochlear cells: y = Y - 7, z = 8 (same depth as
  // A1 but different y).
  const int v1_y = Y - 7;
  const int v1_x0 = 16;
  b.v1.reserve(kV1Neurons);
  for (int i = 0; i < kV1Neurons; ++i) {
    const int x = v1_x0 + i * 4;
    const uint32_t id = place_or_nearby(x, v1_y, 8);
    if (!id) { std::fprintf(stderr, "v1 %d failed near (%d,%d,8)\n",
                             i, x, v1_y); std::exit(1); }
    sim.set_role(id, snc::NeuronRole::INTERNAL, kV1ChannelBase + i);
    sim.set_polarity(id, snc::NeuronPolarity::EXCITATORY);
    b.v1.push_back(id);
  }
  // retina[p] -> v1[i]: 4 retinal pixels per V1 cell (its receptive
  // field, see kV1RF). Weight 0.15 -- 3 of 4 pixels co-firing
  // crosses fire_threshold=0.45 (orientation selectivity).
  // Permanent labelled-line: the receptive field is
  // anatomically defined and shouldn't be plastic.
  for (int i = 0; i < kV1Neurons; ++i) {
    for (int k = 0; k < kV1RFSize; ++k) {
      const int pixel = kV1RF[i][k];
      sim.install_synapse(b.image_in[pixel], b.v1[i], 0.15f, 4, 0, 1.0f);
    }
  }
  // V1 -> motor: PLASTIC at weight 0.0 on branch 1, like A1->motor.
  // STDP-LTP grows the V1 cells whose receptive fields align with
  // each word's image pattern, learned from co-firing during shows.
  for (int i = 0; i < kV1Neurons; ++i) {
    for (int c = 0; c < kClasses; ++c) {
      sim.install_synapse(b.v1[i], b.motors[c], 0.0f, 4, 1, 0.0f);
    }
  }

  // Pack 26-C.tune.lite (post-Phase-1): motor speech via premotor +
  // articulator. 5 articulator cells (jaw / tongue_tip / tongue_body
  // / lips / glottis) at the high-z fringe; one premotor cell per
  // word at z=Z-5 (just below motor cortex). motor[c] -> premotor[c]
  // -> articulators[ kArticulatorPattern[c] ] all permanent
  // labelled-line. The brain's output is now a TEMPORAL motor
  // program (which articulators fire when motor[c] fires), not just
  // an argmax-of-motor-rates.
  const int art_x = 4;
  const int art_y0 = 2;
  for (int i = 0; i < kArticulators; ++i) {
    const int x = art_x;
    const int y = art_y0 + i * 4;
    const uint32_t id = place_or_nearby(x, y, Z - 2);
    if (!id) { std::fprintf(stderr, "articulator %d failed near (%d,%d,%d)\n",
                             i, x, y, Z - 2); std::exit(1); }
    sim.set_role(id, snc::NeuronRole::INTERNAL,
                 kArticulatorChannelBase + i);
    sim.set_polarity(id, snc::NeuronPolarity::EXCITATORY);
    b.articulators.push_back(id);
  }
  // Premotor cells: one per word, placed just below the
  // corresponding motor cluster. Same 2x10 stride-6 layout as
  // motors, at z = Z - 5.
  for (int c = 0; c < kClasses; ++c) {
    const int col = c % 10;
    const int row = c / 10;
    const int xp = 4 + col * 6;
    const int yp = (Y / 2 - 8) + row * 16;
    const uint32_t id = place_or_nearby(xp, yp, Z - 5);
    if (!id) { std::fprintf(stderr, "premotor %d failed near (%d,%d,%d)\n",
                             c, xp, yp, Z - 5); std::exit(1); }
    sim.set_role(id, snc::NeuronRole::INTERNAL,
                 kPremotorChannelBase + c);
    sim.set_polarity(id, snc::NeuronPolarity::EXCITATORY);
    b.premotors.push_back(id);
  }
  // motor[c] -> premotor[c]: permanent labelled-line, weight 0.55,
  // delay 2. Branch 0 so it lands on the dendritic-spike-eligible
  // branch (n_branches=1 default for premotor).
  for (int c = 0; c < kClasses; ++c) {
    sim.install_synapse(b.motors[c], b.premotors[c], 0.55f, 2, 0, 1.0f);
  }
  // premotor[c] -> articulator[i] for each (c, i) in
  // kArticulatorPattern[c]. Weight 0.55 permanent labelled-line.
  for (int c = 0; c < kClasses; ++c) {
    for (int k = 0; k < 2; ++k) {
      const int art_idx = kArticulatorPattern[c][k];
      if (art_idx < 0 || art_idx >= kArticulators) continue;
      sim.install_synapse(b.premotors[c], b.articulators[art_idx],
                          0.55f, 2, 0, 1.0f);
    }
  }
  // Pack 26-C-full: closed-loop articulator -> cochlea. The brain
  // hears its own articulator firings via permanent labelled-line
  // links from each articulator to the cochlear bins that match
  // its rough phonemic frequency profile. Combined with the
  // existing cochlea -> A1 -> motor pathway (Pack 26-A), this
  // closes the speech loop: motor -> premotor -> articulators ->
  // cochlea -> A1 -> motor (corollary discharge analogue).
  for (int i = 0; i < kArticulators; ++i) {
    for (int k = 0; k < 2; ++k) {
      const int bin = kArticulatorCochleaBin[i][k];
      if (bin < 0 || bin >= kCochleaBins) continue;
      sim.install_synapse(b.articulators[i], b.cochlea[bin],
                          0.55f, 2, 0, 1.0f);
    }
  }

  // Innate label priors / efference / lateral inhibition (permanent).
  for (int c = 0; c < kClasses; ++c) {
    for (int f = 0; f < kFeatPerClass; ++f) {
      sim.install_synapse(b.ext_in[c * kFeatPerClass + f],
                          b.motors[c], 0.55f, 4, 0, 1.0f);
    }
  }
  // Innate visual priors: each concept's 4 image pixels also wire to
  // its motor on branch 0. With 4 priors at 0.55 each, the dendritic
  // spike threshold (0.8) is comfortably crossed when all four pixels
  // of a concept's image are active. Same labelled-line treatment as
  // the label priors -- permanent (won't get pruned).
  for (int c = 0; c < kClasses; ++c) {
    for (int p = 0; p < 4; ++p) {
      const int pixel = kImageBits[c][p];
      sim.install_synapse(b.image_in[pixel], b.motors[c], 0.55f, 4, 0, 1.0f);
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

  rebuild_index(b);
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

// Pack 26-A.tune.lite: cochlea + acoustic presentation -----------------

// Greenwood 1990 log-frequency place map. 200-4000 Hz over 8 bins.
int freq_to_bin(float hz) {
  constexpr float lo = 200.0f;
  constexpr float hi = 4000.0f;
  if (hz <= lo) return 0;
  if (hz >= hi) return kCochleaBins - 1;
  const float t = std::log(hz / lo) / std::log(hi / lo);
  int bin = static_cast<int>(t * kCochleaBins);
  if (bin < 0) bin = 0;
  if (bin >= kCochleaBins) bin = kCochleaBins - 1;
  return bin;
}

// Synthesize an 8-bin cochlear activation for `word` at acoustic step
// `t`. Each formant lights a Gaussian-skirted activation (sigma=1)
// with F1/F2/F3 amplitudes 1.0/0.7/0.4. Onset+vowel+offset envelope.
void cochlea_pattern_at(int word, int t, float* out_8) {
  for (int i = 0; i < kCochleaBins; ++i) out_8[i] = 0.0f;
  if (word < 0 || word >= kClasses) return;
  if (t < 0 || t >= kAcousticDuration) return;
  float env;
  if (t < kAcousticOnsetSteps) {
    env = static_cast<float>(t + 1) / kAcousticOnsetSteps;
  } else if (t < kAcousticOnsetSteps + kAcousticVowelSteps) {
    env = 1.0f;
  } else {
    const int into_off = t - kAcousticOnsetSteps - kAcousticVowelSteps;
    env = static_cast<float>(kAcousticOffsetSteps - into_off) /
          kAcousticOffsetSteps;
  }
  if (env < 0.0f) env = 0.0f;
  const WordFormants& f = kFormants[word];
  const int b1 = freq_to_bin(f.f1);
  const int b2 = freq_to_bin(f.f2);
  const int b3 = freq_to_bin(f.f3);
  constexpr float kSigma = 1.0f;
  constexpr float kInv2Sigma2 = 1.0f / (2.0f * kSigma * kSigma);
  for (int i = 0; i < kCochleaBins; ++i) {
    const float d1 = static_cast<float>(i - b1);
    const float d2 = static_cast<float>(i - b2);
    const float d3 = static_cast<float>(i - b3);
    const float a = 1.00f * std::exp(-d1 * d1 * kInv2Sigma2)
                  + 0.70f * std::exp(-d2 * d2 * kInv2Sigma2)
                  + 0.40f * std::exp(-d3 * d3 * kInv2Sigma2);
    out_8[i] = env * a;
  }
}

// Drive the cochlea over kAcousticDuration steps for `word`, then a
// short tail so the volley can reach A1 (cochlea -> A1 conduction
// = 13 steps).
void run_acoustic_present(Brain& b, int word, int tail = 14) {
  float pat[kAllFeatures] = {0};
  for (int t = 0; t < kAcousticDuration; ++t) {
    for (int i = 0; i < kAllFeatures; ++i) pat[i] = 0.0f;
    cochlea_pattern_at(word, t, &pat[kCochleaChannelStart]);
    b.sim.apply_input_pattern(pat, kAllFeatures);
    inject_internal_noise(b);
    b.sim.step();
  }
  float zero[kAllFeatures] = {0};
  for (int t = 0; t < tail; ++t) {
    b.sim.apply_input_pattern(zero, kAllFeatures);
    inject_internal_noise(b);
    b.sim.step();
  }
}


// Commands --------------------------------------------------------------

void cmd_help() {
  say(
      "commands:\n"
      "  help                       list commands\n"
      "  concepts                   list available concepts\n"
      "  babble <N>                 N random motor firings (no reward)\n"
      "  show <concept>             present sensory pattern, observe\n"
      "  teach <concept>            present + prime matching motor\n"
      "  tell <c1> <c2> ...         present a sequence, hear responses\n"
      "  say <concept>              own motor fires + efference predicted\n"
      "                             (self-channel suppressed: 'I said it')\n"
      "  hear <concept>             external drive on self-channel\n"
      "                             (full activation: 'I heard it')\n"
      "  see <concept>              4x4 retinal image input -> motor\n"
      "  imagine <concept>          internal recall: self-cue only,\n"
      "                             motor read-out is the engram's\n"
      "                             reactivation. tags as know/guess/\n"
      "                             unknown.\n"
      "  correct                    last response was right; reward\n"
      "  wrong                      last response was wrong; aversive\n"
      "  sleep [<sws> <rem>]        SWS+REM consolidation cycle\n"
      "  status                     brain stats and rolling accuracy\n"
      "  save <path>                persist brain\n"
      "  load <path>                reload brain (replaces state)\n"
      "  quit                       exit (auto-sleep on the way out)\n");
}

void cmd_concepts() {
  say("concepts (%d):\n", kClasses);
  for (int i = 0; i < kClasses; ++i) {
    say("  %s\n", kWords[i]);
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
  say("[babble] %d trials done. step=%d, synapses=%zu\n",
              n, b.sim.current_step(), b.sim.total_synapses());
}

void cmd_show(Brain& b, const std::string& concept) {
  const int c = word_index(concept);
  if (c < 0) { say("unknown concept '%s'\n", concept.c_str()); return; }
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
  ++b.total_shows;
  if (b.last_match) ++b.correct_shows;
  const char* fam = familiarity_tag(rates);
  say("[show] shown=%s  said=%s  fam=%s  rates=",
              concept.c_str(), said, fam);
  for (int i = 0; i < kClasses; ++i) say(" %s:%.2f", kWords[i], rates[i]);
  say("  show-acc=%d/%d\n", b.correct_shows, b.total_shows);
  // Conversational protocol: when the readout is unknown the brain
  // asks back, inviting the operator to teach. This is the analogue
  // of a child saying "what?" when they hear a word they don't
  // recognise -- curiosity-driven label acquisition rather than
  // silent failure.
  if (std::strcmp(fam, "[unknown]") == 0) {
    say("[ask] what? (heard '%s' but did not recognise it)\n",
                concept.c_str());
  }
}

void cmd_teach(Brain& b, const std::string& concept) {
  const int c = word_index(concept);
  if (c < 0) { say("unknown concept '%s'\n", concept.c_str()); return; }
  b.sim.clear_eligibility();
  b.sim.reset_dynamics();
  float zero[kAllFeatures] = {0};
  for (int s = 0; s < 20; ++s) {
    b.sim.apply_input_pattern(zero, kAllFeatures);
    b.sim.step();
  }
  // Pack 25 / Josselyn & Tonegawa 2020: CREB-style intrinsic
  // excitability bias on the cells anatomically tied to this concept.
  // The label-feature INPUTs, the target motor and its self-perception
  // INPUT all get a 3x boost so promote_engram (later, in cmd_correct)
  // ranks them ahead of noise-driven bulk fetal-seed neurons that
  // otherwise win on raw fire_rate_ema. Reset to 1.0 in cmd_correct /
  // cmd_wrong so the bias doesn't bleed into the next teach episode.
  std::vector<uint32_t> biased;
  biased.reserve(kFeatPerClass + 2 + kA1Neurons);
  for (int f = 0; f < kFeatPerClass; ++f) {
    const uint32_t id = b.ext_in[c * kFeatPerClass + f];
    if (id) { b.sim.set_excitability_bias(id, 3.0f); biased.push_back(id); }
  }
  if (b.motors[c]) {
    b.sim.set_excitability_bias(b.motors[c], 3.0f);
    biased.push_back(b.motors[c]);
  }
  if (b.selfs[c]) {
    b.sim.set_excitability_bias(b.selfs[c], 3.0f);
    biased.push_back(b.selfs[c]);
  }
  // Pack 26-A.tune.lite: A1 tonotopic bias on cells whose preferred
  // bin matches the word's formants. Pack 25.1's bias-floor keeps
  // these cells admissible to engrams even though A1 lives outside
  // the motor niches.
  {
    const WordFormants& f = kFormants[c];
    const int fbins[3] = {freq_to_bin(f.f1), freq_to_bin(f.f2),
                          freq_to_bin(f.f3)};
    for (int i = 0; i < kA1Neurons; ++i) {
      bool match = false;
      for (int fb : fbins) {
        if (std::abs(i - fb) <= 1) { match = true; break; }
      }
      if (match && b.a1[i]) {
        b.sim.set_excitability_bias(b.a1[i], 3.0f);
        biased.push_back(b.a1[i]);
      }
    }
  }
  b.last_biased = std::move(biased);
  // Pack 26-A.tune.lite: acoustic presentation precedes the symbolic
  // drive. Cochlea fires over kAcousticDuration steps shaped by the
  // word's formant envelope; spikes cross the 13-step delay to fire
  // A1 partners. STDP-LTP grows A1 -> motor weights for the cells
  // whose firing aligns with the symbolic teach drive that follows.
  run_acoustic_present(b, c);
  // Multi-modal teaching: present the concept through the label
  // channel (verbal sensory features) AND the self-perception
  // channel ("hearing yourself say it"). The image modality is
  // intentionally excluded -- the canonical 4-pixel patterns
  // overlap heavily across classes (e.g. ball shares pixels with
  // mom, baby, cat, stop), so binding image bits into every
  // engram leaks recall across words. Image is exercised by
  // `see` / `imagine` separately. Mirrors how children acquire
  // word-meaning by hearing + repeating in tight pairing, with
  // visual referent forming a looser cross-modal association.
  float pat[kAllFeatures];
  make_pattern(c, pat);
  pat[kExtFeatures + c] = 0.3f;
  run_present(b, pat, /*prime=*/c, 0.3f, 16);
  float rates[kClasses];
  b.sim.read_output(rates, kClasses);
  const char* said = utter(rates);
  const int said_idx = word_index(said);
  b.last_target = c;
  b.last_said = said_idx;
  b.last_match = (said_idx == c);
  std::memcpy(b.last_rates, rates, sizeof(rates));
  ++b.total_teaches;
  if (b.last_match) ++b.correct_teaches;
  say("[teach] target=%s  said=%s%s  teach-acc=%d/%d\n",
              concept.c_str(), said, b.last_match ? "  (match)" : "",
              b.correct_teaches, b.total_teaches);
}

void cmd_correct(Brain& b) {
  if (b.last_target < 0) {
    say("[correct] no last episode\n");
    return;
  }
  float rewards[kClasses];
  for (int c = 0; c < kClasses; ++c) {
    rewards[c] = (c == b.last_target) ? 1.0f : -0.5f;
  }
  b.sim.apply_reward_per_class(rewards, kClasses, 0.1f);
  for (int s = 0; s < 4; ++s) b.sim.step();
  // Engram protection: when the last response actually matched the
  // target, lock in the recall path. Top 8 internal neurons + the
  // motor output for this class form the cell assembly; their
  // mutual synapses become permanent so subsequent words cannot
  // overwrite them.
  int promoted = 0;
  if (b.last_match) {
    promoted = b.sim.promote_engram(b.last_target, /*top_k_internal=*/8);
  }
  say("[correct] +reward applied to '%s'\n", kWords[b.last_target]);
  if (promoted > 0) {
    say("[engram] %s: promoted %d synapses (total permanent=%zu)\n",
                kWords[b.last_target], promoted,
                b.sim.permanent_synapse_count());
  }
  // Pack 25: drop the teach-episode excitability bias.
  for (uint32_t id : b.last_biased) b.sim.set_excitability_bias(id, 1.0f);
  b.last_biased.clear();
}

void cmd_pair_teach(Brain& b,
                    const std::string& concept_a,
                    const std::string& concept_b) {
  // Pack 29 v1: two-word combinations. Train the brain on a
  // sequential AB pairing -- "A then B". Pack 25's memory linking
  // (same session_id -> 0.5x soft fresh-neuron penalty between
  // co-promoted classes) brings A's and B's engram populations
  // into shared cells so recalling A activates the cells that
  // also drive B. Premotor sequencer (Pack 26-C) carries the
  // actual temporal trajectory.
  const int ca = word_index(concept_a);
  const int cb = word_index(concept_b);
  if (ca < 0 || cb < 0) {
    say("unknown concept '%s' or '%s'\n",
        concept_a.c_str(), concept_b.c_str());
    return;
  }
  b.sim.clear_eligibility();
  b.sim.reset_dynamics();
  float zero[kAllFeatures] = {0};
  for (int s = 0; s < 12; ++s) {
    b.sim.apply_input_pattern(zero, kAllFeatures);
    b.sim.step();
  }

  // Bias the participating cells (both A and B label / motor /
  // self) so promote_engram picks them; reset at end. Pack 25.1
  // bias-floor keeps them admissible regardless of niche distance.
  std::vector<uint32_t> biased;
  auto bias = [&](uint32_t id, float v) {
    if (id) {
      b.sim.set_excitability_bias(id, v);
      biased.push_back(id);
    }
  };
  for (int f = 0; f < kFeatPerClass; ++f) {
    bias(b.ext_in[ca * kFeatPerClass + f], 3.0f);
    bias(b.ext_in[cb * kFeatPerClass + f], 3.0f);
  }
  bias(b.motors[ca], 3.0f); bias(b.motors[cb], 3.0f);
  bias(b.selfs[ca],  3.0f); bias(b.selfs[cb],  3.0f);

  // Phase 1: present A.
  float pat[kAllFeatures];
  make_pattern(ca, pat);
  pat[kExtFeatures + ca] = 0.3f;
  run_present(b, pat, ca, 0.3f, 12);

  // Brief silent gap so A's spike volley reaches motor + premotor
  // before B begins. The premotor sequencer (Pack 26-C) records
  // motor[A]'s firing in this window.
  for (int s = 0; s < 4; ++s) {
    b.sim.apply_input_pattern(zero, kAllFeatures);
    b.sim.step();
  }

  // Phase 2: present B with prime.
  make_pattern(cb, pat);
  pat[kExtFeatures + cb] = 0.3f;
  run_present(b, pat, cb, 0.3f, 12);

  // Reward both A and B together. With a single session_id
  // bracketing both promote_engram calls, Pack 25's memory linking
  // softens the fresh-neuron penalty between the two classes from
  // 10x to 2x -- they end up sharing engram cells.
  float rewards[kClasses];
  for (int c = 0; c < kClasses; ++c) {
    rewards[c] = (c == ca || c == cb) ? 1.0f : -0.3f;
  }
  b.sim.apply_reward_per_class(rewards, kClasses, 0.05f);
  for (int s = 0; s < 4; ++s) b.sim.step();

  const int prom_a = b.sim.promote_engram(ca, /*top_k_internal=*/8);
  const int prom_b = b.sim.promote_engram(cb, /*top_k_internal=*/8);

  for (uint32_t id : biased) b.sim.set_excitability_bias(id, 1.0f);

  // Read the post-episode rates so the user can see the pair was
  // observed in the desired order.
  float rates[kClasses];
  b.sim.read_output(rates, kClasses);
  say("[pair_teach] %s -> %s  promoted=%d/%d  "
      "rates: %s=%.2f %s=%.2f\n",
      concept_a.c_str(), concept_b.c_str(), prom_a, prom_b,
      concept_a.c_str(), rates[ca],
      concept_b.c_str(), rates[cb]);
}

void cmd_wrong(Brain& b) {
  if (b.last_target < 0) {
    say("[wrong] no last episode\n");
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
  // Pack 25: drop the teach-episode excitability bias even when the
  // episode is corrected as wrong, so it doesn't leak into the next
  // teach.
  for (uint32_t id : b.last_biased) b.sim.set_excitability_bias(id, 1.0f);
  b.last_biased.clear();
  say("[wrong] -reward + aversive applied; expected %s\n",
              kWords[b.last_target]);
}

// 4x4 image divided into four 2x2 quadrants. Each saccade fixates
// one quadrant -- the brain only sees those four pixels at a time,
// approximating fovea-only acuity. The image as a whole is built up
// across saccades.
constexpr int kQuadrantPixels[4][4] = {
    /* TL */ { 0,  1,  4,  5},
    /* TR */ { 2,  3,  6,  7},
    /* BL */ { 8,  9, 12, 13},
    /* BR */ {10, 11, 14, 15},
};
constexpr const char* kQuadrantName[4] = {"TL", "TR", "BL", "BR"};

void cmd_see(Brain& b, const std::string& concept) {
  // Visual recognition with saccadic attention. The 4x4 retinotopic
  // image is divided into four 2x2 quadrants; on each fixation only
  // one quadrant's pixels drive the input layer (foveal-acuity
  // analogue). The brain chooses the next saccade by selecting the
  // quadrant with the most currently-active-but-unfixated pixels --
  // a greedy "look at what's still informative" rule, matching the
  // way salience and unfamiliarity bias human gaze (the user's
  // "focusing engine like a human's watch"). Stops when every
  // active pixel in the canonical image has been fixated, or after
  // a 3-saccade budget.
  const int c = word_index(concept);
  if (c < 0) { say("unknown '%s'\n", concept.c_str()); return; }
  b.sim.clear_eligibility();
  b.sim.reset_dynamics();
  float zero[kAllFeatures] = {0};
  for (int s = 0; s < 20; ++s) {
    b.sim.apply_input_pattern(zero, kAllFeatures);
    b.sim.step();
  }
  // Canonical image as a 16-pixel mask.
  bool active[kImageFeatures] = {false};
  for (int p = 0; p < 4; ++p) active[kImageBits[c][p]] = true;
  bool fixated[kImageFeatures] = {false};

  constexpr int kMaxSaccades = 4;
  constexpr int kStepsPerFixation = 12;
  int n_saccades = 0;
  for (int sac = 0; sac < kMaxSaccades; ++sac) {
    // Select quadrant with the most unfixated active pixels.
    int best_q = -1, best_score = 0;
    for (int q = 0; q < 4; ++q) {
      int score = 0;
      for (int p = 0; p < 4; ++p) {
        const int idx = kQuadrantPixels[q][p];
        if (active[idx] && !fixated[idx]) ++score;
      }
      if (score > best_score) { best_score = score; best_q = q; }
    }
    if (best_q < 0) break;  // nothing left to fixate
    ++n_saccades;
    say("[saccade %d] q=%s active_in_q=%d\n",
                sac + 1, kQuadrantName[best_q], best_score);
    // Drive only the quadrant's pixels (and only the active ones),
    // then mark them fixated.
    float pat[kAllFeatures] = {0};
    for (int p = 0; p < 4; ++p) {
      const int idx = kQuadrantPixels[best_q][p];
      if (active[idx]) pat[kImgChannelStart + idx] = 1.0f;
      fixated[idx] = true;
    }
    for (int s = 0; s < kStepsPerFixation; ++s) {
      b.sim.apply_input_pattern(pat, kAllFeatures);
      inject_internal_noise(b);
      b.sim.step();
    }
  }
  // Final read-out after the scanpath. The motor that fires reflects
  // accumulated evidence integrated across all fixations.
  float rates[kClasses];
  b.sim.read_output(rates, kClasses);
  const char* said = utter(rates);
  const char* fam = familiarity_tag(rates);
  say("[see] image=%s  saccades=%d  said=%s  fam=%s  rates=",
              concept.c_str(), n_saccades, said, fam);
  for (int i = 0; i < kClasses; ++i) say(" %s:%.2f", kWords[i], rates[i]);
  say("\n");
}

void cmd_say(Brain& b, const std::string& concept) {
  // Force own motor to fire and *predict* the efference on the
  // matching self-perception channel. With the prediction set, the
  // chemistry phase subtracts it from input_acc so the network's
  // self-channel barely fires -- the network "knows it said the
  // word" rather than re-perceiving its own voice. Mirrors the
  // corollary-discharge mechanism that silences auditory cortex
  // during own-vocalisation.
  const int c = word_index(concept);
  if (c < 0) { say("unknown '%s'\n", concept.c_str()); return; }
  b.sim.clear_eligibility();
  b.sim.reset_dynamics();
  float zero[kAllFeatures] = {0};
  for (int s = 0; s < 20; ++s) {
    b.sim.apply_input_pattern(zero, kAllFeatures);
    b.sim.step();
  }
  float pred[kAllFeatures] = {0};
  pred[kExtFeatures + c] = 1.0f;  // predict the incoming efference
  for (int s = 0; s < 30; ++s) {
    b.sim.apply_input_pattern(zero, kAllFeatures);
    b.sim.apply_prediction_pattern(pred, kAllFeatures);
    b.sim.inject_input(b.motors[c], 1.5f);
    inject_internal_noise(b);
    b.sim.step();
  }
  float rates[kClasses];
  b.sim.read_output(rates, kClasses);
  const float self_r = b.sim.neurons()[b.selfs[c] - 1].fire_rate_ema;
  // Pack 26-C.tune.lite: read the articulator trajectory. The motor
  // cell's spike fires its premotor (delay 2) which fires the
  // word's 1-2 articulators (delay 2). The articulator
  // `fire_rate_ema` after the say episode summarises the brain's
  // speech motor program.
  float art_rate[kArticulators] = {0};
  for (int i = 0; i < kArticulators; ++i) {
    if (b.articulators[i] > 0 &&
        b.articulators[i] <= b.sim.neuron_count()) {
      art_rate[i] = b.sim.neurons()[b.articulators[i] - 1].fire_rate_ema;
    }
  }
  say("[say] motor_%s=%.2f  self_%s=%.2f (predicted away)\n",
      concept.c_str(), rates[c], concept.c_str(), self_r);
  say("[articulator] %s -> ", concept.c_str());
  for (int i = 0; i < kArticulators; ++i) {
    say("%s:%.2f ", kArticulatorName[i], art_rate[i]);
  }
  say("\n");
}

void cmd_hear(Brain& b, const std::string& concept) {
  // External speech: drive the self-perception channel directly,
  // with no prediction set. The network experiences this as "I heard
  // X (not me)" -- the same channel as efference but full effective
  // drive (predicted_input == 0).
  const int c = word_index(concept);
  if (c < 0) { say("unknown '%s'\n", concept.c_str()); return; }
  b.sim.clear_eligibility();
  b.sim.reset_dynamics();
  float zero[kAllFeatures] = {0};
  for (int s = 0; s < 20; ++s) {
    b.sim.apply_input_pattern(zero, kAllFeatures);
    b.sim.step();
  }
  float pat[kAllFeatures] = {0};
  pat[kExtFeatures + c] = 1.0f;
  for (int s = 0; s < 30; ++s) {
    b.sim.apply_input_pattern(pat, kAllFeatures);
    inject_internal_noise(b);
    b.sim.step();
  }
  float rates[kClasses];
  b.sim.read_output(rates, kClasses);
  const float self_r = b.sim.neurons()[b.selfs[c] - 1].fire_rate_ema;
  say("[hear] self_%s=%.2f  motor=", concept.c_str(), self_r);
  for (int i = 0; i < kClasses; ++i) say(" %.2f", rates[i]);
  say("\n");
}

void cmd_imagine(Brain& b, const std::string& concept) {
  // Internal recall: drive a self-channel cue ("I'm thinking of X")
  // with no external label pattern and no motor injection. Whatever
  // the motor read-out shows is the engram's own reactivation from
  // a top-down trigger -- the model's "imagery" of the word. A
  // strong, well-consolidated engram should still produce the
  // matching motor; a missing or interfered engram will read out
  // [unknown] and prove the word is no longer mentally available.
  const int c = word_index(concept);
  if (c < 0) { say("unknown '%s'\n", concept.c_str()); return; }
  b.sim.clear_eligibility();
  b.sim.reset_dynamics();
  float zero[kAllFeatures] = {0};
  for (int s = 0; s < 20; ++s) {
    b.sim.apply_input_pattern(zero, kAllFeatures);
    b.sim.step();
  }
  // Imagery drives the label pattern weakly (internal sensory
  // reactivation -- analogue of V1 imagery activation) AND the
  // self-channel cue (the "this is me thinking" tag). Together
  // they reactivate the engram's recall path from a top-down
  // trigger without the full external sensory drive that `show`
  // gives. A consolidated word produces the matching motor; a
  // forgotten one returns [unknown].
  float pat[kAllFeatures] = {0};
  make_pattern(c, pat);
  for (int i = 0; i < kLabelFeatures; ++i) pat[i] *= 0.5f;
  pat[kExtFeatures + c] = 0.6f;
  for (int s = 0; s < 30; ++s) {
    b.sim.apply_input_pattern(pat, kAllFeatures);
    inject_internal_noise(b);
    b.sim.step();
  }
  float rates[kClasses];
  b.sim.read_output(rates, kClasses);
  const char* heard = utter(rates);
  const char* fam = familiarity_tag(rates);
  say("[imagine] cue=%s  motor=%s  fam=%s  rates=",
              concept.c_str(), heard, fam);
  for (int i = 0; i < kClasses; ++i) say(" %s:%.2f", kWords[i], rates[i]);
  say("\n");
}

void cmd_grow(Brain& b, int dx, int dy, int dz) {
  // Grow the simulated volume by `dx`, `dy`, `dz` voxels per side.
  // Each must be a multiple of region_size (8). Neuron coordinates
  // and the energy field migrate automatically; we rebuild the
  // chat-side index since soma positions shift.
  if (dx < 0 || dy < 0 || dz < 0) {
    say("[grow] negative side amounts not supported here\n");
    return;
  }
  if (dx % 8 != 0 || dy % 8 != 0 || dz % 8 != 0) {
    say("[grow] amounts must be multiples of region_size (8)\n");
    return;
  }
  const auto& g = b.sim.grid();
  say("[grow] %dx%dx%d -> %dx%dx%d\n",
      g.X(), g.Y(), g.Z(),
      g.X() + 2 * dx, g.Y() + 2 * dy, g.Z() + 2 * dz);
  b.sim.grow_volume(dx, dy, dz);
  rebuild_index(b);
  say("[grow] done. neurons=%zu synapses=%zu\n",
      b.sim.neuron_count(), b.sim.total_synapses());
}

void cmd_sleep(Brain& b, int sws_steps = 80, int rem_steps = 60) {
  // SWS-then-REM consolidation cycle. With no recent-pattern history
  // tracked here (chat is open-ended), SWS replays the canonical
  // patterns of every concept once, REM does fragmented internal-noise
  // replay.
  std::vector<std::vector<float>> seq;
  seq.reserve(kClasses);
  for (int c = 0; c < kClasses; ++c) {
    std::vector<float> p(kAllFeatures, 0.0f);
    p[c * kFeatPerClass + 0] = 1.0f;
    p[c * kFeatPerClass + 1] = 1.0f;
    seq.push_back(std::move(p));
  }
  say("[sleep:SWS] sequenced replay of %d concept patterns over %d steps\n",
      kClasses, sws_steps);
  b.sim.sleep_sws_replay(seq, kAllFeatures, sws_steps / kClasses,
                          /*gap_steps=*/2, /*boost=*/1.6f);
  say("[sleep:REM] fragmented replay over %d steps\n", rem_steps);
  b.sim.sleep_rem_replay(rem_steps, seq, kAllFeatures, /*boost=*/1.8f);
  say("[sleep] done. step=%d, synapses=%zu\n",
      b.sim.current_step(), b.sim.total_synapses());
}

void cmd_tell(Brain& b, const std::vector<std::string>& words) {
  // Sequence presentation: each word is shown briefly with a small
  // silent gap between, and the network's spoken response is recorded
  // in order. Useful for compositional teaching ("hi mom" -> greeting
  // followed by addressee). No reward is applied -- callers can still
  // run `correct` / `wrong` against the *last* item if they want.
  if (words.empty()) {
    say("[tell] no words\n");
    return;
  }
  std::vector<std::string> heard;
  heard.reserve(words.size());
  for (const std::string& w : words) {
    const int c = word_index(w);
    if (c < 0) {
      say("[tell] unknown concept '%s'; aborting sequence\n",
                  w.c_str());
      return;
    }
    b.sim.reset_dynamics();
    float zero[kAllFeatures] = {0};
    for (int s = 0; s < 12; ++s) {
      b.sim.apply_input_pattern(zero, kAllFeatures);
      b.sim.step();
    }
    float pat[kAllFeatures];
    make_pattern(c, pat);
    run_present(b, pat, /*prime=*/-1, 0.0f, 30);
    float rates[kClasses];
    b.sim.read_output(rates, kClasses);
    const char* said = utter(rates);
    heard.emplace_back(said);
    // Track the last item so subsequent `correct` / `wrong` works.
    b.last_target = c;
    b.last_said = word_index(said);
    b.last_match = (b.last_said == c);
    std::memcpy(b.last_rates, rates, sizeof(rates));
  }
  say("[tell] heard:");
  for (std::size_t i = 0; i < words.size(); ++i) {
    say(" %s->%s", words[i].c_str(), heard[i].c_str());
  }
  say("\n");
}

void cmd_diagnose(Brain& b) {
  // Pack 27: read-only network diagnostics. Print degree
  // distribution + top-K hubs + activity / weight summaries.
  // Useful for inspecting whether the connectome has acquired the
  // small-world / hub structure of real cortex (Bullmore & Sporns
  // 2012) and for picking workspace-candidate cells (Pack Φ
  // future).
  const auto s = b.sim.network_stats(/*top_k_hubs=*/10);
  say("[diagnose] N=%d  E=%d  perm=%d  active=%d/%d  "
      "mean_w=%.3f  max_w=%.3f  rate_mean=%.4f\n",
      s.n_neurons, s.n_synapses, s.n_permanent,
      s.n_active, s.n_neurons,
      s.mean_weight, s.max_weight, s.mean_fire_rate_ema);
  say("[diagnose] degree  in: mean=%.1f max=%d  "
      "out: mean=%.1f max=%d  total: std=%.1f\n",
      s.mean_in_degree, s.max_in_degree,
      s.mean_out_degree, s.max_out_degree, s.std_total_degree);
  say("[diagnose] top hubs (by total degree):\n");
  for (const auto& h : s.top_hubs) {
    say("  id=%u  in=%d  out=%d  total=%d  soma=(%d,%d,%d)\n",
        h.neuron_id, h.in_degree, h.out_degree, h.total_degree,
        static_cast<int>(h.soma.x),
        static_cast<int>(h.soma.y),
        static_cast<int>(h.soma.z));
  }
}

void cmd_status(Brain& b) {
  b.sim.refresh_position_features();
  say("[status] step=%d  neurons=%zu  synapses=%zu  "
              "structural-blobs=%d  bins=%zu  "
              "shows=%d/%d  teaches=%d/%d  "
              "grid=%dx%dx%d  occ=%.3f  perm=%zu\n",
              b.sim.current_step(), b.sim.neuron_count(),
              b.sim.total_synapses(),
              b.sim.count_structural_neurons(),
              b.sim.position_bin_count(),
              b.correct_shows, b.total_shows,
              b.correct_teaches, b.total_teaches,
              b.sim.grid().X(), b.sim.grid().Y(), b.sim.grid().Z(),
              b.sim.occupancy_fraction(),
              b.sim.permanent_synapse_count());
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
  else if (cmd == "tell") {
    std::vector<std::string> words;
    std::string w;
    while (is >> w) words.push_back(w);
    cmd_tell(b, words);
  }
  else if (cmd == "say") {
    std::string c; is >> c; cmd_say(b, c);
  }
  else if (cmd == "hear") {
    std::string c; is >> c; cmd_hear(b, c);
  }
  else if (cmd == "see") {
    std::string c; is >> c; cmd_see(b, c);
  }
  else if (cmd == "imagine") {
    std::string c; is >> c; cmd_imagine(b, c);
  }
  else if (cmd == "correct") cmd_correct(b);
  else if (cmd == "wrong")   cmd_wrong(b);
  else if (cmd == "pair_teach") {
    std::string a, b2; is >> a >> b2; cmd_pair_teach(b, a, b2);
  }
  else if (cmd == "sleep") {
    int sws = 80; int rem = 60;
    is >> sws >> rem;
    cmd_sleep(b, sws, rem);
  }
  else if (cmd == "grow") {
    int dx = 0, dy = 0, dz = 0;
    is >> dx >> dy >> dz;
    cmd_grow(b, dx, dy, dz);
  }
  else if (cmd == "status")  cmd_status(b);
  else if (cmd == "diagnose") cmd_diagnose(b);
  else if (cmd == "dump") {
    std::string p; is >> p;
    if (p.empty()) p = "chat_dump";
    const bool ok = b.sim.dump_csv(p.c_str());
    say("[dump] %s_{voxels,neurons,synapses}.csv  %s\n",
        p.c_str(), ok ? "ok" : "FAILED");
  }
  else if (cmd == "save") {
    std::string p; is >> p;
    if (p.empty()) p = "chat_brain.snc";
    const bool ok = b.sim.save_state(p.c_str());
    if (ok) {
      ChatMeta m;
      m.session_count = b.session_count;
      save_meta(p, m);
    }
    say("[save] %s -> %s\n", p.c_str(), ok ? "ok" : "FAILED");
  }
  else if (cmd == "load") {
    std::string p; is >> p;
    if (p.empty()) p = "chat_brain.snc";
    say("[load] %s -> %s\n", p.c_str(),
                b.sim.load_state(p.c_str()) ? "ok" : "FAILED");
  }
  else if (cmd == "quit" || cmd == "exit") {
    // Auto-sleep on quit so the next session resumes from a
    // consolidated state. Mimics the day-end sleep cycle.
    say("[quit] auto-sleep before exit\n");
    cmd_sleep(b, 60, 40);
    return false;
  }
  else say("unknown command '%s' (try 'help')\n", cmd.c_str());
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  // CLI parsing: --load <path> (loads brain), --log <path> (writes
  // session transcript). Defaults to opening chat_session.log if no
  // explicit --log was given.
  const char* load_path = nullptr;
  const char* log_path = "chat_session.log";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--load" && i + 1 < argc) {
      load_path = argv[++i];
    } else if (arg == "--log" && i + 1 < argc) {
      log_path = argv[++i];
    } else if (arg == "--no-log") {
      log_path = nullptr;
    } else {
      std::fprintf(stderr,
                   "usage: snc_chat [--load <path>] [--log <path> | --no-log]\n");
      return 1;
    }
  }
  if (log_path) {
    g_log = std::fopen(log_path, "a");
    if (g_log) {
      std::time_t now = std::time(nullptr);
      char ts[64];
      std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S",
                    std::localtime(&now));
      std::fprintf(g_log, "\n=== snc_chat session started %s ===\n", ts);
      std::fflush(g_log);
    } else {
      std::fprintf(stderr, "warning: could not open %s for logging\n",
                   log_path);
    }
  }

  Brain b{make_config()};

  if (load_path) {
    if (!b.sim.load_state(load_path)) {
      std::fprintf(stderr, "load %s failed\n", load_path);
      return 1;
    }
    say("[ready] loaded brain from %s\n", load_path);
    rebuild_index(b);
    // Read the session count from the meta sidecar and decide which
    // developmental stage we should be in. Auto-grow if the saved
    // brain is smaller than the target stage's dimensions.
    ChatMeta meta = load_meta(load_path);
    meta.session_count += 1;  // this load = a new session
    b.session_count = meta.session_count;
    const int target = stage_for_session(meta.session_count);
    const auto& stage = kStages[target];
    const auto& g = b.sim.grid();
    const int dx = (stage.x - g.X()) / 2;
    const int dy = (stage.y - g.Y()) / 2;
    const int dz = (stage.z - g.Z()) / 2;
    if (dx > 0 || dy > 0 || dz > 0) {
      say("[stage] session %d -> %s (%dx%dx%d -> %dx%dx%d)\n",
          meta.session_count, stage.name, g.X(), g.Y(), g.Z(),
          stage.x, stage.y, stage.z);
      b.sim.grow_volume(dx, dy, dz);
      rebuild_index(b);
    } else {
      say("[stage] session %d (%s)\n", meta.session_count, stage.name);
    }
    // Demand-driven growth: even within a developmental stage, if
    // the substrate is filling up (occupancy >= 0.04) expand by one
    // region_size on each side. The threshold is calibrated to the
    // sparse coding regime this simulator runs in -- the cortical
    // analogue is "pyramidal cell density approaches local
    // saturation"; the absolute number is small because synapses
    // and somas occupy individual voxels in a 3D matrix that is
    // mostly empty by design (volume exclusion + extracellular
    // space). Capped at the preadolescent ceiling (128x128x96)
    // regardless of session count -- biological volume does not
    // grow past adolescence.
    {
      const auto& gd = b.sim.grid();
      const float occ = b.sim.occupancy_fraction();
      const int ceil_x = kStages[kNumStages - 1].x;
      const int ceil_y = kStages[kNumStages - 1].y;
      const int ceil_z = kStages[kNumStages - 1].z;
      const int gx = gd.X();
      const int gy = gd.Y();
      const int gz = gd.Z();
      const bool below_ceiling =
          gx < ceil_x || gy < ceil_y || gz < ceil_z;
      if (occ >= 0.10f && below_ceiling) {
        // Step +8 on each side per axis, only on axes still under
        // the ceiling. region_size = 8.
        const int ddx = (gx + 16 <= ceil_x) ? 8 : 0;
        const int ddy = (gy + 16 <= ceil_y) ? 8 : 0;
        const int ddz = (gz + 16 <= ceil_z) ? 8 : 0;
        if (ddx > 0 || ddy > 0 || ddz > 0) {
          say("[grow:demand] occ=%.2f -> +%dx%dx%d on each side "
              "(now %dx%dx%d)\n",
              occ, ddx, ddy, ddz,
              gx + 2 * ddx, gy + 2 * ddy, gz + 2 * ddz);
          b.sim.grow_volume(ddx, ddy, ddz);
          rebuild_index(b);
        }
      }
    }
    // Lifelong neurogenesis: stage-scaled batch of newborn VZ neurons.
    const int newborn = b.sim.birth_neurons(stage.newborns_per_session);
    if (newborn > 0) {
      b.skip_noise.resize(b.sim.neuron_count() + 1, false);
      say("[wake] %d newborn neurons added to the VZ\n", newborn);
    }
    // Save the bumped session count back so the next load advances.
    save_meta(load_path, meta);
    // Pack 25: tell the simulator which session id we're in so
    // memory-linking can soften the fresh-neuron penalty between
    // classes promoted in *this* session vs. previous sessions.
    b.sim.set_session_id(b.session_count);
  } else {
    build_anatomy(b);
    b.sim.set_session_id(b.session_count);
    say("[ready] new brain. %d concepts: ", kClasses);
    for (int i = 0; i < kClasses; ++i) {
      say("%s%s", kWords[i], (i + 1 == kClasses) ? "\n" : " ");
    }
    say("[ready] session 1 (%s). type 'help' for commands.\n",
        kStages[0].name);
  }

  std::string line;
  while (std::getline(std::cin, line)) {
    log_input(line);
    if (!process_line(b, line)) break;
  }
  if (g_log) {
    std::time_t now = std::time(nullptr);
    char ts[64];
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S",
                  std::localtime(&now));
    std::fprintf(g_log, "=== snc_chat session ended %s ===\n", ts);
    std::fclose(g_log);
    g_log = nullptr;
  }
  return 0;
}
