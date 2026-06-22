// Input encoding layer (new-plan.md section 6).
//
// Real SNN systems do not feed raw tensors into spiking neurons; they convert
// values into spike trains / injected currents. Every encoder here turns a
// normalised feature vector (values in [0, 1], one per input channel) into a
// stream of InputEvent over a fixed simulation window of `num_steps`.
//
// The runtime consumes InputEvent streams uniformly: at simulation step `time`
// it adds `amplitude` to the external drive of every INPUT neuron whose
// channel == channel_id. "Direct" encoding emits a per-step current; the spike
// encoders (Poisson/latency) emit unit-amplitude spikes at chosen times.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace snc {

struct InputEvent {
  int sample_id;   // index of the source sample within a batch (0 for single)
  int channel_id;  // input feature / channel
  int time;        // simulation step in [0, num_steps)
  float amplitude; // injected current (direct) or spike magnitude (spiking)
};

enum class Encoder : uint8_t { Direct = 0, Poisson = 1, Latency = 2 };

// Parse "direct" | "poisson" | "latency"; returns false on unknown name.
bool parse_encoder(const std::string& name, Encoder& out);
const char* encoder_name(Encoder e);

struct EncoderParams {
  float gain = 1.0f;       // scales direct current and spike magnitude
  float max_rate = 0.5f;   // Poisson: peak spike prob per step for feature==1
  float threshold = 0.0f;  // features <= threshold emit nothing (sparsity)
};

// Encode one sample (features[0..n_channels)) into events spanning [0,num_steps).
// `seed` makes Poisson deterministic; Direct/Latency ignore it.
std::vector<InputEvent> encode(Encoder enc, const float* features,
                               int n_channels, int num_steps,
                               const EncoderParams& p, uint64_t seed,
                               int sample_id = 0);

}  // namespace snc
