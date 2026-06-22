#include "snc/encoders.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace snc {

bool parse_encoder(const std::string& name, Encoder& out) {
  if (name == "direct") { out = Encoder::Direct; return true; }
  if (name == "poisson") { out = Encoder::Poisson; return true; }
  if (name == "latency") { out = Encoder::Latency; return true; }
  return false;
}

const char* encoder_name(Encoder e) {
  switch (e) {
    case Encoder::Direct: return "direct";
    case Encoder::Poisson: return "poisson";
    case Encoder::Latency: return "latency";
  }
  return "?";
}

std::vector<InputEvent> encode(Encoder enc, const float* features,
                               int n_channels, int num_steps,
                               const EncoderParams& p, uint64_t seed,
                               int sample_id) {
  std::vector<InputEvent> events;
  switch (enc) {
    case Encoder::Direct: {
      // Constant injected current every step -- the stable, spike-light
      // baseline used for debugging and early training.
      events.reserve(static_cast<std::size_t>(n_channels) * num_steps);
      for (int c = 0; c < n_channels; ++c) {
        const float v = features[c];
        if (v <= p.threshold) continue;
        for (int t = 0; t < num_steps; ++t)
          events.push_back({sample_id, c, t, v * p.gain});
      }
      break;
    }
    case Encoder::Poisson: {
      // Bernoulli(p) spike each step with p proportional to feature value.
      std::mt19937_64 rng(seed);
      std::uniform_real_distribution<float> uni(0.0f, 1.0f);
      for (int c = 0; c < n_channels; ++c) {
        const float v = features[c];
        if (v <= p.threshold) continue;
        const float prob = std::min(1.0f, v * p.max_rate);
        for (int t = 0; t < num_steps; ++t)
          if (uni(rng) < prob)
            events.push_back({sample_id, c, t, p.gain});
      }
      break;
    }
    case Encoder::Latency: {
      // Time-to-first-spike: brighter feature -> earlier single spike.
      for (int c = 0; c < n_channels; ++c) {
        const float v = features[c];
        if (v <= p.threshold) continue;
        const float vc = std::min(1.0f, v);
        int t = static_cast<int>(std::lround((num_steps - 1) * (1.0f - vc)));
        t = std::clamp(t, 0, num_steps - 1);
        events.push_back({sample_id, c, t, p.gain});
      }
      break;
    }
  }
  return events;
}

}  // namespace snc
