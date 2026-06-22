// snc_bench -- structure-aware SNN benchmark harness (new-plan.md section 10).
//
// Builds a sparse spiking graph (dense / random-sparse / static-snc), encodes a
// dataset into spike/current events, runs forward inference on a chosen runtime
// backend (cpu / openmp / cuda-*), and reports ML, SNN, systems and structural
// metrics. The first-PR scope is forward inference + measurement; supervised
// training (surrogate gradient) and structural epochs are later phases.
//
//   ./snc_bench --structure static-snc --encoder poisson --backend cpu \
//               --num-steps 50 --num-samples 64

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "neuron.hpp"
#include "simulator.hpp"
#include "snc/encoders.hpp"
#include "snc/runtime.hpp"
#include "snc/snn_graph.hpp"

namespace {

using namespace snc;

struct Options {
  std::string dataset = "synthetic";
  std::string encoder = "poisson";
  std::string backend = "cpu";
  std::string structure = "static-snc";
  std::string data_dir;
  std::string log_json;
  std::string log_csv;
  int num_steps = 50;
  int num_samples = 64;
  int hidden = 128;
  int dim = 784;
  int classes = 10;
  int synapse_budget = 50000;
  int delay = 1;
  int refractory = 1;
  float threshold = 1.0f;
  float decay = 0.9f;
  float inhib = 0.2f;
  float gain = 1.0f;
  uint64_t seed = 1;
  bool compare_backends = false;
  bool self_test = false;
};

void usage() {
  std::printf(
      "snc_bench -- structure-aware SNN benchmark\n"
      "  --dataset    synthetic|mnist        (default synthetic)\n"
      "  --encoder    direct|poisson|latency (default poisson)\n"
      "  --backend    cpu|openmp|cuda-atomic|cuda-bucket|cuda-sort\n"
      "  --structure  dense|random-sparse|static-snc\n"
      "  --num-steps N  --num-samples N  --hidden N  --dim N  --classes N\n"
      "  --synapse-budget S  --delay D  --threshold T  --decay D  --refractory R\n"
      "  --inhib F  --gain G  --seed S  --data-dir DIR\n"
      "  --compare-backends   run cpu/openmp/cuda and report parity + speedup\n"
      "  --self-test          validate compile_from_simulator and exit\n"
      "  --log-json FILE  --log-csv FILE\n");
}

template <typename T>
bool arg_i(int& i, int argc, char** argv, const char* flag, T& out) {
  if (std::strcmp(argv[i], flag) != 0) return false;
  if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", flag); std::exit(2); }
  out = static_cast<T>(std::stod(argv[++i]));
  return true;
}
bool arg_s(int& i, int argc, char** argv, const char* flag, std::string& out) {
  if (std::strcmp(argv[i], flag) != 0) return false;
  if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", flag); std::exit(2); }
  out = argv[++i];
  return true;
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") { usage(); std::exit(0); }
    if (a == "--compare-backends") { o.compare_backends = true; continue; }
    if (a == "--self-test") { o.self_test = true; continue; }
    if (arg_s(i, argc, argv, "--dataset", o.dataset)) continue;
    if (arg_s(i, argc, argv, "--encoder", o.encoder)) continue;
    if (arg_s(i, argc, argv, "--backend", o.backend)) continue;
    if (arg_s(i, argc, argv, "--structure", o.structure)) continue;
    if (arg_s(i, argc, argv, "--data-dir", o.data_dir)) continue;
    if (arg_s(i, argc, argv, "--log-json", o.log_json)) continue;
    if (arg_s(i, argc, argv, "--log-csv", o.log_csv)) continue;
    if (arg_i(i, argc, argv, "--num-steps", o.num_steps)) continue;
    if (arg_i(i, argc, argv, "--num-samples", o.num_samples)) continue;
    if (arg_i(i, argc, argv, "--hidden", o.hidden)) continue;
    if (arg_i(i, argc, argv, "--dim", o.dim)) continue;
    if (arg_i(i, argc, argv, "--classes", o.classes)) continue;
    if (arg_i(i, argc, argv, "--synapse-budget", o.synapse_budget)) continue;
    if (arg_i(i, argc, argv, "--delay", o.delay)) continue;
    if (arg_i(i, argc, argv, "--refractory", o.refractory)) continue;
    if (arg_i(i, argc, argv, "--threshold", o.threshold)) continue;
    if (arg_i(i, argc, argv, "--decay", o.decay)) continue;
    if (arg_i(i, argc, argv, "--inhib", o.inhib)) continue;
    if (arg_i(i, argc, argv, "--gain", o.gain)) continue;
    if (arg_i(i, argc, argv, "--seed", o.seed)) continue;
    std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
    usage();
    std::exit(2);
  }
  return o;
}

// ---- Datasets --------------------------------------------------------------

struct Dataset {
  int dim = 0;
  int classes = 0;
  std::vector<std::vector<float>> x;  // [num_samples][dim], values in [0,1]
  std::vector<int> y;                 // [num_samples]
};

// Class-separable synthetic data: each class is a random prototype + noise.
// Deterministic for a fixed seed. Lets the pipeline run with no external files.
Dataset make_synthetic(int n, int dim, int classes, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> uni(0.0f, 1.0f);
  std::normal_distribution<float> noise(0.0f, 0.15f);
  std::vector<std::vector<float>> proto(classes, std::vector<float>(dim));
  for (auto& p : proto)
    for (float& v : p) v = uni(rng);
  Dataset d;
  d.dim = dim;
  d.classes = classes;
  for (int i = 0; i < n; ++i) {
    const int c = i % classes;
    std::vector<float> s(dim);
    for (int k = 0; k < dim; ++k)
      s[k] = std::clamp(proto[c][k] + noise(rng), 0.0f, 1.0f);
    d.x.push_back(std::move(s));
    d.y.push_back(c);
  }
  return d;
}

uint32_t read_be32(std::ifstream& f) {
  unsigned char b[4];
  f.read(reinterpret_cast<char*>(b), 4);
  return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) |
         uint32_t(b[3]);
}

// Minimal MNIST IDX loader. Expects train-images-idx3-ubyte +
// train-labels-idx1-ubyte (uncompressed) under data_dir.
Dataset load_mnist(const std::string& dir, int n) {
  Dataset d;
  std::ifstream img(dir + "/train-images-idx3-ubyte", std::ios::binary);
  std::ifstream lab(dir + "/train-labels-idx1-ubyte", std::ios::binary);
  if (!img || !lab) {
    std::fprintf(stderr,
                 "mnist: could not open IDX files under '%s'\n"
                 "  expected train-images-idx3-ubyte + train-labels-idx1-ubyte\n",
                 dir.c_str());
    std::exit(1);
  }
  read_be32(img);  // magic
  const int count = static_cast<int>(read_be32(img));
  const int rows = static_cast<int>(read_be32(img));
  const int cols = static_cast<int>(read_be32(img));
  read_be32(lab);  // magic
  read_be32(lab);  // label count
  d.dim = rows * cols;
  d.classes = 10;
  const int take = std::min(n, count);
  std::vector<unsigned char> buf(d.dim);
  for (int i = 0; i < take; ++i) {
    img.read(reinterpret_cast<char*>(buf.data()), d.dim);
    unsigned char l = 0;
    lab.read(reinterpret_cast<char*>(&l), 1);
    std::vector<float> s(d.dim);
    for (int k = 0; k < d.dim; ++k) s[k] = buf[k] / 255.0f;
    d.x.push_back(std::move(s));
    d.y.push_back(l);
  }
  return d;
}

// ---- Graph construction ----------------------------------------------------

SNNGraph build_graph(const Options& o, const std::vector<int>& layers) {
  if (o.structure == "dense")
    return make_dense(layers, /*weight=*/0.5f, o.delay);
  if (o.structure == "random-sparse")
    return make_random_sparse(layers, o.synapse_budget, 0.5f, o.delay, o.inhib,
                              o.seed);
  if (o.structure == "static-snc" || o.structure == "dynamic-snc")
    return make_static_snc(layers, o.synapse_budget, 0.5f, o.delay, o.inhib,
                           o.seed);
  std::fprintf(stderr, "unknown structure: %s\n", o.structure.c_str());
  std::exit(2);
}

// ---- compile_from_simulator self-test --------------------------------------

int self_test() {
  SimConfig cfg;
  cfg.X = 32; cfg.Y = 32; cfg.Z = 8;
  Simulator sim(cfg);
  const uint32_t a = sim.add_neuron_at(1, 1, 1);   // INPUT  ch0
  const uint32_t b = sim.add_neuron_at(3, 3, 1);   // INTERNAL (inhibitory)
  const uint32_t c = sim.add_neuron_at(5, 5, 1);   // OUTPUT ch0
  if (!a || !b || !c) { std::fprintf(stderr, "self-test: placement failed\n"); return 1; }
  sim.set_role(a, NeuronRole::INPUT, 0);
  sim.set_role(c, NeuronRole::OUTPUT, 0);
  sim.set_polarity(b, NeuronPolarity::INHIBITORY);
  sim.install_synapse(a, b, 0.3f, 2);
  sim.install_synapse(b, c, 0.7f, 4);
  sim.install_synapse(a, c, 0.5f, 1);

  SNNGraph g = compile_from_simulator(sim);
  std::string err;
  bool ok = g.validate(err);
  // Expect 3 neurons, 3 synapses, b inhibitory, channel mapping preserved.
  ok = ok && g.num_neurons == 3 && g.num_synapses() == 3;
  ok = ok && g.num_input_channels == 1 && g.num_output_channels == 1;
  int n_inhib = 0;
  for (int8_t s : g.sign) if (s < 0) ++n_inhib;
  ok = ok && n_inhib == 1;
  // Verify the a->b->c delays survived (2 and 4) by scanning a's row.
  bool saw_d2 = false, saw_d4 = false;
  for (int i = 0; i < g.num_neurons; ++i)
    for (int e = g.row_ptr[i]; e < g.row_ptr[i + 1]; ++e) {
      if (g.delays[e] == 2) saw_d2 = true;
      if (g.delays[e] == 4) saw_d4 = true;
    }
  ok = ok && saw_d2 && saw_d4;
  std::printf("self-test: %s  (%s)\n", ok ? "PASS" : "FAIL",
              err.empty() ? format_stats(compute_stats(g)).c_str() : err.c_str());
  return ok ? 0 : 1;
}

// ---- Evaluation ------------------------------------------------------------

struct EvalResult {
  RunStats stats;          // summed over samples
  double ms = 0.0;         // total sim wall-clock
  int correct = 0;
  int total = 0;
  Backend used_backend = Backend::Cpu;
};

EvalResult evaluate(const SNNGraph& g, const Dataset& d, const Options& o,
                    Encoder enc, Backend backend) {
  EncoderParams ep;
  ep.gain = o.gain;
  LIFParams lif;
  lif.threshold = o.threshold;
  lif.decay = o.decay;
  lif.refractory = o.refractory;

  EvalResult r;
  r.total = static_cast<int>(d.x.size());
  for (int i = 0; i < r.total; ++i) {
    auto events = encode(enc, d.x[i].data(), d.dim, o.num_steps, ep,
                         o.seed + i, /*sample_id=*/0);
    ForwardResult fr = forward(g, events, o.num_steps, lif, backend);
    r.used_backend = fr.used_backend;
    r.stats.spikes += fr.stats.spikes;
    r.stats.synaptic_events += fr.stats.synaptic_events;
    r.ms += fr.stats.ms;
    // Decode: argmax over output-channel spike counts.
    int pred = 0;
    float best = -1.0f;
    for (int c = 0; c < static_cast<int>(fr.logits.size()); ++c)
      if (fr.logits[c] > best) { best = fr.logits[c]; pred = c; }
    if (pred == d.y[i]) ++r.correct;
  }
  return r;
}

void report(const Options& o, const GraphStats& gs, const EvalResult& r,
            Encoder enc) {
  const double secs = r.ms / 1000.0;
  const double events_per_s = secs > 0 ? r.stats.synaptic_events / secs : 0.0;
  const double spikes_per_s = secs > 0 ? r.stats.spikes / secs : 0.0;
  // Energy proxy (new-plan.md 7.5): alpha*spikes + beta*synaptic_events.
  const double alpha = 1.0, beta = 0.2;
  const double energy = alpha * r.stats.spikes + beta * r.stats.synaptic_events;
  std::printf("\n=== snc_bench ===\n");
  std::printf("structure=%s encoder=%s backend=%s(->%s) steps=%d samples=%d\n",
              o.structure.c_str(), encoder_name(enc), o.backend.c_str(),
              backend_name(r.used_backend), o.num_steps, r.total);
  std::printf("graph: %s\n", format_stats(gs).c_str());
  std::printf("sim:   spikes=%lld synaptic_events=%lld time=%.2f ms\n",
              r.stats.spikes, r.stats.synaptic_events, r.ms);
  std::printf("rate:  %.3g spikes/s  %.3g events/s\n", spikes_per_s, events_per_s);
  std::printf("ml:    argmax-acc=%.3f (%d/%d, untrained forward pass)\n",
              r.total ? double(r.correct) / r.total : 0.0, r.correct, r.total);
  std::printf("energy_proxy=%.3g (a*spikes + b*events)\n", energy);
}

void write_json(const Options& o, const GraphStats& gs, const EvalResult& r,
                Encoder enc) {
  std::ofstream f(o.log_json);
  if (!f) { std::fprintf(stderr, "could not open %s\n", o.log_json.c_str()); return; }
  f << "{\n"
    << "  \"structure\": \"" << o.structure << "\",\n"
    << "  \"encoder\": \"" << encoder_name(enc) << "\",\n"
    << "  \"backend\": \"" << backend_name(r.used_backend) << "\",\n"
    << "  \"num_steps\": " << o.num_steps << ",\n"
    << "  \"num_samples\": " << r.total << ",\n"
    << "  \"neurons\": " << gs.num_neurons << ",\n"
    << "  \"synapses\": " << gs.num_synapses << ",\n"
    << "  \"avg_fan_out\": " << gs.avg_fan_out << ",\n"
    << "  \"density\": " << gs.density << ",\n"
    << "  \"spikes\": " << r.stats.spikes << ",\n"
    << "  \"synaptic_events\": " << r.stats.synaptic_events << ",\n"
    << "  \"sim_ms\": " << r.ms << ",\n"
    << "  \"argmax_acc\": " << (r.total ? double(r.correct) / r.total : 0.0)
    << "\n}\n";
  std::printf("wrote %s\n", o.log_json.c_str());
}

void write_csv(const Options& o, const GraphStats& gs, const EvalResult& r,
               Encoder enc) {
  std::ofstream f(o.log_csv);
  if (!f) { std::fprintf(stderr, "could not open %s\n", o.log_csv.c_str()); return; }
  f << "structure,encoder,backend,num_steps,num_samples,neurons,synapses,"
       "density,spikes,synaptic_events,sim_ms,argmax_acc\n";
  f << o.structure << ',' << encoder_name(enc) << ',' << backend_name(r.used_backend)
    << ',' << o.num_steps << ',' << r.total << ',' << gs.num_neurons << ','
    << gs.num_synapses << ',' << gs.density << ',' << r.stats.spikes << ','
    << r.stats.synaptic_events << ',' << r.ms << ','
    << (r.total ? double(r.correct) / r.total : 0.0) << '\n';
  std::printf("wrote %s\n", o.log_csv.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  Options o = parse(argc, argv);
  if (o.self_test) return self_test();

  Encoder enc;
  if (!parse_encoder(o.encoder, enc)) {
    std::fprintf(stderr, "unknown encoder: %s\n", o.encoder.c_str());
    return 2;
  }

  Dataset d = o.dataset == "mnist"
                  ? load_mnist(o.data_dir, o.num_samples)
                  : make_synthetic(o.num_samples, o.dim, o.classes, o.seed);

  const std::vector<int> layers = {d.dim, o.hidden, d.classes};
  SNNGraph g = build_graph(o, layers);
  std::string err;
  if (!g.validate(err)) {
    std::fprintf(stderr, "graph validation failed: %s\n", err.c_str());
    return 1;
  }
  const GraphStats gs = compute_stats(g);

  if (o.compare_backends) {
    std::printf("graph: %s\n", format_stats(gs).c_str());
    std::printf("\nbackend           spikes      events     time(ms)  speedup  match\n");
    EvalResult ref;
    bool have_ref = false;
    for (const char* bn : {"cpu", "openmp", "cuda-atomic"}) {
      Backend b;
      parse_backend(bn, b);
      EvalResult r = evaluate(g, d, o, enc, b);
      if (!have_ref) { ref = r; have_ref = true; }
      const double speedup = r.ms > 0 ? ref.ms / r.ms : 0.0;
      // Spike-count match vs cpu reference (deterministic ground truth).
      const char* match = r.stats.spikes == ref.stats.spikes ? "exact"
                          : std::llabs(r.stats.spikes - ref.stats.spikes) <=
                                  ref.stats.spikes / 1000 + 1
                              ? "approx"
                              : "DIFF";
      std::printf("%-16s %10lld %11lld %10.2f %8.2f  %s%s\n", backend_name(r.used_backend),
                  r.stats.spikes, r.stats.synaptic_events, r.ms, speedup, match,
                  backend_is_cuda(b) && r.used_backend != b ? " (fallback)" : "");
    }
    return 0;
  }

  Backend backend;
  if (!parse_backend(o.backend, backend)) {
    std::fprintf(stderr, "unknown backend: %s\n", o.backend.c_str());
    return 2;
  }
  EvalResult r = evaluate(g, d, o, enc, backend);
  report(o, gs, r, enc);
  if (!o.log_json.empty()) write_json(o, gs, r, enc);
  if (!o.log_csv.empty()) write_csv(o, gs, r, enc);
  return 0;
}
