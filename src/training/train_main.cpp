// snc_train -- frozen-structure weight training via e-prop (new-plan.md 8.1).
//
// Builds a sparse spiking graph (dense / random-sparse / static-snc), freezes
// its topology, and trains the synaptic weights with local e-prop on a chosen
// dataset, logging loss / train-acc / test-acc / spikes / energy per epoch.
//
//   ./snc_train --structure static-snc --epochs 20 --num-train 300 \
//               --num-test 100 --classes 10 --dim 200

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "snc/cuda_trainer.hpp"
#include "snc/dataset.hpp"
#include "snc/snn_graph.hpp"
#include "snc/trainer.hpp"

namespace {
using namespace snc;

struct Options {
  std::string dataset = "synthetic";
  std::string encoder = "poisson";
  std::string structure = "static-snc";
  std::string train_mode = "all";   // all | readout
  std::string device = "cpu";        // cpu | cuda
  std::string data_dir;
  std::string log_csv;
  int batch = 64;                    // minibatch size (cuda device only)
  int num_steps = 30;
  int epochs = 20;
  std::string hidden = "256";        // hidden layer width(s), e.g. "256" or "256,256"
  int dim = 200;
  int classes = 10;
  int num_train = 300;
  int num_test = 100;
  int synapse_budget = 40000;
  int delay = 1;
  int refractory = 1;
  float noise = 0.35f;
  float lr = 0.08f;
  float w_init = 0.02f;  // keep initial drive near threshold (surrogate alive)
  float w_max = 4.0f;
  float gamma = 0.3f;
  float decay = 0.9f;
  float threshold = 1.0f;
  float gain = 1.0f;
  float inhib = 0.2f;
  float feedback = 1.0f;
  uint64_t seed = 1;
};

void usage() {
  std::printf(
      "snc_train -- frozen-structure e-prop weight training\n"
      "  --dataset synthetic|mnist   --encoder direct|poisson|latency\n"
      "  --structure dense|random-sparse|static-snc   --train-mode all|readout\n"
      "  --device cpu|cuda   --batch N (cuda minibatch)\n"
      "  --epochs N --num-steps N --hidden W|W1,W2,.. --dim N --classes N\n"
      "  --num-train N --num-test N --synapse-budget S --delay D\n"
      "  --lr R --w-init W --w-max W --gamma G --decay D --threshold T\n"
      "  --refractory R --noise S --gain G --inhib F --feedback F --seed S\n"
      "  --data-dir DIR --log-csv FILE\n");
}

template <typename T>
bool arg_v(int& i, int argc, char** argv, const char* flag, T& out) {
  if (std::strcmp(argv[i], flag) != 0) return false;
  if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", flag); std::exit(2); }
  out = static_cast<T>(std::stod(argv[++i]));
  return true;
}
bool arg_str(int& i, int argc, char** argv, const char* flag, std::string& out) {
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
    if (arg_str(i, argc, argv, "--dataset", o.dataset)) continue;
    if (arg_str(i, argc, argv, "--encoder", o.encoder)) continue;
    if (arg_str(i, argc, argv, "--structure", o.structure)) continue;
    if (arg_str(i, argc, argv, "--train-mode", o.train_mode)) continue;
    if (arg_str(i, argc, argv, "--device", o.device)) continue;
    if (arg_str(i, argc, argv, "--data-dir", o.data_dir)) continue;
    if (arg_v(i, argc, argv, "--batch", o.batch)) continue;
    if (arg_str(i, argc, argv, "--log-csv", o.log_csv)) continue;
    if (arg_v(i, argc, argv, "--num-steps", o.num_steps)) continue;
    if (arg_v(i, argc, argv, "--epochs", o.epochs)) continue;
    if (arg_str(i, argc, argv, "--hidden", o.hidden)) continue;
    if (arg_v(i, argc, argv, "--dim", o.dim)) continue;
    if (arg_v(i, argc, argv, "--classes", o.classes)) continue;
    if (arg_v(i, argc, argv, "--num-train", o.num_train)) continue;
    if (arg_v(i, argc, argv, "--num-test", o.num_test)) continue;
    if (arg_v(i, argc, argv, "--synapse-budget", o.synapse_budget)) continue;
    if (arg_v(i, argc, argv, "--delay", o.delay)) continue;
    if (arg_v(i, argc, argv, "--refractory", o.refractory)) continue;
    if (arg_v(i, argc, argv, "--noise", o.noise)) continue;
    if (arg_v(i, argc, argv, "--lr", o.lr)) continue;
    if (arg_v(i, argc, argv, "--w-init", o.w_init)) continue;
    if (arg_v(i, argc, argv, "--w-max", o.w_max)) continue;
    if (arg_v(i, argc, argv, "--gamma", o.gamma)) continue;
    if (arg_v(i, argc, argv, "--decay", o.decay)) continue;
    if (arg_v(i, argc, argv, "--threshold", o.threshold)) continue;
    if (arg_v(i, argc, argv, "--gain", o.gain)) continue;
    if (arg_v(i, argc, argv, "--inhib", o.inhib)) continue;
    if (arg_v(i, argc, argv, "--feedback", o.feedback)) continue;
    if (arg_v(i, argc, argv, "--seed", o.seed)) continue;
    std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
    usage();
    std::exit(2);
  }
  return o;
}

// {dim} + comma-separated hidden widths + {classes}. "256,256" -> 2 hidden layers.
std::vector<int> build_layers(int dim, const std::string& spec, int classes) {
  std::vector<int> L{dim};
  for (std::size_t i = 0; i < spec.size();) {
    std::size_t j = spec.find(',', i);
    if (j == std::string::npos) j = spec.size();
    std::string tok = spec.substr(i, j - i);
    if (!tok.empty()) L.push_back(std::stoi(tok));
    i = j + 1;
  }
  L.push_back(classes);
  return L;
}

SNNGraph build_graph(const Options& o, const std::vector<int>& layers) {
  if (o.structure == "dense") return make_dense(layers, 0.5f, o.delay);
  if (o.structure == "random-sparse")
    return make_random_sparse(layers, o.synapse_budget, 0.5f, o.delay, o.inhib, o.seed);
  if (o.structure == "static-snc" || o.structure == "dynamic-snc")
    return make_static_snc(layers, o.synapse_budget, 0.5f, o.delay, o.inhib, o.seed);
  std::fprintf(stderr, "unknown structure: %s\n", o.structure.c_str());
  std::exit(2);
}

// Split one dataset into the first `n_train` and next `n_test` rows. Synthetic
// rows are class-interleaved, so an index split keeps both halves balanced.
void split(const Dataset& all, int n_train, int n_test, Dataset& tr, Dataset& te) {
  tr.dim = te.dim = all.dim;
  tr.classes = te.classes = all.classes;
  const int total = all.size();
  n_train = std::min(n_train, total);
  n_test = std::min(n_test, total - n_train);
  for (int i = 0; i < n_train; ++i) { tr.x.push_back(all.x[i]); tr.y.push_back(all.y[i]); }
  for (int i = 0; i < n_test; ++i) {
    te.x.push_back(all.x[n_train + i]);
    te.y.push_back(all.y[n_train + i]);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Options o = parse(argc, argv);

  Encoder enc;
  if (!parse_encoder(o.encoder, enc)) {
    std::fprintf(stderr, "unknown encoder: %s\n", o.encoder.c_str());
    return 2;
  }

  Dataset train, test;
  if (o.dataset == "mnist") {
    train = load_mnist(o.data_dir, o.num_train, /*test_split=*/false);
    test = load_mnist(o.data_dir, o.num_test, /*test_split=*/true);
  } else {
    Dataset all = make_synthetic(o.num_train + o.num_test, o.dim, o.classes,
                                 o.noise, o.seed);
    split(all, o.num_train, o.num_test, train, test);
  }

  const std::vector<int> layers = build_layers(train.dim, o.hidden, train.classes);
  SNNGraph g = build_graph(o, layers);
  std::string err;
  if (!g.validate(err)) { std::fprintf(stderr, "graph invalid: %s\n", err.c_str()); return 1; }

  TrainConfig cfg;
  cfg.num_steps = o.num_steps;
  cfg.lif.threshold = o.threshold;
  cfg.lif.decay = o.decay;
  cfg.lif.refractory = o.refractory;
  cfg.encoder = enc;
  cfg.enc.gain = o.gain;
  cfg.lr = o.lr;
  cfg.w_init = o.w_init;
  cfg.w_max = o.w_max;
  cfg.surrogate_scale = o.gamma;
  cfg.feedback_scale = o.feedback;
  cfg.train_hidden = (o.train_mode != "readout");
  cfg.seed = o.seed;

  GraphStats gs = compute_stats(g);
  std::printf("structure=%s  %s\n", o.structure.c_str(), format_stats(gs).c_str());
  std::printf("train=%d test=%d classes=%d steps=%d lr=%.3g mode=%s chance=%.3f\n",
              train.size(), test.size(), train.classes, o.num_steps, o.lr,
              o.train_mode.c_str(), train.classes ? 1.0 / train.classes : 0.0);

  const bool want_gpu = o.device == "cuda";
  const bool use_gpu = want_gpu && cudatrain::available(cfg);
  std::printf("device=%s%s\n", use_gpu ? "cuda" : "cpu",
              want_gpu && !use_gpu ? " (cuda unavailable -> cpu fallback)" : "");

  std::ofstream csv;
  if (!o.log_csv.empty()) {
    csv.open(o.log_csv);
    csv << "epoch,loss,train_acc,test_acc,spikes,synaptic_events,energy\n";
  }
  auto log_epoch = [&](int e, const EpochStats& st) {
    std::printf("%5d %8.4f %10.3f %9.3f %12lld %13lld\n", e, st.loss,
                st.train_acc, st.test_acc, st.spikes, st.synaptic_events);
    if (csv)
      csv << e << ',' << st.loss << ',' << st.train_acc << ',' << st.test_acc
          << ',' << st.spikes << ',' << st.synaptic_events << ',' << st.energy << '\n';
  };
  std::printf("\nepoch     loss  train_acc  test_acc      spikes        events\n");

  if (use_gpu) {
    CudaTrainSession* sess = cudatrain::create(g, cfg, o.batch);
    std::printf("%5d %8s %10.3f %9.3f %12s %13s\n", 0, "-",
                cudatrain::evaluate(sess, train), cudatrain::evaluate(sess, test),
                "-", "-");
    for (int e = 1; e <= o.epochs; ++e)
      log_epoch(e, cudatrain::train_epoch(sess, train, test, e));
    cudatrain::destroy(sess);
  } else {
    Trainer trainer(g, cfg);
    std::printf("%5d %8s %10.3f %9.3f %12s %13s\n", 0, "-",
                trainer.evaluate(train), trainer.evaluate(test), "-", "-");
    for (int e = 1; e <= o.epochs; ++e)
      log_epoch(e, trainer.train_epoch(train, test, e));
  }
  if (!o.log_csv.empty()) std::printf("wrote %s\n", o.log_csv.c_str());
  return 0;
}
