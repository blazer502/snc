// snc_cotrain -- two-timescale structure + weight co-training (new-plan.md 8.2).
//
// Inner loop: train weights for `--inner` epochs with e-prop (fast path).
// Outer loop: run one structural update -- prune the weakest synapses and grow
// the same number of new local, demand-driven ones -- then recompile the graph
// and continue, carrying the learned weights across.
//
// `--grow 0` disables structural mutation (static baseline), so dynamic vs
// static structure can be compared at an equal synapse budget and equal total
// epoch count.
//
//   ./snc_cotrain --outer 12 --inner 2 --grow 300         # dynamic
//   ./snc_cotrain --outer 12 --inner 2 --grow 0           # static baseline

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "snc/connectome.hpp"
#include "snc/dataset.hpp"
#include "snc/snn_graph.hpp"
#include "snc/trainer.hpp"

namespace {
using namespace snc;

struct Options {
  std::string dataset = "synthetic";
  std::string encoder = "poisson";
  std::string data_dir, log_csv;
  int outer = 12;
  int inner = 2;
  int grow = 300;            // synapses pruned+regrown per structural round (0=static)
  int num_steps = 25;
  int hidden = 200;
  int dim = 200;
  int classes = 10;
  int num_train = 300;
  int num_test = 100;
  int synapse_budget = 16000;
  int delay = 1;
  int refractory = 1;
  int protect = 2;
  float noise = 0.35f;
  float lr = 0.08f;
  float w_init = 0.02f;
  float w_grow = 0.02f;
  float w_max = 4.0f;
  float gamma = 0.3f;
  float decay = 0.9f;
  float threshold = 1.0f;
  float gain = 1.0f;
  float locality = 0.2f;
  float feedback = 1.0f;
  uint64_t seed = 1;
};

void usage() {
  std::printf(
      "snc_cotrain -- two-timescale structure+weight co-training\n"
      "  --outer N --inner K --grow G(0=static) --structural-budget S\n"
      "  --dataset synthetic|mnist --encoder direct|poisson|latency\n"
      "  --num-steps N --hidden N --dim N --classes N --num-train N --num-test N\n"
      "  --lr R --w-init W --w-grow W --gamma G --decay D --threshold T\n"
      "  --refractory R --noise S --gain G --locality F --protect R --feedback F\n"
      "  --seed S --data-dir DIR --log-csv FILE\n");
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
    if (arg_str(i, argc, argv, "--data-dir", o.data_dir)) continue;
    if (arg_str(i, argc, argv, "--log-csv", o.log_csv)) continue;
    if (arg_v(i, argc, argv, "--outer", o.outer)) continue;
    if (arg_v(i, argc, argv, "--inner", o.inner)) continue;
    if (arg_v(i, argc, argv, "--grow", o.grow)) continue;
    if (arg_v(i, argc, argv, "--num-steps", o.num_steps)) continue;
    if (arg_v(i, argc, argv, "--hidden", o.hidden)) continue;
    if (arg_v(i, argc, argv, "--dim", o.dim)) continue;
    if (arg_v(i, argc, argv, "--classes", o.classes)) continue;
    if (arg_v(i, argc, argv, "--num-train", o.num_train)) continue;
    if (arg_v(i, argc, argv, "--num-test", o.num_test)) continue;
    if (arg_v(i, argc, argv, "--structural-budget", o.synapse_budget)) continue;
    if (arg_v(i, argc, argv, "--delay", o.delay)) continue;
    if (arg_v(i, argc, argv, "--refractory", o.refractory)) continue;
    if (arg_v(i, argc, argv, "--protect", o.protect)) continue;
    if (arg_v(i, argc, argv, "--noise", o.noise)) continue;
    if (arg_v(i, argc, argv, "--lr", o.lr)) continue;
    if (arg_v(i, argc, argv, "--w-init", o.w_init)) continue;
    if (arg_v(i, argc, argv, "--w-grow", o.w_grow)) continue;
    if (arg_v(i, argc, argv, "--gamma", o.gamma)) continue;
    if (arg_v(i, argc, argv, "--decay", o.decay)) continue;
    if (arg_v(i, argc, argv, "--threshold", o.threshold)) continue;
    if (arg_v(i, argc, argv, "--gain", o.gain)) continue;
    if (arg_v(i, argc, argv, "--locality", o.locality)) continue;
    if (arg_v(i, argc, argv, "--feedback", o.feedback)) continue;
    if (arg_v(i, argc, argv, "--seed", o.seed)) continue;
    std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
    usage();
    std::exit(2);
  }
  return o;
}

void split(const Dataset& all, int n_train, int n_test, Dataset& tr, Dataset& te) {
  tr.dim = te.dim = all.dim;
  tr.classes = te.classes = all.classes;
  const int total = all.size();
  n_train = std::min(n_train, total);
  n_test = std::min(n_test, total - n_train);
  for (int i = 0; i < n_train; ++i) { tr.x.push_back(all.x[i]); tr.y.push_back(all.y[i]); }
  for (int i = 0; i < n_test; ++i) {
    te.x.push_back(all.x[n_train + i]); te.y.push_back(all.y[n_train + i]);
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
  const int dim = train.dim, classes = train.classes;

  Connectome con = Connectome::layered_local({dim, o.hidden, classes},
                                             o.synapse_budget, o.delay, o.w_init,
                                             o.seed);

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
  cfg.train_hidden = true;
  cfg.seed = o.seed;

  StructConfig scfg;
  scfg.grow_per_epoch = o.grow;
  scfg.delay = o.delay;
  scfg.w_grow_init = o.w_grow;
  scfg.locality_window = o.locality;
  scfg.protect_rounds = o.protect;
  scfg.seed = o.seed;

  const char* mode = o.grow > 0 ? "dynamic" : "static";
  std::printf("mode=%s  dataset=%s  layers=[%d,%d,%d]  budget=%d  synapses=%d\n", mode,
              o.dataset.c_str(), dim, o.hidden, classes, o.synapse_budget,
              con.num_synapses());
  std::printf("train=%d test=%d classes=%d  outer=%d inner=%d grow=%d  chance=%.3f\n",
              train.size(), test.size(), classes, o.outer, o.inner, o.grow,
              classes ? 1.0 / classes : 0.0);
  std::printf("\nround  synapses   pruned   grown  train_acc  test_acc       spikes\n");

  std::ofstream csv;
  if (!o.log_csv.empty()) {
    csv.open(o.log_csv);
    csv << "round,synapses,pruned,grown,train_acc,test_acc,spikes\n";
  }

  int global_epoch = 0;
  double best_test = 0.0;
  for (int r = 0; r < o.outer; ++r) {
    SNNGraph g = con.to_graph();
    Trainer tr(g, cfg);
    if (r > 0) tr.set_weights(con.edge_weights());  // carry learned weights
    tr.reset_stats();

    EpochStats es;
    for (int k = 0; k < o.inner; ++k) es = tr.train_epoch(train, test, ++global_epoch);
    con.set_edge_weights(tr.weights());  // persist learned weights onto edges
    best_test = std::max(best_test, es.test_acc);

    StructReport sr{0, 0, con.num_synapses()};
    if (o.grow > 0 && r < o.outer - 1) {
      ActivityStats as{tr.syn_deliveries(), tr.neuron_fires()};
      sr = con.structural_update(as, scfg);
    }
    std::printf("%5d %9d %8d %7d %10.3f %9.3f %12lld\n", r, con.num_synapses(),
                sr.pruned, sr.grown, es.train_acc, es.test_acc, es.spikes);
    if (csv)
      csv << r << ',' << con.num_synapses() << ',' << sr.pruned << ',' << sr.grown
          << ',' << es.train_acc << ',' << es.test_acc << ',' << es.spikes << '\n';
  }

  std::printf("\nbest test_acc = %.3f (%s structure, %d synapses)\n", best_test,
              mode, con.num_synapses());
  if (!o.log_csv.empty()) std::printf("wrote %s\n", o.log_csv.c_str());
  return 0;
}
