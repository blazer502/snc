// snc_export -- dump an SNC graph to a flat binary for the PyTorch bridge.
//
// Generates a sparse spiking graph with the SNC generators (the same code the
// C++ runtime/trainers use) and writes its CSR connectivity so the Python /
// PyTorch surrogate-gradient trainer can load the *exact* topology.
//
//   ./snc_export --structure static-snc --layers 784,256,10 \
//                --synapse-budget 40000 --seed 1 --out graph.bin
//
// Binary layout (all little-endian int32):
//   magic, N, S, num_input_channels, num_output_channels,
//   pre[S], post[S], delays[S], role[N], channel[N]

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "snc/snn_graph.hpp"

namespace {
using namespace snc;

std::vector<int> parse_ints(const std::string& spec) {
  std::vector<int> v;
  for (std::size_t i = 0; i < spec.size();) {
    std::size_t j = spec.find(',', i);
    if (j == std::string::npos) j = spec.size();
    std::string tok = spec.substr(i, j - i);
    if (!tok.empty()) v.push_back(std::stoi(tok));
    i = j + 1;
  }
  return v;
}

void put(std::ofstream& f, int32_t v) {
  f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
}  // namespace

int main(int argc, char** argv) {
  std::string structure = "static-snc", layers_spec = "784,256,10", out = "graph.bin";
  int budget = 40000, delay = 1;
  float inhib = 0.2f;
  uint64_t seed = 1;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--structure") structure = next();
    else if (a == "--layers") layers_spec = next();
    else if (a == "--out") out = next();
    else if (a == "--synapse-budget") budget = std::stoi(next());
    else if (a == "--delay") delay = std::stoi(next());
    else if (a == "--inhib") inhib = std::stof(next());
    else if (a == "--seed") seed = std::stoull(next());
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
  }

  const std::vector<int> layers = parse_ints(layers_spec);
  SNNGraph g;
  if (structure == "dense") g = make_dense(layers, 0.5f, delay);
  else if (structure == "random-sparse") g = make_random_sparse(layers, budget, 0.5f, delay, inhib, seed);
  else if (structure == "static-snc" || structure == "dynamic-snc")
    g = make_static_snc(layers, budget, 0.5f, delay, inhib, seed);
  else { std::fprintf(stderr, "unknown structure: %s\n", structure.c_str()); return 2; }

  std::string err;
  if (!g.validate(err)) { std::fprintf(stderr, "invalid graph: %s\n", err.c_str()); return 1; }

  // Derive pre[] (CSR row owner) from row_ptr.
  const int N = g.num_neurons, S = g.num_synapses();
  std::vector<int32_t> pre(S);
  for (int i = 0; i < N; ++i)
    for (int e = g.row_ptr[i]; e < g.row_ptr[i + 1]; ++e) pre[e] = i;

  std::ofstream f(out, std::ios::binary);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", out.c_str()); return 1; }
  put(f, 0x534E4347);  // 'SNCG'
  put(f, N);
  put(f, S);
  put(f, g.num_input_channels);
  put(f, g.num_output_channels);
  f.write(reinterpret_cast<const char*>(pre.data()), (std::streamsize)S * 4);
  f.write(reinterpret_cast<const char*>(g.post_ids.data()), (std::streamsize)S * 4);
  f.write(reinterpret_cast<const char*>(g.delays.data()), (std::streamsize)S * 4);
  std::vector<int32_t> role(N), channel(N);
  for (int i = 0; i < N; ++i) {
    role[i] = static_cast<int32_t>(g.role[i]);
    channel[i] = g.channel[i];
  }
  f.write(reinterpret_cast<const char*>(role.data()), (std::streamsize)N * 4);
  f.write(reinterpret_cast<const char*>(channel.data()), (std::streamsize)N * 4);

  std::printf("wrote %s: %s  N=%d S=%d in=%d out=%d  %s\n", out.c_str(),
              structure.c_str(), N, S, g.num_input_channels, g.num_output_channels,
              format_stats(compute_stats(g)).c_str());
  return 0;
}
