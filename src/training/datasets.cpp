#include "snc/dataset.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>

namespace snc {

Dataset make_synthetic(int n, int dim, int classes, float noise, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> uni(0.0f, 1.0f);
  std::normal_distribution<float> ng(0.0f, noise);
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
      s[k] = std::clamp(proto[c][k] + ng(rng), 0.0f, 1.0f);
    d.x.push_back(std::move(s));
    d.y.push_back(c);
  }
  return d;
}

static uint32_t read_be32(std::ifstream& f) {
  unsigned char b[4];
  f.read(reinterpret_cast<char*>(b), 4);
  return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) |
         uint32_t(b[3]);
}

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

}  // namespace snc
