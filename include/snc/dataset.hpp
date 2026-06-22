// Datasets for the SNN substrate (new-plan.md section 7.2 / 3.1).
//
// A Dataset is just normalised feature vectors (values in [0, 1]) plus integer
// labels. Encoders turn the feature rows into InputEvent streams. The synthetic
// generator makes the pipeline runnable with no external files; load_mnist
// reads standard uncompressed IDX files when a data directory is provided.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace snc {

struct Dataset {
  int dim = 0;                         // features per sample
  int classes = 0;                     // number of distinct labels
  std::vector<std::vector<float>> x;   // [num_samples][dim], in [0,1]
  std::vector<int> y;                  // [num_samples]

  int size() const { return static_cast<int>(x.size()); }
};

// Class-separable synthetic data: each class is a random prototype + Gaussian
// noise. Deterministic for a fixed seed. `noise` controls difficulty.
Dataset make_synthetic(int n, int dim, int classes, float noise, uint64_t seed);

// Minimal MNIST IDX loader. Expects train-images-idx3-ubyte +
// train-labels-idx1-ubyte (uncompressed) under `dir`; loads up to `n` samples.
// Exits with a clear message if the files are missing.
Dataset load_mnist(const std::string& dir, int n);

}  // namespace snc
