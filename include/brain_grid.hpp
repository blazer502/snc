// Structural grid: a 3D matrix of 2-bit cells encoding the spatial state of
// the neural tissue. The structural matrix is fully decoupled from the
// per-neuron chemistry; only spatial occupation, synaptic contact and
// volume-exclusion / no-synapse zones are stored here.
//
// Cell encoding (2 bits):
//   0 EMPTY    : free space, available for sprouting
//   1 NEURON   : occupied by some neuron's body (soma / dendrite / axon)
//   2 SYNAPSE  : a synaptic contact voxel between two neurons
//   3 BLOCKED  : tissue that may exist but cannot become a synapse
//                (e.g. structural scaffold, glia, vasculature analogue)
//
// 32 cells are packed into a single uint64_t word. Iteration that intends to
// be parallelised across z-slices must satisfy `(X * Y) % 32 == 0` so that no
// two slices share a packed word; the constructor asserts this invariant.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace snc {

class BrainGrid {
 public:
  enum Cell : uint8_t {
    EMPTY = 0,
    NEURON = 1,
    SYNAPSE = 2,
    BLOCKED = 3,
  };

  BrainGrid(int X, int Y, int Z);

  int X() const noexcept { return X_; }
  int Y() const noexcept { return Y_; }
  int Z() const noexcept { return Z_; }
  std::size_t volume() const noexcept {
    return static_cast<std::size_t>(X_) * Y_ * Z_;
  }

  bool in_bounds(int x, int y, int z) const noexcept {
    return static_cast<unsigned>(x) < static_cast<unsigned>(X_) &&
           static_cast<unsigned>(y) < static_cast<unsigned>(Y_) &&
           static_cast<unsigned>(z) < static_cast<unsigned>(Z_);
  }

  int linear(int x, int y, int z) const noexcept {
    return x + y * X_ + z * X_ * Y_;
  }

  Cell get(int x, int y, int z) const noexcept {
    return read(front_, linear(x, y, z));
  }
  Cell get_back(int x, int y, int z) const noexcept {
    return read(back_, linear(x, y, z));
  }
  void set(int x, int y, int z, Cell c) noexcept {
    write(front_, linear(x, y, z), c);
  }
  void set_back(int x, int y, int z, Cell c) noexcept {
    write(back_, linear(x, y, z), c);
  }

  // Double-buffer helpers for callers that prefer synchronous (Game-of-Life
  // style) updates: read from front, write to back, then swap.
  void swap_buffers() noexcept { front_.swap(back_); }
  void copy_front_to_back() { back_ = front_; }

  // Raw access to the packed front buffer; used by the sleep (save/load)
  // path which writes the raw bits to disk. The number of valid words is
  // `(X*Y*Z + 31) / 32`.
  const std::vector<uint64_t>& raw_front() const noexcept { return front_; }
  std::vector<uint64_t>& raw_front() noexcept { return front_; }

 private:
  static Cell read(const std::vector<uint64_t>& buf, int lin) noexcept {
    const int w = lin >> 5;
    const int s = (lin & 31) << 1;
    return static_cast<Cell>((buf[w] >> s) & 0x3ULL);
  }
  static void write(std::vector<uint64_t>& buf, int lin, Cell c) noexcept {
    const int w = lin >> 5;
    const int s = (lin & 31) << 1;
    const uint64_t mask = 0x3ULL << s;
    buf[w] = (buf[w] & ~mask) | (static_cast<uint64_t>(c) << s);
  }

  int X_, Y_, Z_;
  std::vector<uint64_t> front_, back_;
};

}  // namespace snc
