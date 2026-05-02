// High-level simulator that ties together the structural grid, the per-neuron
// chemistry, and the regional energy field.
//
// One simulation step performs, in order:
//
//   1. chemistry_phase            -- per-neuron leaky integrate-and-fire,
//                                    parallel across neurons.
//   2. synaptic_transmission_phase-- spikes delivered to post-synaptic input
//                                    accumulators, paid for from local energy.
//   3. energy_regen_phase         -- regions regenerate a small amount of
//                                    energy; firing somas pay their cost.
//   4. sprouting_phase            -- active neurons attempt to extend a
//                                    neuron-state voxel into a free neighbour
//                                    under volume-exclusion + energy gates.
//   5. synaptogenesis_phase       -- when two neurons' tissue meet, an
//                                    eligible voxel can become SYNAPSE,
//                                    capped per-region.
//   6. pruning_phase              -- weak / silent synapses are removed and
//                                    their voxels demoted back to NEURON.
//
// Phases 1 and 2 are parallel; structural mutation phases (4, 5, 6) run
// serially because they mutate the grid, owner map and per-neuron body
// vectors, and the volume of work in those phases is bounded by activity
// (only firing or recently-active neurons drive growth) rather than total
// grid size.

#pragma once

#include "brain_grid.hpp"
#include "energy_field.hpp"
#include "neuron.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace snc {

struct SimConfig {
  // Grid dimensions. (X * Y) must be a multiple of 32; see BrainGrid.
  int X = 64;
  int Y = 64;
  int Z = 64;

  // Coarse region size used by the energy field. Smaller = tighter local
  // metabolic competition; larger = more global pooling.
  int region_size = 8;

  // Energy budget parameters.
  float energy_max = 100.0f;
  float energy_regen = 0.5f;
  float fire_cost = 5.0f;
  float sprout_cost = 0.5f;
  float synapse_form_cost = 2.0f;
  float synapse_use_cost = 0.05f;

  // Chemistry parameters.
  float fire_threshold = 1.0f;
  float potential_decay = 0.9f;
  float fire_rate_alpha = 0.05f;

  // Sprouting parameters.
  float sprout_prob = 0.4f;
  int sprout_attempts = 4;
  int max_neighbors_for_sprout = 4;  // volume-exclusion threshold

  // Synaptogenesis parameters.
  float synapse_form_prob = 0.6f;
  int max_synapses_per_region = 256;

  // Pruning / plasticity parameters.
  int prune_inactive_steps = 300;
  float prune_weight_floor = 0.02f;
  float weight_decay = 0.995f;
  float weight_potentiation = 0.05f;
  float weight_max = 1.0f;
  float initial_weight = 0.2f;

  // Three-factor learning (eligibility-trace + reward-modulated plasticity).
  // Trace is per-synapse; reward is a global scalar broadcast to all synapses.
  // Each synapse still updates only from its own local information, so the
  // per-neuron-independent computation property is preserved.
  float eligibility_decay = 0.92f;
  float eligibility_potentiation = 0.4f;  // pre+post coincidence increment
  float reward_lr = 0.08f;
  float input_drive_strength = 1.5f;      // scale on injected feature value

  // Energy-gated spike forwarding: a neuron's soma must have at least
  // `forward_min_energy` to enqueue all outgoing spikes. Below that, only
  // synapses with weight >= `forward_low_energy_floor` are forwarded -- the
  // analogue of vesicle-pool depletion sparing only the strongest synapses
  // when ATP is scarce. No randomness; the decision is a function of the
  // soma's local energy and the synapse's own weight.
  float forward_min_energy = 5.0f;
  float forward_low_energy_floor = 0.4f;

  // STDP (spike-timing-dependent plasticity). A synapse strengthens when
  // its delivery preceded the post's firing (LTP) and weakens when the
  // post fired before the delivery arrived (LTD). Both rules are local:
  // each synapse only consults its own last_delivery_step and the post's
  // last_fire_step. The amplitudes follow exponential temporal kernels,
  // matching the curves measured experimentally in cortex.
  float stdp_a_ltp = 0.04f;
  float stdp_a_ltd = 0.045f;          // LTD slightly larger -> net stable
  int   stdp_window = 20;
  float stdp_tau = 8.0f;

  // Homeostatic synaptic scaling. Each post neuron broadcasts (via its
  // own state) the discrepancy between its current total incoming weight
  // and the target. Each pre synapse rescales itself toward the target,
  // multiplicatively. This stabilises the network against runaway LTP /
  // collapse from LTD and is exactly the role of TNF-alpha / BDNF
  // retrograde signalling in cortex (Turrigiano 2008).
  float homeostatic_target_in = 1.5f;
  float homeostatic_rate = 0.004f;

  // Spine-retraction floor: a synapse whose efficacy (weight) has been
  // driven below this by STDP / scaling / decay is *physically* eliminated
  // in the next pruning sweep. This is the deterministic consequence of
  // the actin cytoskeleton no longer being able to maintain the spine.
  float spine_retraction_floor = 0.02f;

  unsigned seed = 1234u;
};

// Axis-aligned (x, y) bounding box used to scope cortical-area-specific
// operations (e.g. per-area neurogenesis). Inclusive on both sides.
struct BoundingBox {
  int x_lo, x_hi;
  int y_lo, y_hi;
};

// Counters describing what happened during the most recent step. Reset at the
// start of every step(); inspect via Simulator::last_stats() to plot
// developmental curves (synaptogenesis vs. pruning).
struct StepStats {
  int sprouts = 0;
  int synapses_formed = 0;
  int synapses_pruned = 0;
  int spikes = 0;
};

// Initial state inspired by late-fetal cortical development.
//
// The grid's z-axis is interpreted as the cortical depth axis:
//   z = 0 .. vz_thickness            -- ventricular zone (VZ): dense
//                                       progenitor / newborn-neuron layer
//   z = vz_thickness .. Z-cp_thickness -- intermediate / migrating zone:
//                                       neurons in transit, each with a small
//                                       leading process pointing towards the
//                                       pia (i.e. +z)
//   z = Z-cp_thickness .. Z-1        -- early cortical plate (CP): a few
//                                       early-arrived deep-layer neurons
//
// A sparse set of (x, y) columns are filled with BLOCKED voxels to act as
// radial-glia scaffolds: they cannot become synapses but provide structural
// guidance for later sprouting. The energy field is initialised with a
// vertical gradient -- high near the VZ where neurogenesis is most active,
// low near the CP which is metabolically quiet at this stage.
struct FetalSeed {
  int vz_thickness = 6;            // depth of the ventricular zone (voxels)
  int cp_thickness = 4;            // depth of the cortical plate band
  float radial_glia_density = 0.03f;  // fraction of (x,y) columns to fill
  int vz_neurons = 200;            // dense progenitors at the VZ
  int migrating_neurons = 80;      // mid-z neurons with leading process
  int cortical_plate_neurons = 16; // early-arrived deep-layer neurons
  int leading_process_voxels = 2;  // length of migrating leading process

  // Fractions in [0, 1] of the simulator's `energy_max`.
  float vz_energy_scale = 1.0f;
  float cp_energy_scale = 0.2f;
};

class Simulator {
 public:
  explicit Simulator(SimConfig cfg);

  // Place `n` neurons with a single-voxel soma at random free locations.
  void seed_neurons(int n);

  // Place neurons + radial-glia scaffold + energy gradient in a pattern that
  // reflects late-fetal cortical development. See `FetalSeed` for parameters.
  void seed_fetal(const FetalSeed& f = {});

  // Place exactly one neuron at the given voxel and return its id. Returns 0
  // if the voxel is out of bounds or already occupied. Used by training
  // demos to place INPUT / OUTPUT neurons at chosen anatomical locations.
  uint32_t add_neuron_at(int x, int y, int z);

  // Set or query a neuron's polarity (Dale's principle). By default seeded
  // neurons are EXCITATORY; use this to designate the ~20% GABAergic pool
  // that the network needs for sparse coding and output decoding.
  void set_polarity(uint32_t neuron_id, NeuronPolarity pol);

  // Randomly assign INHIBITORY polarity to approximately `fraction` of the
  // currently-existing neurons (uses the simulator's rng). Used after
  // seeding to install the cortical E/I balance in one call.
  void randomize_polarity(float inhibitory_fraction);

  // Install a synapse from `pre_id` to `post_id` directly, bypassing the
  // sprouting / synaptogenesis machinery. Useful for tests (where we want
  // exact control of the wiring) and for bootstrapping demos that need an
  // initial input -> output path before structural plasticity has had time
  // to grow one. The grid is *not* mutated; structural plasticity will
  // continue to evolve the connectome from here.
  void install_synapse(uint32_t pre_id, uint32_t post_id, float weight,
                       int conduction_delay);

  // Add `n` newly-born neurons inside the current ventricular-zone band
  // (tracked across grow_volume calls). When `area` is non-null the new
  // neurons are restricted to the given (x, y) box; this is how the demo
  // gives different cortical areas different developmental schedules.
  // Returns the number actually placed.
  int birth_neurons(int n, const BoundingBox* area = nullptr);

  // Statistics for the most-recently completed step. Updated at the end of
  // step(). Useful for plotting synaptogenesis / pruning curves.
  const StepStats& last_stats() const noexcept { return last_stats_; }

  // Expand the simulated volume by `dx_each_side` voxels on each side along
  // the x axis (and analogously for y, z). Existing tissue is centered in
  // the new volume; all neuron coordinates, the owner map, the energy field
  // and the structural grid are migrated together. New voxels are EMPTY at
  // full energy. Each growth amount must be a multiple of `region_size` so
  // the energy field can be copied region-by-region without resampling.
  //
  // This is the analogue of physical brain growth: the matrix itself gets
  // bigger over developmental time, giving room for new sprouting.
  void grow_volume(int dx_each_side, int dy_each_side, int dz_each_side);

  // Mutable access to the runtime config so callers can ramp parameters
  // (e.g. tighten pruning over development) without rebuilding the sim.
  SimConfig& mutable_config() noexcept { return cfg_; }

  // External input injection (e.g. sensory stimulation). Adds `amount` to the
  // neuron's input accumulator for the current step.
  void inject_input(uint32_t neuron_id, float amount);

  // -------- Data-driven training API ------------------------------------
  //
  // Training preserves the per-neuron-independent computation rule. The only
  // global signal is a scalar `reward` broadcast to every synapse, modelled
  // after a neuromodulator (e.g. dopamine) diffusing across the cortex; each
  // synapse still updates from its own local eligibility trace.

  // Mark `neuron_id` as INPUT or OUTPUT, attached to feature/class `channel`.
  // Pass NeuronRole::INTERNAL to clear the role.
  void set_role(uint32_t neuron_id, NeuronRole role, int channel);

  // Drive every INPUT neuron whose channel matches an index in `features`.
  // For each such neuron the input_acc is set to
  // `features[channel] * input_drive_strength`. Replaces (not adds to) the
  // existing input_acc for INPUT neurons so external features always
  // dominate; INTERNAL neurons are untouched.
  void apply_input_pattern(const float* features, int n_features);

  // Read the mean fire_rate_ema across OUTPUT neurons grouped by channel.
  // `out[i]` will hold the mean rate of OUTPUT neurons with channel == i;
  // channels with no neurons get 0.
  void read_output(float* out, int n_classes) const;

  // Broadcast a scalar reward (typically in [-1, +1]). For every outgoing
  // synapse: `weight += reward_lr * reward * eligibility`, then clamped to
  // [0, weight_max]. The eligibility trace is *not* cleared by this call;
  // callers may zero it explicitly with `clear_eligibility()` between
  // independent trials if desired.
  void apply_reward(float reward);

  // Per-class reward variant. Each synapse looks up its post-synaptic
  // neuron's role/channel and applies the appropriate reward:
  //   - post is OUTPUT with valid channel: rewards[channel]
  //   - otherwise:                          internal_reward
  // This is still strictly local on the synapse side (it consults only the
  // post it already knows about), but lets the global teaching signal
  // distinguish between "should fire" and "should not fire" outputs --
  // necessary to avoid the degenerate "always predict class X" attractor
  // that a single scalar reward cannot escape.
  void apply_reward_per_class(const float* rewards, int n_classes,
                              float internal_reward = 0.0f);

  // Reset every synapse's eligibility trace to zero (between independent
  // trials in a training loop).
  void clear_eligibility();

  // -------- Sleep: persist / restore the entire brain state --------------
  //
  // The "matrix = brain structure and state" snapshot captures everything
  // needed to resume the simulation later: the 2-bit structural grid,
  // every neuron's chemistry / queues / outgoing synapses, the regional
  // energy field, the owner map, the simulation step and the VZ band.
  // Returns true on success.
  bool save_state(const char* path) const;
  bool load_state(const char* path);

  // Run one full simulation step.
  void step();

  int current_step() const noexcept { return step_; }
  std::size_t neuron_count() const noexcept { return neurons_.size(); }
  std::size_t total_synapses() const noexcept;
  std::size_t total_neuron_voxels() const noexcept;
  float total_energy() const noexcept;

  const BrainGrid& grid() const noexcept { return grid_; }
  const std::vector<Neuron>& neurons() const noexcept { return neurons_; }
  const EnergyField& energy() const noexcept { return energy_; }

 private:
  std::size_t lin(int x, int y, int z) const noexcept {
    return static_cast<std::size_t>(x) +
           static_cast<std::size_t>(y) * cfg_.X +
           static_cast<std::size_t>(z) * cfg_.X * cfg_.Y;
  }

  // Per-step pipeline. The synaptic transmission has been factored into
  // four explicit queue stages (incoming -> integrate -> dispatch ->
  // scheduler). Pruning is *not* a probabilistic decision -- it is the
  // deterministic structural consequence of weights driven below the spine
  // retraction floor by STDP, homeostatic scaling and slow decay.
  //
  //   integrate_incoming_phase   queue 1 -> potential
  //   chemistry_phase            potential -> fire decision
  //   stdp_phase                 LTP from this-step post-firings
  //                              + eligibility kick for reward learning
  //   fire_dispatch_phase        fire -> queue 4 (energy-gated)
  //   scheduler_dispatch_phase   queue 4 -> queue 1, plus LTD on delivery
  //   homeostatic_phase          per-post synaptic scaling
  //   pruning_phase              spine retraction: weight < floor -> remove
  void integrate_incoming_phase();
  void chemistry_phase();
  void stdp_phase();
  void fire_dispatch_phase();
  void scheduler_dispatch_phase();
  void homeostatic_phase();
  void sprouting_phase();
  void synaptogenesis_phase();
  void pruning_phase();
  void energy_regen_phase();

  bool try_set_neuron(int x, int y, int z, uint32_t owner_id);

  SimConfig cfg_;
  BrainGrid grid_;
  EnergyField energy_;

  // Auxiliary owner map: 0 means "no owner", otherwise neuron id.
  // This is *not* part of the structural 2-bit matrix (which intentionally
  // carries no identity information). It exists only to let synaptogenesis
  // tell whether neighbouring NEURON voxels belong to two different cells.
  std::vector<uint32_t> owner_;

  std::vector<Neuron> neurons_;

  StepStats last_stats_{};

  // Currently-active ventricular-zone band, in world coordinates. Updated by
  // seed_fetal and shifted by grow_volume so birth_neurons keeps placing
  // newborns at the (now-displaced) inner surface of the cortex.
  int vz_lo_ = 0;
  int vz_hi_ = 0;

  int step_ = 0;
  std::mt19937 rng_;
};

}  // namespace snc
