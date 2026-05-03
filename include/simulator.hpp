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

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <unordered_map>
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

  // BCM metaplasticity (Bienenstock-Cooper-Munro). The post's running
  // baseline activity acts as a sliding threshold for LTP. Above the
  // threshold LTP magnitude shrinks; below it normal LTP applies. The
  // threshold itself moves with a slow EMA of `fire_rate_ema`.
  float bcm_baseline_alpha = 0.0005f;     // very slow EMA
  float bcm_target = 0.08f;               // (unused -- baseline is per-neuron)

  // Heterosynaptic damping. After STDP each post neuron measures the
  // total LTP magnitude it received and shrinks every one of its
  // incoming synapses by a small fraction of that, mimicking the
  // post-synaptic-density resource competition.
  float heterosynaptic_damp = 0.003f;

  // Sensitive period: a developmental window early in the simulation
  // during which STDP amplitude is boosted, then exponentially decays
  // to baseline. Captures critical-period plasticity (e.g. the language
  // acquisition window in primary auditory cortex).
  int   sensitive_period_tau = 6000;      // exponential decay constant in steps
  float sensitive_period_boost = 1.3f;    // peak multiplier on STDP at step 0

  // Synaptic-tagging-and-capture. LTP events above `tag_threshold` set a
  // long-timescale consolidation tag; tags decay slowly and protect the
  // synapse's contact site from spine retraction.
  float tag_decay = 0.9985f;
  float tag_threshold = 0.01f;             // LTP delta required to set a tag
  float tag_protection = 0.3f;             // tag value above which spines are protected

  // Neuromodulator pool. Three additional global scalars beyond dopamine
  // (which is the per-class reward broadcast):
  //   acetylcholine -> multiplies STDP gain (attention / learning rate)
  //   norepinephrine -> shifts BCM threshold (novelty / arousal)
  //   serotonin -> multiplies reward consolidation rate (mood / valence)
  // Implemented as plain config scalars so the demo can crank attention
  // up at the start of a trial, etc., without breaking any locality
  // property -- each neuron / synapse just reads the global as if it were
  // a diffuse molecule bathing the cortex.
  float acetylcholine_level = 1.0f;
  float norepinephrine_level = 0.0f;
  float serotonin_level = 1.0f;

  // Aversive-learning amplification. Real brains weight punishment more
  // strongly than equivalent reward (negativity bias). When
  // `apply_aversive` is called, the per-synapse weight change is
  // `aversive_amplification * reward_lr * intensity * eligibility`.
  float aversive_amplification = 2.0f;

  // Refractory period (in steps). After firing, a neuron's potential is
  // forced to 0 for `refractory_steps` steps -- it cannot fire again
  // during this window even if its input would otherwise drive it. Real
  // cortical neurons have ~3-5ms absolute refractory + ~10ms relative;
  // mapped to our coarser step units this typically corresponds to a
  // few-step block. Default 0 keeps legacy demos identical.
  int refractory_steps = 0;

  // Probability that a spike actually crosses each synapse on dispatch.
  // Real cortical synapses release neurotransmitter with ~0.2-0.6
  // probability per spike; the resulting baseline noise is essential for
  // breaking the deterministic attractors that pure spike+threshold
  // models tend to fall into. 1.0 = legacy deterministic transmission.
  float release_probability = 1.0f;

  // Short-term plasticity (Tsodyks-Markram style). Each successful
  // release depletes the per-synapse vesicle pool by `release_depression`
  // (fraction of full pool); per step the pool recovers by
  // `release_recovery`. The effective release magnitude scales with the
  // current pool fraction, producing short-term depression on bursts
  // and recovery during quiescence. Default 0/0 keeps `vesicle_state`
  // pinned at 1.0 (no STP) for legacy demos.
  float release_depression = 0.0f;
  float release_recovery   = 0.0f;

  // Tripartite-synapse / astrocyte calcium modulation. A per-region
  // calcium accumulator integrates synaptic activity (every release
  // adds `astrocyte_release_increment` to its region), decays each
  // step by `astrocyte_decay`, and during STDP multiplies the LTP
  // amplitude by (1 + astrocyte_modulation * local_calcium). High
  // local synaptic traffic therefore *gates* plasticity in that
  // patch of cortex -- the astrocyte is reporting "this region is
  // working hard, go ahead and consolidate". Default 0 disables.
  float astrocyte_release_increment = 0.0f;
  float astrocyte_decay = 0.99f;
  float astrocyte_modulation = 0.0f;

  // Dendritic-compartment integration. A neuron with `n_branches > 1`
  // sums incoming spikes per branch in `branch_potential`. integrate
  // converts each branch's potential into a soma contribution:
  //   if branch_potential[b] >= dendritic_threshold:
  //       soma_drive += dendritic_spike_amplitude
  //       branch_potential[b] = 0   (the branch fired its NMDA plateau)
  //   else:
  //       soma_drive += branch_potential[b] * dendritic_passive_gain
  //       branch_potential[b] *= dendritic_decay
  // Defaults make the multi-branch path a no-op for n_branches == 1.
  float dendritic_threshold = 1.0e9f;     // never triggered by default
  float dendritic_spike_amplitude = 1.5f;
  float dendritic_passive_gain = 1.0f;
  float dendritic_decay = 0.0f;            // 0 = instantaneous (legacy)

  // When sprouting / synaptogenesis create a new synapse, its `branch`
  // field is set to this. Demos that want sprouted plasticity to land
  // on a different dendrite from hand-installed priors can flip this.
  uint8_t synaptogenesis_default_branch = 0;

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

// Position-binned neuron features (cortical-map-style instrumentation).
// Each populated bin (one `region_size` cube) reports aggregate statistics
// over the neurons whose soma lives there. Two roles:
//   - read-only debug / visualisation (cortical map view)
//   - soft prior for newly-born neurons -- a freshly placed neuron
//     inherits the local BCM `activity_baseline` so its plasticity is
//     calibrated to its anatomical neighbourhood, mirroring how cortical
//     columns in real cortex acquire similar tuning to their neighbours.
struct PositionFeatures {
  int n_neurons = 0;
  float mean_fire_rate_ema = 0.0f;
  float mean_activity_baseline = 0.0f;
  float mean_incoming_weight = 0.0f;
  // Tuning curve: total weight of incoming INPUT->bin synapses
  // indexed by the source's channel. argmax(tuning_curve) is the
  // bin's dominant input channel; the magnitude reports how strongly
  // it is wired to its preferred stimulus. Empty until refreshed.
  std::vector<float> tuning_curve;
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

  // ------------------ "DNA"-level innate priors ------------------------
  //
  // Real brains do not start as undifferentiated neural sheets; the
  // fetus already contains rough functional subdivisions that genetics
  // wires before any experience: brainstem oscillators, thalamic relay
  // nuclei, the amygdala etc. These extra cohorts are placed alongside
  // the cortex so the network has the analogue of innate reflex /
  // sensory / aversive circuits the moment seed_fetal returns.

  // GABAergic subtype distribution among the cortical neurons. The
  // fractions below are taken from rodent cortex (Tremblay et al. 2016).
  float frac_pv  = 0.14f;   // parvalbumin basket cells (perisomatic)
  float frac_sst = 0.04f;   // somatostatin Martinotti cells (dendritic)
  float frac_vip = 0.02f;   // vasoactive intestinal peptide (disinhibitory)

  // Brainstem analogue: a small population of always-on tonic neurons
  // placed in a narrow z-stripe near the bottom of the volume. They
  // start each step with a positive bias so the network has spontaneous
  // baseline drive even before any stimulus arrives -- the simulator's
  // analogue of brainstem rhythm generators (locus coeruleus, raphe
  // nuclei, etc.).
  int brainstem_neurons = 12;

  // Thalamic relay analogue: a population that sits between sensory
  // INPUT neurons (after the demo wires them) and the cortex, providing
  // an innate sensory hub. Pre-wired connections are *not* installed by
  // seed_fetal -- the demo decides whether to use them as a relay --
  // but the cells exist as recognisable anatomical landmarks.
  int thalamic_relay_neurons = 16;

  // Innate aversive nucleus (amygdala analogue): a small group of cells
  // that the demo can wire to "danger" inputs so a special pattern fires
  // them automatically. Whatever they fire onto is a candidate for the
  // sim's apply_aversive() to be triggered against. Like the brainstem
  // population, presence-only here; downstream wiring is up to demos.
  int aversive_nucleus_neurons = 6;
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
  // continue to evolve the connectome from here. `branch` selects which
  // dendritic branch on the post neuron the synapse lands on (default 0).
  //
  // `innate_tag` sets the synapse's consolidation_tag to that value at
  // install time -- a tag >= cfg.tag_protection (default 0.3) shields
  // the synapse from spine retraction even when it stays silent for a
  // long stretch. Pass 0 (default) for normal "use it or lose it"
  // dynamics; pass 1.0 for a fully-protected innate / labelled-line
  // connection that survives multi-stage curricula where its source
  // channel may not fire for a while.
  void install_synapse(uint32_t pre_id, uint32_t post_id, float weight,
                       int conduction_delay, uint8_t branch = 0,
                       float innate_tag = 0.0f);

  // Configure the number of dendritic branches on a neuron. Branches are
  // independent integrators -- a synapse on branch 0 cannot pool into the
  // same dendritic spike as a synapse on branch 1. Default is 1 (legacy
  // single-compartment behaviour).
  void set_branches(uint32_t neuron_id, uint8_t n_branches);

  // Override the dendritic-spike threshold or passive-gain on a single
  // branch of a single neuron. Useful for circuits where the
  // hand-installed innate-prior branch must spike easily while the
  // synaptogenesis-default branch must stay quiet under bulk noise.
  // Pass NaN (or omit) to fall back to the global `cfg` value for
  // that branch.
  void set_branch_threshold(uint32_t neuron_id, uint8_t branch,
                            float threshold);
  void set_branch_passive_gain(uint32_t neuron_id, uint8_t branch,
                               float gain);

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

  // Shrink the simulated volume by trimming `dx_each_side` voxels from
  // each side along x (and analogously for y, z). Returns true on
  // success. Refuses (returns false) if any non-EMPTY voxel sits in the
  // to-be-removed boundary -- the caller must first arrange for the
  // outer rim to be empty (e.g. by deferring growth until tissue
  // density warrants it). Like grow_volume, the trim must be a multiple
  // of `region_size`.
  bool shrink_volume(int dx_each_side, int dy_each_side, int dz_each_side);

  // Count "structural neurons": connected components of NEURON-state
  // voxels in the 2-bit grid, where SYNAPSE and BLOCKED voxels (and
  // EMPTY) are walls. This is the user's preferred definition --
  // a single biological neuron is whatever blob of tissue is bounded
  // by synapses or scaffolding -- and may differ from `neuron_count()`
  // because two seeded `Neuron` entries can share a connected blob
  // (rarely, when sprouting bridges them without a synapse forming).
  int count_structural_neurons() const;

  // Distribution of structural-neuron sizes, in number of voxels per
  // connected component. Useful for diagnosing whether the brain is
  // populated by many small cells (early development) or fewer big
  // ones (matured arborisation).
  std::vector<int> structural_neuron_sizes() const;

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

  // Predictive-coding companion to `apply_input_pattern`. Sets the
  // `predicted_input` field of every INPUT neuron whose channel lies
  // in [0, n_features) to `predictions[channel] * input_drive_strength`.
  // The chemistry phase will subtract this from `input_acc` before
  // integration, so a perfectly-predicted stimulus produces zero
  // effective drive ("not surprising"). Useful for self-voice
  // prediction: when a motor fires, callers set predictions on the
  // matching efference channel; the actual efference delivery then
  // arrives at the predicted level and the residual drive is small.
  void apply_prediction_pattern(const float* predictions, int n_features);

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

  // Aversive ("punishment" / "danger") signal. Modelled on amygdala /
  // habenula-driven aversive learning, distinct from VTA dopamine reward.
  // Each excitatory synapse independently weakens its weight in
  // proportion to its eligibility trace (the pre/post pair that produced
  // the bad outcome shouldn't repeat); each inhibitory synapse instead
  // strengthens (gating against repetition). Net effect: the network
  // builds an "avoid this" representation around the pattern that just
  // caused harm.
  //
  // This is the simulator's analogue of "the baby touched the kettle and
  // learned not to" -- after one strong aversive event the network's
  // weights drift away from the trajectory that led to it. With
  // `aversive_amplification` set above 1, aversive learning is faster
  // than reward learning, matching the well-established negativity bias
  // (Baumeister 2001, Kahneman & Tversky 1979).
  void apply_aversive(float intensity);

  // Reset every synapse's eligibility trace to zero (between independent
  // trials in a training loop).
  void clear_eligibility();

  // Reset all transient dynamics so the next step starts from a fresh
  // chemistry baseline: per-neuron membrane potential, input
  // accumulator, fire-rate EMA, dendritic branch potentials and
  // incoming queue, plus every synapse's transit packets. Structural
  // weights, eligibility traces, consolidation tags and the structural
  // grid are *not* touched -- only the activity state. Useful between
  // independent probe scenes to avoid leak-over from the previous
  // scene's saturation.
  void reset_dynamics();

  // -------- Sleep: persist / restore the entire brain state --------------
  //
  // The "matrix = brain structure and state" snapshot captures everything
  // needed to resume the simulation later: the 2-bit structural grid,
  // every neuron's chemistry / queues / outgoing synapses, the regional
  // energy field, the owner map, the simulation step and the VZ band.
  // Returns true on success.
  bool save_state(const char* path) const;
  bool load_state(const char* path);

  // Dump the structural state to CSV files at `prefix_voxels.csv`,
  // `prefix_neurons.csv` and `prefix_synapses.csv`. Used by the
  // visualisation scripts under `scripts/` to render the 3D anatomy and
  // the connectome graph. Only structure is written -- no chemistry.
  bool dump_csv(const char* prefix) const;

  // -------- Position-binned features (cortical-map instrumentation) -------
  // Bin index of a neuron's soma in `cfg.region_size`-voxel units.
  // Returns {0, 0, 0} for an invalid id.
  std::array<int, 3> position_bin_for(uint32_t neuron_id) const;

  // Recompute the position-feature map from current neuron state. O(n).
  // Demos call this before reading or dumping features.
  void refresh_position_features();

  // Look up aggregate features at a bin (after refresh). Returns nullptr
  // if the bin has no neurons.
  const PositionFeatures* position_features_at(int bx, int by, int bz) const;

  // Number of populated bins (after refresh).
  std::size_t position_bin_count() const noexcept {
    return position_features_.size();
  }

  // Dump per-bin features to a CSV at `path`. Useful for visualisation
  // overlays. Caller is responsible for an up-to-date refresh.
  bool dump_position_features_csv(const char* path) const;

  // Sleep replay / consolidation. Drives the network with internal noise
  // only (no external input), with `boost` multiplied STDP amplitude, for
  // `n_steps` steps. Replays the patterns currently encoded in the
  // connectome, strengthening their most-fired co-activations -- the
  // behavioural correlate of slow-wave / REM sleep replay observed in
  // hippocampus and cortex.
  void sleep_consolidate(int n_steps, float boost = 1.6f);

  // Sleep with pattern rehearsal. In addition to internal-noise replay,
  // each step picks a random pattern from `patterns` and applies it to
  // the network's INPUT neurons (treating it like an external stimulus
  // would, but without reward). This models hippocampal replay of recent
  // experience: the network re-traverses its waking trajectories and
  // STDP / homeostatic / tag-and-capture machinery consolidates them.
  void sleep_replay_patterns(int n_steps,
                              const std::vector<std::vector<float>>& patterns,
                              int n_features,
                              float boost = 1.6f);

  // Slow-wave-sleep style sequenced replay. Each pattern in the supplied
  // list is presented in order for `present_per_pattern` steps, then a
  // brief silence (`gap_steps`) lets the connectome consolidate before
  // the next pattern. Mirrors hippocampal sharp-wave-ripples: the most
  // recently lived sequences are replayed in their original temporal
  // order, recruiting STDP windows that depend on causal ordering.
  // STDP gain is boosted (`boost`); attention modulator is attenuated
  // (sleep is a low-acetylcholine state).
  void sleep_sws_replay(const std::vector<std::vector<float>>& sequence,
                         int n_features,
                         int present_per_pattern = 12,
                         int gap_steps = 4,
                         float boost = 1.8f);

  // REM-style fragmented replay: random patterns from the pool, drawn
  // independently each step, with high acetylcholine and very strong
  // STDP boost so unusual co-firings get rapidly consolidated. Models
  // the dream-like recombination characteristic of REM sleep.
  void sleep_rem_replay(int n_steps,
                         const std::vector<std::vector<float>>& patterns,
                         int n_features,
                         float boost = 2.0f);

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
  void heterosynaptic_phase();
  void fire_dispatch_phase();
  void scheduler_dispatch_phase();
  void homeostatic_phase();
  void sprouting_phase();
  void synaptogenesis_phase();
  void pruning_phase();
  void energy_regen_phase();

  // Multiplier applied to STDP amplitudes from the sensitive-period
  // window: high near step 0, exponentially decays to 1.0.
  float developmental_factor() const noexcept;

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

  // Position-binned aggregate features (refreshed on demand).
  std::unordered_map<int64_t, PositionFeatures> position_features_;

  // Initialise a freshly-placed neuron's BCM `activity_baseline` from
  // the running mean of its bin neighbours. Soft cortical-map prior:
  // newborn neurons join with the local plasticity calibration, no
  // hard parameter sharing required.
  void apply_position_prior(Neuron& nu);

  // Per-region astrocyte calcium accumulator. Same coarse spatial
  // grid as the energy field; integrates synaptic-release events,
  // decays each step, and scales local STDP amplitude. Empty unless
  // the demo configures non-zero astrocyte fields.
  std::vector<float> astrocyte_ca_;

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
