// Unit test for STDP weight dynamics.
//
// Constructs a minimal two-neuron network: a pre-synaptic neuron (id=1) with
// a single synapse onto a post-synaptic neuron (id=2). The synapse is wired
// directly via install_synapse so the test does not depend on the
// stochastic synaptogenesis pipeline. We then drive deliberately-timed
// firings and verify the weight moves in the biologically-expected
// direction:
//
//   pre delivers, then post fires shortly after  -> LTP (weight up)
//   post fires, then pre delivers shortly after  -> LTD (weight down)
//   post fires far away from any delivery        -> no change
//
// Each scenario is run with a fresh simulator so the synapse's state does
// not carry over between tests. The test exits non-zero on any failure.

#include "simulator.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr float kInitialWeight = 0.5f;

// Build a simulator containing exactly two neurons + one synapse pre->post,
// at known positions (close enough that grid placement always succeeds in a
// 32^3 volume) and with all developmental parameters disabled. This isolates
// STDP from sprouting / synaptogenesis / pruning effects so the test only
// exercises the rule we want to verify.
snc::Simulator make_two_neuron_sim(int conduction_delay) {
  snc::SimConfig cfg;
  cfg.X = 32;
  cfg.Y = 32;
  cfg.Z = 32;
  cfg.region_size = 8;
  cfg.fire_threshold = 0.8f;
  cfg.potential_decay = 0.0f;        // potential resets every step
  cfg.weight_decay = 1.0f;           // no slow decay
  cfg.weight_potentiation = 0.0f;    // no other Hebbian rule
  cfg.homeostatic_rate = 0.0f;       // disable scaling
  cfg.spine_retraction_floor = 0.0f; // disable spine retraction
  cfg.prune_inactive_steps = 1000000;
  cfg.synapse_form_prob = 0.0f;      // no new synapses
  cfg.sprout_prob = 0.0f;            // no sprouting
  cfg.eligibility_decay = 1.0f;
  cfg.eligibility_potentiation = 0.0f;
  cfg.input_drive_strength = 1.0f;
  cfg.weight_max = 5.0f;             // big enough to see LTP

  snc::Simulator sim(cfg);
  const uint32_t pre = sim.add_neuron_at(10, 10, 10);
  const uint32_t post = sim.add_neuron_at(20, 20, 20);
  if (pre == 0 || post == 0) {
    std::fprintf(stderr, "failed to place neurons\n");
    std::exit(2);
  }
  sim.install_synapse(pre, post, kInitialWeight, conduction_delay);
  return sim;
}

float synapse_weight(const snc::Simulator& sim) {
  return sim.neurons()[0].outgoing[0].weight;
}

bool test_ltp() {
  // Conduction delay of 1: a spike fired at step T arrives at step T+1.
  // Drive pre at step T0 by injecting a strong external input. With
  // potential_decay=0 it fires that step. The packet is then in transit;
  // at step T0+1 the scheduler delivers it to post.incoming_queue. We
  // drive post strongly at step T0+2 so post fires THAT step. The
  // delivery happened at step T0+1 and post fired at step T0+2, dt=1 ->
  // LTP should fire.
  auto sim = make_two_neuron_sim(/*conduction_delay=*/1);

  // Step 0: drive pre to fire. (We need to reach pre AFTER chemistry
  // already integrates, i.e. apply the input before step()).
  sim.inject_input(1, 5.0f);
  sim.step();  // pre fires at step 0; packet enters transit

  // Step 1: scheduler delivers packet to post.incoming_queue at end.
  // No external drive on post yet.
  sim.step();

  // Step 2: drive post strongly so it fires this step. Now LTP detects:
  // syn.last_delivery_step = 1, post.fired_this_step at step 2, dt = 1.
  sim.inject_input(2, 5.0f);
  sim.step();

  const float w = synapse_weight(sim);
  const bool ok = w > kInitialWeight + 1e-4f;
  std::printf("[LTP] initial=%.4f  final=%.4f  -> %s\n",
              kInitialWeight, w, ok ? "PASS" : "FAIL");
  return ok;
}

bool test_ltd() {
  // Anti-causal: post fires first, then pre's spike arrives later.
  auto sim = make_two_neuron_sim(/*conduction_delay=*/3);

  // Step 0: drive post to fire. post.last_fire_step = 0.
  sim.inject_input(2, 5.0f);
  sim.step();

  // Step 1: drive pre to fire. Packet enters transit with delay 3, so it
  // will arrive at step 4 (after three propagation steps).
  sim.inject_input(1, 5.0f);
  sim.step();

  // Steps 2, 3: nothing.
  sim.step();
  sim.step();

  // Step 4: scheduler delivers. dt = step - post.last_fire_step = 4 - 0 = 4
  // (within stdp_window). LTD fires.
  sim.step();

  const float w = synapse_weight(sim);
  const bool ok = w < kInitialWeight - 1e-4f;
  std::printf("[LTD] initial=%.4f  final=%.4f  -> %s\n",
              kInitialWeight, w, ok ? "PASS" : "FAIL");
  return ok;
}

bool test_outside_window() {
  // Conduction delay of 1, but post fires far in the future, well past
  // stdp_window from the delivery -> no STDP change expected.
  auto sim = make_two_neuron_sim(/*conduction_delay=*/1);

  sim.inject_input(1, 5.0f);
  sim.step();          // pre fires, packet in transit
  sim.step();          // delivery at step 1
  // Idle far past the STDP window (default 20 steps).
  for (int i = 0; i < 60; ++i) sim.step();
  // Now drive post -- delta is 60+ steps from the delivery, way out of window.
  sim.inject_input(2, 5.0f);
  sim.step();

  const float w = synapse_weight(sim);
  const bool ok = std::fabs(w - kInitialWeight) < 1e-4f;
  std::printf("[outside-window] initial=%.4f  final=%.4f  -> %s\n",
              kInitialWeight, w, ok ? "PASS" : "FAIL");
  return ok;
}

bool test_ltp_decays_with_dt() {
  // Two runs: dt=1 and dt=5. The exponential STDP kernel should produce
  // a smaller LTP at dt=5.
  auto sim_short = make_two_neuron_sim(1);
  sim_short.inject_input(1, 5.0f);
  sim_short.step();
  sim_short.step();
  sim_short.inject_input(2, 5.0f);
  sim_short.step();
  const float w_short = synapse_weight(sim_short) - kInitialWeight;

  auto sim_long = make_two_neuron_sim(1);
  sim_long.inject_input(1, 5.0f);
  sim_long.step();
  sim_long.step();
  for (int i = 0; i < 4; ++i) sim_long.step();  // wait extra 4 steps
  sim_long.inject_input(2, 5.0f);
  sim_long.step();
  const float w_long = synapse_weight(sim_long) - kInitialWeight;

  const bool ok = w_short > w_long && w_long > 0.0f;
  std::printf("[temporal-decay] dt=1 dw=%.4f, dt=5 dw=%.4f  -> %s\n",
              w_short, w_long, ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main() {
  int fails = 0;
  fails += !test_ltp();
  fails += !test_ltd();
  fails += !test_outside_window();
  fails += !test_ltp_decays_with_dt();
  if (fails == 0) {
    std::printf("\nall STDP tests passed\n");
    return 0;
  }
  std::printf("\n%d STDP test(s) failed\n", fails);
  return 1;
}
