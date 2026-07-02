# Developmental Multicenter SNC — embodied navigation ("fetch the named object")

The phase after the [v1 nursery prototype](nursery-v1.md). v1 built a two-center
namer; this phase makes the agent *act*: it adds a **spatial** and a **motor**
center (four centers now, plan §5.2) and has the agent **fetch a named object** —
using the v1 language→vision recall to decide *which* object the word means, then
walking there. It reaches the plan's developmental milestones for sensorimotor
grounding and cross-modal grounding (§7.3 stages 1 & 4), and it makes the learning
rule genuinely **three-factor**: the motor policy is trained by reward, which
varies (unlike v1's naming, where reward is held at 1.0).

## What it adds

```
  language ──recall──▶ visual        "which object is 'red'?"   (v1 pathway)
  spatial  ──policy──▶ motor         "which way, and which move?"  (new)
```

- **Spatial center** (`navigation.SpatialCenter`) — a fixed spike code for the
  target's direction relative to the agent (one neuron per relative direction: a
  3×3 sign(dx), sign(dy) grid, 9 cells).
- **Motor center** (`navigation.MotorCenter`) — a policy over the four move
  actions, trained by a **reward-modulated three-factor rule**: the local
  eligibility `(action_taken − policy) × spatial_activity`, gated by
  `(reward − running_baseline)`. That is REINFORCE-with-baseline written as a
  synaptic three-factor update — the plan's motor center (reinforcement learning,
  local eligibility traces, motor babbling → skill, §5.2). Reward is a shaped
  distance signal, so the third factor genuinely varies.
- **NavAgent** (`navigation.NavAgent`) — composes the v1 language/recall pathways
  with the spatial and motor centers into a four-center `MulticenterGraph`.

The fetch loop: the teacher names a target by colour; **recall** (`language →
vision`) predicts the target's visual code and picks the matching object in view;
the **spatial → motor** loop walks to it. Identifying *which* object to fetch is
pure cross-modal grounding; walking there is the learned motor skill.

## How to run

```bash
python3 python/train_nursery.py --selftest        # includes navigation assertions
python3 python/train_nursery.py --nav-seeds 8     # prints the table below
```

## Results

Fetch a named object among 4 distinct-coloured objects on a 7×7 grid, with the
cross-modal pathway on vs. lesioned. Means over 8 seeds.

| condition | motor success | fetch success | avg steps |
|---|--:|--:|--:|
| **cross-modal on** (recall identifies the target) | 1.00 | **0.91** | 6.8 |
| cross-modal off (pathway lesioned) | 1.00 | 0.27 | 6.6 |

Chance baselines (both computed/analytic): a uniform-random motor policy reaches
the target cell 0.26 of the time under the same protocol; fetch by guessing the
target among 4 objects is 1/4 = 0.25 (the lesioned row's 0.27).

### What the numbers say

- **The motor skill is learned, by reward.** `motor_success = 1.00` vs. **0.26**
  for a uniform-random policy under the identical protocol (reported as
  `motor_random`, computed in `run_navigation`): the reward-modulated three-factor
  rule learns to navigate to an arbitrary target cell. This is the first place in
  the project where reward is a real varying factor rather than a constant — the
  three-factor framing is now exercised, not just scaffolded.
- **Cross-modal grounding drives embodied behavior (RQ5 → action).** The motor
  skill is identical in both rows (1.00), so the fetch gap is entirely the
  cross-modal pathway: with recall the agent fetches the correct named object 0.91
  of the time; lesion it and fetching falls to 0.27 ≈ the 1/4 chance of guessing
  the target among four objects. Naming knowledge, learned only from co-observation
  in v1, now *steers action*.
- **Four centers, coordinated.** Perception, language, space, and action are
  separate centers linked by pathways; the behavior emerges from their coordination
  (plan RQ1/RQ2), not from one shared model.

## Milestones reached (plan §7.3)

| stage | milestone | evidence |
|---|---|---|
| 1 — sensorimotor grounding | move toward a target; associate moves with spatial change | motor_success 1.00 |
| 4 — cross-modal grounding | use language to find a visual object | fetch_success 0.91 vs 0.27 lesion |

## Limitations & next steps

- The motor policy is trained on an obstacle-free grid; fetching tolerates sparse
  obstacles by skipping blocked moves, but there is no learned obstacle avoidance.
- Targets are identified by colour (a single-attribute query); multi-attribute or
  ambiguous targets are future work.
- Next: object permanence / spatial memory (stage 2) — search for an occluded named
  object using the memory center; learned obstacle avoidance; and scaling the motor
  learning to a richer action space.

**Update:** object permanence is now built — see
[nursery-permanence.md](nursery-permanence.md) (a fifth center, episodic memory,
lets the agent search for a named object it can no longer see).

Code: `python/dev_snc/navigation.py`, `tasks.run_navigation`,
`experiment.run_navigation_suite`; CLI `python/train_nursery.py --nav-seeds N`.
