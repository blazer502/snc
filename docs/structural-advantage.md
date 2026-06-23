# Using SNC's structural advantage well

Across the studies, the brain-inspired structural prior helps — but *conditionally*.
This note distills **when** structure pays off and **how to use it better**, then
backs the framing with an accuracy-vs-budget frontier.

## What the experiments actually showed

| observation | source |
|---|---|
| structure-aware > random sparsity at equal budget (+4 pts) | Exp 1 |
| the gap **widens as the budget tightens** | Exp 2, frontier (below) |
| the gap **shrinks as the learner strengthens** (+4 e-prop → +0.7 BPTT) | Exp 4 |
| **dynamic** structure ≫ static in the scarce regime (+13 pts) | Exp 2 |
| sparsity is what makes wide models *fit at all* (dense recurrent OOMs) | LM scaling |

The throughline: **structure substitutes for budget and for optimizer strength.**
It matters most exactly where deep dense nets are weakest — tiny synapse budgets,
cheap/local learning, memory-bound scale.

## Four ways to use it better

### 1. Optimize the *frontier*, not a single point

A single accuracy number at one budget undersells (or oversells) structure. The
honest figure of merit is **accuracy per synapse** (and per energy). Structure
should *dominate the Pareto frontier* — equal-or-better accuracy at every budget,
with the largest margin where synapses are scarce. We measure this directly below.

### 2. Make structural plasticity the engine, not a one-shot prior

The biggest single win was **dynamic** grow/prune/rewire (+13 pts at ~2k
synapses), not a fixed topology. This is SNC's place in the dynamic-sparse-training
family (DeepR/RigL/SET — see [`related-work.md`](related-work.md)), with the twist
that regrowth is **locality- and morphology-constrained** rather than in abstract
weight space. The lever to push: drive regrowth by *data* (activity correlations),
anneal the rewiring rate, and run it on the slow structural clock during training.

### 3. Pair structure with *local* learning — don't fight BPTT on accuracy

Structure's edge collapses under BPTT (+0.7 pts) because a strong global optimizer
compensates for a weaker topology. So the productive niche is the opposite corner:
**structure + e-prop** (local, online, no backprop). There, the morphology prior
does real work, and the whole stack is on-device-friendly. The claim to make is
**accuracy-per-joule under local learning**, not leaderboard accuracy under BPTT.

### 4. Match the prior to the data — and use *delays*

Locality is only an advantage when the data is local. MNIST barely exercises it;
**event-based vision (N-MNIST, DVS) and cochlear audio (SHD)** are where local
spatiotemporal structure is the right inductive bias. SNC also has a lever random
graphs lack entirely: **conduction delays derived from morphology**, a temporal
inductive bias for sequence/temporal tasks (the `delay>1` runtime). This is the
most under-exploited structural advantage.

## Evidence: the accuracy-vs-synapse frontier

Frozen-structure e-prop on full MNIST (256 hidden, GPU, 2 seeds). static-snc vs
random-sparse at a shared budget; best test accuracy, mean ± std:

| synapse budget | static-snc | random-sparse | gap |
|---:|---|---|---:|
| 2,000  | 0.330 ± 0.004 | 0.235 ± 0.015 | +0.095 |
| 4,000  | **0.623 ± 0.019** | 0.241 ± 0.008 | **+0.382** |
| 8,000  | 0.865 ± 0.001 | 0.537 ± 0.062 | +0.328 |
| 16,000 | 0.914 ± 0.001 | 0.749 ± 0.017 | +0.164 |
| 40,000 | 0.937 ± 0.000 | 0.898 ± 0.000 | +0.038 |

**Reading it:** static-snc Pareto-dominates random-sparse at *every* budget, and
the gap is largest where synapses are scarce (**+38 pts at 4k**) — exactly the
regime where the structural prior substitutes for missing synapses. Equivalently,
**static-snc at 16k synapses (0.914) beats random-sparse at 40k (0.898)**: the
morphology prior buys a ~2.5× synapse reduction at equal accuracy, and more at
tighter budgets. As budget → the dense limit both saturate and the gap closes
(+0.04 at 40k) — mirroring the BPTT result, where a strong optimizer also closes
the gap. Structure is worth most precisely where you have least.

## The sharpened thesis

SNC's structural advantage is **not** "better accuracy." It is **better accuracy
per synapse and per joule in the budget-constrained, locally-learned, temporally-
structured regime** — and the feasibility of widths that dense connectivity cannot
fit. Used that way (frontier metric + dynamic plasticity + local learning +
task-matched locality/delays), the brain-inspired structure is a genuine lever;
used as a drop-in replacement for a dense BPTT-trained layer, it is roughly
neutral. The experiments say: aim it where it bites.
