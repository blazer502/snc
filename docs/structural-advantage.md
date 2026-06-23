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
| **dynamic** structure ≫ static in the scarce regime (+13–18 pts) | Exp 2, frontier |
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

The biggest single win was **dynamic** grow/prune/rewire, not a fixed topology:
on the frontier below it lifts static-snc by **+17.6 pts at a 2k-synapse budget**
(and +8.9 at 4k), fading to neutral once synapses are plentiful. This is SNC's
place in the dynamic-sparse-training family (DeepR/RigL/SET — see
[`related-work.md`](related-work.md)), with the twist that regrowth is **locality-
and morphology-constrained** rather than in abstract weight space. The lever to
push: drive regrowth by *data* (activity correlations), anneal the rewiring rate,
and run it on the slow structural clock during training.

### 3. Pair structure with *local* learning — don't fight BPTT on accuracy

Structure's edge collapses under BPTT (+0.7 pts) because a strong global optimizer
compensates for a weaker topology. So the productive niche is the opposite corner:
**structure + e-prop** (local, online, no backprop). There, the morphology prior
does real work, and the whole stack is on-device-friendly. The claim to make is
**accuracy-per-joule under local learning**, not leaderboard accuracy under BPTT.

### 4. Match the prior to the data — and use *delays*, not locality, for time

Locality is an advantage only when the data is local **along the connectivity
axis**, and it is *not* a free lunch. On MNIST (adjacent pixels correlated) it
Pareto-dominates random sparsity. On **SHD** (audio) it **loses** to random
sparsity in both the recurrent core and the input projection — digit
discrimination needs broad spectral/temporal mixing, which local connectivity
starves ([`experiments-shd.md`](experiments-shd.md)). So the rule is literally
*match the prior to the task*: brain-like locality helps spatially-local vision
and hurts integration-heavy audio. The lever that *should* fit temporal data —
and which random graphs lack entirely — is **conduction delays derived from
morphology** (the `delay>1` runtime), a temporal inductive bias we have not yet
exploited. That, not locality, is the structural bet to test on SHD next.

## Evidence: the accuracy-vs-synapse frontier

e-prop on full MNIST (256 hidden, GPU, 2 seeds), best test accuracy. Three
topology regimes at a shared budget: random sparse, frozen morphology-constrained
(`static-snc`), and **dynamically rewired** (`snc_cotrain`, budget-constant
grow/prune/rewire):

| budget | random-sparse | static-snc | **dynamic** | dyn − static |
|---:|---|---|---|---:|
| 2,000  | 0.235 | 0.330 | **0.506** | **+0.176** |
| 4,000  | 0.241 | 0.623 | **0.712** | **+0.089** |
| 8,000  | 0.537 | 0.865 | 0.866 | +0.001 |
| 16,000 | 0.749 | 0.914 | 0.920 | +0.006 |
| 40,000 | 0.898 | 0.937 | 0.944 | +0.007 |

(A frozen-vs-`cotrain` static control matched within noise — 0.331 vs 0.330 at
2k, 0.939 vs 0.937 at 40k — so the dynamic column's lift is the rewiring, not the
generator.)

**Reading it.** Two nested effects, both concentrated where synapses are scarce:

1. **static-snc Pareto-dominates random-sparse** at every budget — gap largest
   when tight (+0.38 at 4k), closing toward the dense limit (+0.04 at 40k).
   Equivalently, static-snc at 16k synapses (0.914) beats random-sparse at 40k
   (0.898): a ~2.5× synapse reduction at equal accuracy.
2. **Dynamic rewiring lifts the frontier further still**, and *only* at the tight
   end: **+17.6 pts at 2k, +8.9 at 4k**, fading to neutral by 8k. At budget 2,000
   dynamic (0.506) more than doubles random-sparse (0.235).

The pattern is the thesis in one figure: each step of "more structure" — local
prior, then learned-during-training topology — buys the most exactly where the
budget is tightest, and washes out as synapses (or optimizer strength, cf. the
BPTT result) become plentiful. **Structure is worth most precisely where you have
least.** This is also why dynamic plasticity, not a fixed prior, is the engine to
push (§2): topology *search* pays off most in the scarce regime where a single
fixed local prior leaves accuracy on the table.

## The sharpened thesis

SNC's structural advantage is **not** "better accuracy." It is **better accuracy
per synapse and per joule in the budget-constrained, locally-learned, temporally-
structured regime** — and the feasibility of widths that dense connectivity cannot
fit. Used that way (frontier metric + dynamic plasticity + local learning +
task-matched locality/delays), the brain-inspired structure is a genuine lever;
used as a drop-in replacement for a dense BPTT-trained layer, it is roughly
neutral. The experiments say: aim it where it bites.
