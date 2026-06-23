# Related Work — where SNC sits

SNC is not a single new algorithm; it is a **substrate** that combines several
existing threads. This note places it against the closest prior work and states,
honestly, what is shared and what is distinctive. (None of our results claim to
beat these methods on absolute accuracy — see [`experiments-mnist.md`](experiments-mnist.md).)

## Three threads SNC draws on

### 1. Dynamic sparse training (train a sparse net by rewiring)

The lineage SNC's two-timescale grow/prune/rewire (`snc_cotrain`) belongs to:

- **Deep Rewiring (DeepR)** — Bellec, Kappel, Maass, Legenstein, ICLR 2018.
  Trains a strictly sparse net by stochastically rewiring during SGD; reaches
  good MNIST accuracy at ~2% connectivity. The closest ancestor of our rewiring.
- **SET** (Mocanu et al., Nature Comm 2018) — magnitude-prune + **random** regrow.
- **RigL** (Evci et al., ICML 2020) — magnitude-prune + **gradient-based** regrow.
- **Lottery Ticket Hypothesis** (Frankle & Carbin, ICLR 2019) — sparse subnetworks
  that train well in isolation.

These rewire in **abstract weight space**. SNC's distinction: rewiring is
**morphology- and locality-constrained in a 3-D embedding** (a pre connects only
to spatially-near posts), with conduction **delays derived from geometry**, run
on a **slow structural clock** separate from the fast spike loop.

### 2. Sparse / structurally-plastic spiking networks

The SNN-specific neighbors, several very recent:

- **Adaptive sparse structure development** for SNNs (Han et al.) — dendritic-spine
  plasticity, neuronal pruning + synaptic regeneration.
- **Gradient Rewiring for SNN pruning** (Chen et al., 2021) — DeepR-style rewiring
  for spiking nets.
- **Spiking-activity-based pruning** (2024) and **compression-efficiency sparse
  structure learning** (2025) — prune/regrow deep SNNs during training.
- **GPU frameworks for SNN structural plasticity** — report large training
  speedups from sparsity, the same systems bet SNC makes.

SNC overlaps strongly here. What it adds is the **integrated runtime** (CPU /
OpenMP / three CUDA event-delivery backends) and the matched-budget study showing
when structure-aware sparsity beats random sparsity.

### 3. Brain-inspired structured connectivity

- **Small-world SNNs via neuroevolution** (iScience 2024) and **brain-inspired
  evolutionary architectures for SNNs** — evolve locally-dense / globally-sparse
  topologies; report accuracy gains from brain-like structure.
- **Sparse connectivity in cortex-like ANNs** (PMC, 2025) — locally-dense,
  globally-sparse connectivity aids time/data-efficient processing.

SNC shares the thesis (brain-like local sparsity helps); it differs by
*generating* the topology from an explicit morphology/voxel substrate rather than
evolving it, and by running it on an event-driven spike engine.

### 4. Learning rules

- **e-prop** — Bellec et al., *Nature Communications* 2020 ("A solution to the
  learning dilemma for recurrent networks of spiking neurons"). SNC's local
  trainer **is** e-prop (three-factor eligibility); hidden credit uses
  random-feedback / **direct feedback alignment** (Nøkland 2016).
- **Surrogate-gradient BPTT** — Neftci, Mostafa & Zenke 2019; the standard SNN
  training rule, which the SNC PyTorch bridge uses (Phase 5).

## Positioning table

| | DeepR / RigL | sparse/plastic SNNs | small-world SNNs | e-prop | **SNC** |
|---|---|---|---|---|---|
| spiking | no | yes | yes | yes | **yes** |
| sparse-during-training | yes | yes | (evolved) | no | **yes (cotrain)** |
| structure mechanism | weight-space rewire | prune/regrow | neuroevolution | fixed | **3-D morphology + locality** |
| geometric delays | no | no | no | no | **yes** |
| local learning | no (SGD) | usually BPTT | BPTT | yes | **yes (e-prop) + BPTT** |
| event-driven CUDA runtime | n/a | some | no | n/a | **yes (atomic/bucket/sort)** |
| seq / language model | — | rare | no | yes (RSNN) | **yes (spiking char-LM)** |

## The honest delta

No single mechanism in SNC is new: rewiring is DeepR/RigL-flavored, the local rule
is e-prop, the gradient path is surrogate BPTT, the locality bias appears in
small-world-SNN work. SNC's contribution is **the combination as one substrate** —
a morphology-generated, locality-and-delay-constrained sparse graph that can be
(a) executed by event-driven CPU/CUDA backends, (b) trained by *either* local
e-prop *or* BPTT, and (c) evolved by two-timescale plasticity — plus a set of
**controlled, matched-budget measurements** of *when* structure helps:

- structure-aware > random sparsity at equal budget, widening as the budget
  tightens and as the learner weakens (e-prop +4 pts → BPTT +0.7 pts);
- dynamic > static structure in the scarce-synapse regime (+13 pts);
- sparsity is what makes width-scaling feasible at all (dense recurrent OOMs).

What SNC does **not** claim: state-of-the-art accuracy. On MNIST it is a
single-hidden-layer rate-coded SNN; the value is the substrate and the
when-does-structure-help analysis, not a leaderboard number.

## Sources

- DeepR: <https://arxiv.org/abs/1711.05136> · RigL: <https://arxiv.org/abs/1911.11134>
- e-prop (Nature Comm 2020): <https://www.nature.com/articles/s41467-020-17236-y>
- Adaptive sparse structure development (SNN): <https://arxiv.org/abs/2211.12219>
- Gradient rewiring for SNN pruning: <https://arxiv.org/abs/2105.04916>
- Spiking-activity-based pruning: <https://arxiv.org/abs/2406.01072>
- Sparse structure learning / compression efficiency (SNN): <https://arxiv.org/abs/2502.13572>
- Small-world SNN via neuroevolution (iScience 2024): <https://www.cell.com/iscience/fulltext/S2589-0042(24)00066-X>
- Sparse connectivity in cortex-like ANNs: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11966417/>
- Surrogate-gradient learning in SNNs: <https://arxiv.org/abs/1901.09948>
- Heidelberg spiking datasets (SHD/SSC): <https://arxiv.org/abs/1910.07407>
