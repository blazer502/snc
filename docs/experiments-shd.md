# SHD — a genuinely temporal benchmark (and where the structural prior fails)

MNIST barely exercises SNC's locality/timing prior. **SHD** (Spiking Heidelberg
Digits, Cramer et al. 2019) does: spoken digits 0–9 in English + German (20
classes) rendered by a cochlear model into spike trains over **700 frequency
channels and ~1 s**. This is the test the structural thesis pointed to — and the
result is an honest, instructive one: on SHD the brain-inspired locality prior
**does not help; random sparsity wins.**

## Setup

| | |
|---|---|
| Data | SHD — 8156 train / 2264 test, 20 balanced classes |
| Encoding | event passthrough, binned to T=100 × 700 channels |
| Model | recurrent LIF, SNC-sparse recurrent core, rate readout (mean spike count) |
| Learning | surrogate-gradient **BPTT**, Adam, 25 epochs |
| Hardware | GPU (PyTorch); `./scripts/fetch_shd.sh` to get the data |

```bash
./scripts/fetch_shd.sh
python3 python/train_shd.py --rec-structure random-sparse --hidden 256 --device cuda
```

The substrate handles the temporal task well: **~77% test accuracy** (chance 5%),
in the range of standard recurrent-SNN baselines on SHD (~71% RSNN, ~48%
feedforward in Cramer et al.; attention/adaptive models reach ~90%+). This is the
first non-MNIST, genuinely temporal task on the substrate.

## Does morphology-constrained structure help? No — random sparsity wins.

Two controlled comparisons, 2 seeds, best test accuracy.

**Recurrent core** (`H→H`, input projection dense):

| recurrent topology | edges | test acc |
|---|---:|---|
| dense | 65,536 | 0.751 ± 0.006 |
| static-snc (local) | 8,192 | 0.743 ± 0.008 |
| **random-sparse** | 8,192 | **0.775 ± 0.007** |

**Input projection** (700 cochlear channels → H; recurrent core fixed random-sparse):

| input topology | edges | test acc |
|---|---:|---|
| dense | 179,200 | 0.745 ± 0.003 |
| static-snc (frequency-local) | 16,384 | 0.755 ± 0.005 |
| **random-sparse** | 16,384 | **0.771 ± 0.015** |

**Reading it.** Random sparse connectivity is best in *both* places, and
morphology-constrained locality is worst (recurrent) or middling (input). Two
honest conclusions:

- **Sparsity still helps** — both sparse variants match or beat dense at far fewer
  edges (e.g. local input 0.755 > dense 0.745 at ~11× fewer edges). The substrate's
  efficiency bet survives.
- **But *locality* is the wrong prior here.** A recurrent reservoir wants rich,
  long-range mixing, which random connectivity gives and local connectivity
  starves; and a digit needs broad spectral/temporal integration, so each hidden
  unit benefits from sampling *diverse* frequencies (random) rather than a narrow
  local band. Locality restricts exactly the mixing the task needs.

## What this means for the structural thesis

This is the opposite of the MNIST result, where `static-snc` Pareto-dominated
random-sparse by +4 to +38 points ([`structural-advantage.md`](structural-advantage.md)).
Taken together they say something sharper and more credible than "brain-like
structure helps":

> **The morphology/locality prior helps only when the data's correlation
> structure is local along the connectivity axis** — adjacent pixels in an image
> — and it *hurts* when the task needs broad integration across the axis —
> frequency bands in audio. It is a genuine inductive bias, not a free lunch, and
> it must be *matched to the task*.

What SNC contributes here is therefore not "structure wins" but the **controlled
measurement of when it wins** — and a working spiking pipeline on a real temporal
neuromorphic benchmark.

## The *right* structural prior for time: conduction delays

If locality is the wrong prior for audio, the right one should be **temporal**.
SNC tracks a per-synapse conduction delay; giving the recurrent synapses a
*spread* of delays (`--delay-max D`, each synapse routed from `1..D` steps in the
past via a spike ring buffer) is a multi-timescale memory that uniform-delay
graphs cannot express. Same connectivity and seed throughout — only the delays
vary — random-sparse recurrent core, 2 seeds:

| recurrent delay | test acc |
|---:|---|
| 1 (uniform) | 0.744 ± 0.007 |
| spread 1–8  | 0.751 ± 0.007 |
| spread 1–16 | 0.757 ± 0.005 |
| **spread 1–30** | **0.778 ± 0.004** |

**Delays help, monotonically (+3.4 pts, non-overlapping bars)** — the opposite of
locality. This is the clean confirmation of the rule: the structural prior must
match the data's axis. On SHD that axis is *time*, so the temporal structural
feature (delays) helps where the spatial one (locality) hurt. The two SHD results
together — locality fails, delays win — are a sharper statement of the thesis than
either alone:

> Brain-inspired structure is a real inductive bias, not a free lunch. **Spatial
> locality** helps spatially-local data (MNIST) and hurts integration-heavy audio;
> **conduction delays** help temporal data (SHD) where locality fails. Use the
> structural feature whose geometry matches the task.

(Learnable delays reach SHD SOTA ~95%; here delays are fixed/structural, and the
point is the controlled +3.4-pt effect of a delay *spread*, not the absolute
number. Deriving the spread from morphology rather than a seeded draw is the
faithful next step.)
