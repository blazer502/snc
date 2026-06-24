# SNC Documentation

Map of the docs. SNC is a structure-aware spiking neural computing substrate
(see the [project README](../README.md)); these documents cover its design,
results, and origins.

## Design & plan

| doc | what it covers |
|---|---|
| [architecture.md](architecture.md) | The substrate: three-layer separation (structural → compiled CSR graph → runtime), execution model, all CPU/CUDA backends, and the training paths (e-prop CPU+GPU, two-timescale co-training, PyTorch BPTT). The implementation reference. |
| [new-plan.md](new-plan.md) | The repositioning plan (brain-sim → SNN substrate) and phased roadmap. |
| [../python/README.md](../python/README.md) | The PyTorch bridge: `snc_export` → load graph → surrogate-gradient BPTT (SNN / LM / SHD models). |
| [voxel-encoding.md](voxel-encoding.md) | Design study: 2-bit vs 1-bit structural voxels (why 2-bit stays). |

## Experiments & results

| doc | what it covers |
|---|---|
| [experiments-mnist.md](experiments-mnist.md) | Full-MNIST multi-seed study: structure-aware vs random sparsity (Exp 1), dynamic co-training (Exp 2), depth (Exp 3), surrogate-gradient BPTT (Exp 4). Error-barred. |
| [experiments-shd.md](experiments-shd.md) | SHD/SSC spoken-audio: recurrent spiking classifier (locality fails, delays help); 2-layer adaptive + augmentation reaches SHD 0.855 / SSC 0.640; spike-reg accuracy/energy frontier. |
| [structural-advantage.md](structural-advantage.md) | Synthesis: *when* structure helps, and the accuracy-vs-synapse-budget frontier (static / dynamic / random), incl. the dynamic-co-training frontier. |
| [llm-direction.md](llm-direction.md) | Recurrent spiking language model (next-token BPTT) and the width/depth/context scaling study. |

## Positioning

| doc | what it covers |
|---|---|
| [related-work.md](related-work.md) | SNC vs DeepR/RigL/SET, e-prop, sparse/plastic SNNs, small-world SNNs, surrogate-gradient BPTT — with the honest delta. |

## Background — the brain-development model (origins)

| doc | what it covers |
|---|---|
| [brain-model.md](brain-model.md) | The original brain-development engine and biological model — voxel grid, morphology, synaptogenesis, three-factor learning, engrams, sleep — with primary-source grounding and the pack-by-pack history. The substrate's slow structural layer. |
| [ROADMAP.md](ROADMAP.md) | Long-form brain-model roadmap (legacy). |
| [MORPHOLOGY_REFACTOR.md](MORPHOLOGY_REFACTOR.md) · [DISCUSSION_CONSCIOUSNESS.md](DISCUSSION_CONSCIOUSNESS.md) | Morphology refactor and consciousness/predictive-coding notes (legacy). |

## Reproducing the studies

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DSNC_ENABLE_CUDA=ON
cmake --build build-cuda -j
./scripts/fetch_mnist.sh ; ./scripts/fetch_shd.sh ; ./scripts/fetch_ssc.sh

DEVICE=cuda ./scripts/run_mnist_study.sh   # Exp 1-2 (e-prop + cotrain)
./scripts/run_bptt_sweep.sh                # Exp 4 (BPTT, multi-seed)
./scripts/run_depth_sweep.sh               # Exp 3 (depth)
python3 scripts/aggregate_study.py         # + aggregate_bptt.py / aggregate_depth.py
```
