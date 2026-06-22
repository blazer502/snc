# SNC PyTorch bridge — surrogate-gradient BPTT (Phase 5)

Train an **SNC-structured** spiking network with true backprop-through-time. The
division of labour follows `new-plan.md` §9: PyTorch owns data / optimizer /
loss / logging; SNC owns the **sparse topology** and the spiking forward model.

```
C++ snc_export  ──(graph.bin: CSR pre/post/role/channel)──▶  python/snc.load_graph
                                                                      │
                                                       snc.SNN (torch.nn.Module)
                                                 LIF + surrogate-gradient Heaviside
                                                   autograd unrolls T timesteps
                                                                      │
                                                   PyTorch Adam / cross-entropy
```

## Why this is BPTT, not e-prop

`snc.SNN.forward` runs the LIF recurrence in plain torch ops (sparse `index_add`
message passing over the SNC edge list). The spike is a custom
`autograd.Function` — Heaviside forward, fast-sigmoid surrogate backward — so
PyTorch autograd differentiates **through the whole unrolled timestep loop**.
That is exact surrogate-gradient BPTT, a stronger credit-assignment rule than the
local e-prop / direct-feedback-alignment used by the C++ trainers.

## Requirements

PyTorch (CUDA build recommended) + numpy. MNIST IDX files via
`./scripts/fetch_mnist.sh`. Build the bridge tool: `cmake --build build --target
snc_export`.

## Usage

```bash
# one command: generates the SNC topology, then trains it with BPTT
python3 python/train.py --structure static-snc --hidden 256 \
    --num-train 60000 --num-test 10000 --epochs 15 --device cuda

# deeper net (BPTT makes depth productive, unlike local learning):
python3 python/train.py --structure static-snc --hidden 256,256 --epochs 15

# baselines at the same synapse budget:
python3 python/train.py --structure random-sparse --hidden 256
python3 python/train.py --structure dense        --hidden 256
```

`--structure`, `--hidden W1,W2,..`, `--synapse-budget`, and `--seed` are passed
straight through to `snc_export`, so the PyTorch model trains the exact topology
the C++ generators produce.

## Files

| file | role |
|------|------|
| `snc/graph.py` | load the `snc_export` binary (CSR) |
| `snc/model.py` | `SNN` module + `SurrogateSpike` autograd function |
| `snc/data.py`  | MNIST IDX loader (numpy) |
| `train.py`     | CLI: export graph → load → train (Adam + BPTT) → log |

Limitation: the torch path assumes delay-1 feedforward graphs (the SNC
generators' default).
