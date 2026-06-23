# Scaling SNC toward an LLM-like model

An LLM is, mechanically, a **sequence model trained on next-token prediction**.
The brain-inspired SNC substrate is natively temporal and recurrent, so the
faithful way to make it "LLM-like" is not to bolt on a transformer but to train
a **recurrent spiking network** on next-token prediction by BPTT. This note
describes the approach, the first proof of concept, and how it scales.

## Architecture (`python/snc/lm.py`)

A **stack** of recurrent spiking layers; every recurrent and inter-layer weight
is SNC-sparse, only the embedding and readout are dense.

```
token x_t ──Embed──▶ input current to layer 0
                       │
   layer 0: LIF pool, SNC-sparse recurrent core (H0→H0)
                       │ SNC-sparse feedforward (H0→H1)
   layer 1: LIF pool, SNC-sparse recurrent core (H1→H1)
                       │ ...
   top-layer spikes ─ low-pass trace ─▶ linear readout ─▶ next-token logits
```

Per token step, layers update bottom-up; recurrence reads the layer's own
previous-step spikes, feedforward reads the current step of the layer below:

```
rec_l = W_rec[l] · s_l(t-1)
ff_l  = Embed(x_t)            if l == 0   else  W_ff[l] · s_{l-1}(t)
v_l   = decay·v_l(t-1) + ff_l + rec_l ;  s_l = H(v_l - thr) ;  reset
logit = Readout( ro_decay·r(t-1) + s_top )
```

PyTorch autograd unrolls the token loop → true surrogate-gradient BPTT. **SNC
owns all the recurrent + feedforward structure** (graphs from `snc_export` over
`[H,H]` and `[H_{l-1},H_l]`); embedding, readout, optimizer, loss and
tokenization are standard PyTorch. Sequence memory lives in the leaky membrane
plus the recurrent spikes; there is no attention.

## Proof of concept (char-level tiny-shakespeare)

```bash
python3 python/train_lm.py --layers 512        --structure static-snc --seq 128
python3 python/train_lm.py --layers 1024       --structure static-snc   # wider
python3 python/train_lm.py --layers 512,512    --structure static-snc   # deeper
```

Metric is **bits-per-char** (cross-entropy / ln 2); the unigram baseline for
this corpus is ≈ 4.77 bits/char. The spiking substrate learns language structure
well below that. Sampled text (temperature 0.8) is clearly English/Shakespeare-
shaped — speaker tags, words, punctuation, line breaks:

```
ROMEO:
I him kisth, lokis unigh is striteng my, hath doo dees
The comuere for king soe? of thes his had-keing ...
ISs Seir live itme nown ed would she ...
```

## Scaling study (GPU, full BPTT)

Each run trains on a GPU (PyTorch BPTT, ~17 GB resident for the dense graph).
val bits/char on held-out tiny-shakespeare:

| config (recurrent core) | params | seq | val BPC |
|---|---:|---:|---:|
| static-snc  512          | 83k  | 128 | 3.16 |
| static-snc  1024         | 166k | 128 | 3.12 |
| static-snc  512,512 (2L) | 132k | 128 | 3.49 |
| static-snc  768          | 124k | 256 | **3.10** |
| dense       512          | 329k | 128 | **3.00** |
| dense       1024         | —    | 128 | **OOM** |

What scaling does and does not do here:

- **Sparsity is what enables scale.** A dense `H=1024` recurrent core is a
  ~1M-edge matrix whose BPTT activation graph **OOMs a 47 GB GPU**; the
  SNC-sparse `H=1024` core is ~33k edges and trains comfortably to 3.12. On a
  fixed GPU, sparse can go wider than dense can.
- **Width helps a little; longer context + training helps most** (3.16 → 3.10).
- **Depth hurts at equal compute.** A 2-layer stack (3.49) trains much slower
  than one wide layer — deep spiking RNNs are hard to optimise, consistent with
  the MNIST depth result; it would need far more iterations to pay off.
- **Dense is best per-config but doesn't scale.** dense-512 (3.00, 329k params)
  edges out the sparse cores, but at ~4× the parameters and with no path to
  larger widths. sparse-1024 (3.12, 166k) gets close at half the params.

Honest read: this basic spiking-RNN char-LM **plateaus around ~3.0–3.2 BPC** —
it is optimisation/architecture-limited, not capacity-limited, so raw size gives
diminishing returns. The clear win is the systems one: the SNC sparse substrate
is what makes the GPU-resident BPTT affordable enough to scale at all.

This is a *tiny* model and an honest proof of concept, not a real LLM — but it is
the same task and training signal an LLM uses, running end-to-end on spikes.

## How this scales toward LLM-like usage

The substrate is built for the levers that matter:

- **Width / depth.** Larger hidden pools and stacked recurrent spiking layers
  (`--layers W1,W2,..`) already work. Width gives modest gains here; depth needs
  a stronger optimiser (see the scaling study) — making deep spiking RNNs train
  well is the open problem, not the plumbing.
- **Sparsity is the scaling bet.** The recurrent core is SNC-structured sparse,
  so parameter/compute cost grows with the synapse budget, not `H²`. Combined
  with **event-driven spikes** (most neurons silent most steps) and the **CUDA
  bucket/sort delivery backends**, the substrate targets the regime where dense
  transformers are expensive.
- **Structural plasticity.** Two-timescale grow/prune/rewire (`snc_cotrain`)
  can evolve the recurrent topology during training — a learned, sparse,
  brain-like connectome instead of a fixed dense matrix.
- **Tokenization.** Char-level today; a BPE/subword embedding table is a drop-in
  change to the input/readout layers (the spiking core is token-agnostic).
- **Longer context.** Currently truncated BPTT over a window; longer memory
  needs slower membrane/adaptation time constants, multi-timescale neurons, or
  delayed synapses (the `delay>1` runtime exists in C++; the torch path is
  delay-1 today).

## Honest limits / next

Char-level, single GPU, truncated-BPTT window, spiking-RNN (no attention, harder
to optimize than transformers), and far below real-LLM scale. The contribution
is a *working* spiking sequence model on the SNC substrate and a concrete path
to scale it. Natural next steps: structure-vs-dense recurrent comparison at
scale, subword tokenization, and delayed-synapse temporal credit assignment.
