#!/usr/bin/env python3
"""Train the (stacked) recurrent spiking language model by BPTT.

Char-level next-token prediction -- the LLM-shaped task -- on the SNC substrate:
every recurrent and inter-layer weight uses an SNC-generated sparse topology;
PyTorch owns data/optimizer/loss. Reports bits-per-char vs the unigram baseline.

    python3 python/train_lm.py --layers 512,512 --structure static-snc \\
        --iters 3000 --seq 128 --device cuda
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F

import snc
from snc.lm import SpikingLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=os.path.join(ROOT, "data/text/tinyshakespeare.txt"))
    p.add_argument("--layers", default="512", help="recurrent layer widths, e.g. 512 or 512,512")
    p.add_argument("--structure", default="static-snc",
                   choices=["dense", "random-sparse", "static-snc"])
    p.add_argument("--rec-fanin", type=int, default=32, help="avg recurrent fan-in")
    p.add_argument("--ff-fanin", type=int, default=32, help="avg inter-layer fan-in")
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--decay", type=float, default=0.95)
    p.add_argument("--ro-decay", type=float, default=0.9)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--surrogate-scale", type=float, default=10.0)
    p.add_argument("--w-scale", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--export-bin", default=os.path.join(ROOT, "build/snc_export"))
    return p.parse_args()


def export_graph(a, A, B, budget, sub_seed):
    """Export an SNC [A,B] graph; return (pre in [0,A), post in [0,B)) + edge count."""
    tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False); tmp.close()
    cmd = [a.export_bin, "--structure", a.structure, "--layers", f"{A},{B}",
           "--synapse-budget", str(budget), "--seed", str(a.seed * 1000 + sub_seed),
           "--out", tmp.name]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"snc_export failed:\n{r.stderr}")
    g = snc.load_graph(tmp.name)
    pre = torch.from_numpy(g["pre"].astype(np.int64))
    post = torch.from_numpy(g["post"].astype(np.int64)) - A  # layer1 -> [0,B)
    return pre, post, g["S"]


def batch(data, B, T, device, rng):
    ix = rng.integers(0, len(data) - T - 1, size=B)
    x = np.stack([data[i:i + T] for i in ix])
    y = np.stack([data[i + 1:i + 1 + T] for i in ix])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def val_bpc(model, data, B, T, device, rng, n_batches=8):
    model.eval()
    tot = 0.0
    for _ in range(n_batches):
        x, y = batch(data, B, T, device, rng)
        logits = model(x)
        tot += F.cross_entropy(logits.reshape(-1, model.vocab), y.reshape(-1)).item()
    return tot / n_batches / math.log(2)


def main():
    a = parse()
    torch.manual_seed(a.seed)
    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"

    text = open(a.text, encoding="utf-8", errors="ignore").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    V = len(chars)
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    split = int(0.9 * len(data))
    train, val = data[:split], data[split:]
    p = np.bincount(train, minlength=V) / len(train)
    unigram_bpc = -np.sum(p[p > 0] * np.log2(p[p > 0]))

    hidden = [int(h) for h in a.layers.split(",")]
    rec_edges, ff_edges, tot_edges = [], [None], 0
    for l, H in enumerate(hidden):
        pre, post, S = export_graph(a, H, H, a.rec_fanin * H, sub_seed=l * 2)
        rec_edges.append((pre, post)); tot_edges += S
        if l >= 1:
            fp, fq, Sf = export_graph(a, hidden[l - 1], H, a.ff_fanin * H, sub_seed=l * 2 + 1)
            ff_edges.append((fp, fq)); tot_edges += Sf

    model = SpikingLM(V, hidden, rec_edges, ff_edges, decay=a.decay, thr=a.threshold,
                      ro_decay=a.ro_decay, surrogate_scale=a.surrogate_scale,
                      w_scale=a.w_scale, seed=a.seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(a.seed)
    n_params = sum(pp.numel() for pp in model.parameters())

    print(f"device={device} vocab={V} layers={hidden} structure={a.structure} "
          f"sparse_edges={tot_edges} params={n_params} seq={a.seq} batch={a.batch}")
    print(f"unigram baseline = {unigram_bpc:.3f} bits/char\n")
    print(f"{'iter':>6}  {'train_bpc':>9}  {'val_bpc':>8}")

    for it in range(1, a.iters + 1):
        model.train()
        x, y = batch(train, a.batch, a.seq, device, rng)
        opt.zero_grad()
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % a.eval_interval == 0 or it == 1:
            print(f"{it:>6}  {loss.item()/math.log(2):>9.3f}  "
                  f"{val_bpc(model, val, a.batch, a.seq, device, rng):>8.3f}")

    ids = model.generate([stoi[c] for c in "ROMEO:"], 300, temperature=0.8)
    print(f"\n--- sample (unigram={unigram_bpc:.2f} bits/char) ---")
    print("".join(itos[i] for i in ids))


if __name__ == "__main__":
    main()
