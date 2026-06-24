#!/usr/bin/env python3
"""Train the recurrent spiking SHD classifier by surrogate-gradient BPTT.

Genuinely temporal benchmark (spoken digits as cochlear spike trains). The
recurrent core uses an SNC-generated sparse topology, so we can ask whether the
morphology/locality prior helps on event/temporal data, where it should matter
more than on static MNIST.

    python3 python/train_shd.py --rec-structure static-snc --hidden 256 \\
        --epochs 30 --device cuda
"""
import argparse
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F

import snc
from snc.shd import load_shd, N_CHANNELS, N_CLASSES
from snc.shd_model import SHDNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data/shd"))
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--rec-structure", default="static-snc",
                   choices=["dense", "random-sparse", "static-snc"])
    p.add_argument("--rec-budget", type=int, default=8192)
    p.add_argument("--delay-max", type=int, default=1,
                   help=">1 gives recurrent synapses a delay spread [1,delay-max]")
    p.add_argument("--in-structure", default="dense",
                   choices=["dense", "random-sparse", "static-snc"],
                   help="input projection topology (700 cochlear channels -> hidden)")
    p.add_argument("--in-budget", type=int, default=16384)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--decay", type=float, default=0.95)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--surrogate-scale", type=float, default=10.0)
    p.add_argument("--w-rec-scale", type=float, default=0.5)
    # structure-preserving accuracy/efficiency upgrades
    p.add_argument("--adaptive", type=int, default=0, help="AdLIF adaptive threshold (1/0; opt-in)")
    p.add_argument("--learn-tau", type=int, default=0, help="learnable membrane decay PLIF (1/0; opt-in)")
    p.add_argument("--readout", default="rate", choices=["rate", "leaky"])
    p.add_argument("--spike-reg", type=float, default=0.0, help="firing-rate penalty (inference cost)")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--export-bin", default=os.path.join(ROOT, "build/snc_export"))
    p.add_argument("--log-csv", default="")
    return p.parse_args()


def export_graph(a, structure, A, B, budget, sub_seed, delay_max=1):
    """SNC [A,B] graph -> (pre in [0,A), post in [0,B), delays) + edge count."""
    tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False); tmp.close()
    cmd = [a.export_bin, "--structure", structure, "--layers", f"{A},{B}",
           "--synapse-budget", str(budget), "--seed", str(a.seed * 100 + sub_seed),
           "--delay-max", str(delay_max), "--out", tmp.name]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"snc_export failed:\n{r.stderr}")
    print(r.stdout.strip())
    g = snc.load_graph(tmp.name)
    pre = torch.from_numpy(g["pre"].astype(np.int64))
    post = torch.from_numpy(g["post"].astype(np.int64)) - A
    delays = torch.from_numpy(g["delays"].astype(np.int64))
    return pre, post, delays, g["S"]


@torch.no_grad()
def evaluate(model, X, y, batch, device):
    model.eval()
    correct = 0
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().to(device)
        correct += (model(xb).argmax(1).cpu().numpy() == y[i:i + batch]).sum()
    return correct / len(X)


def main():
    a = parse()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"

    xtr, ytr, xte, yte = load_shd(a.data_dir, T=a.steps)
    pre, post, rec_delays, E = export_graph(a, a.rec_structure, a.hidden, a.hidden,
                                            a.rec_budget, 0, delay_max=a.delay_max)
    in_edges = None
    if a.in_structure != "dense":
        ip, iq, _, _ = export_graph(a, a.in_structure, N_CHANNELS, a.hidden, a.in_budget, 1)
        in_edges = (ip, iq)
    model = SHDNet(N_CHANNELS, a.hidden, N_CLASSES, pre, post, in_edges=in_edges,
                   rec_delays=(rec_delays if a.delay_max > 1 else None),
                   decay=a.decay, thr=a.threshold, surrogate_scale=a.surrogate_scale,
                   w_rec_scale=a.w_rec_scale, seed=a.seed, adaptive=bool(a.adaptive),
                   learn_tau=bool(a.learn_tau), readout=a.readout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"device={device} SHD train={len(xtr)} test={len(xte)} steps={a.steps} "
          f"rec={a.rec_structure} in={a.in_structure} delay_max={a.delay_max} "
          f"adaptive={a.adaptive} learn_tau={a.learn_tau} readout={a.readout} "
          f"rec_edges={E} params={n_params} chance={1/N_CLASSES:.3f}")
    print(f"\n{'epoch':>5}  {'loss':>6}  {'train_acc':>9}  {'test_acc':>8}  {'spk/n/step':>10}")
    csv = open(a.log_csv, "w") if a.log_csv else None
    if csv: csv.write("epoch,loss,train_acc,test_acc,spike_rate\n")

    best = 0.0
    for e in range(1, a.epochs + 1):
        model.train()
        order = np.random.permutation(len(xtr))
        loss_sum, correct = 0.0, 0
        for i in range(0, len(xtr), a.batch):
            idx = order[i:i + a.batch]
            xb = torch.from_numpy(xtr[idx]).float().to(device)
            yb = torch.from_numpy(ytr[idx]).to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb) + a.spike_reg * model.spk_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
        sched.step()
        tr = correct / len(xtr)
        te = evaluate(model, xte, yte, a.batch, device)
        best = max(best, te)
        print(f"{e:>5}  {loss_sum/len(xtr):>6.3f}  {tr:>9.4f}  {te:>8.4f}  {model.spike_rate:>10.4f}")
        if csv: csv.write(f"{e},{loss_sum/len(xtr):.5f},{tr:.5f},{te:.5f},{model.spike_rate:.5f}\n")
    print(f"best test_acc = {best:.4f}")
    if csv: csv.close(); print(f"wrote {a.log_csv}")


if __name__ == "__main__":
    main()
