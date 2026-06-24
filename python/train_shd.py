#!/usr/bin/env python3
"""Train the (stacked) recurrent spiking SHD classifier by surrogate-gradient BPTT.

Genuinely temporal benchmark (spoken digits as cochlear spike trains). Every
recurrent and inter-layer weight uses an SNC-generated sparse topology with
optional heterogeneous conduction delays; PyTorch owns data/optimizer/loss.
Structure-preserving accuracy/efficiency levers: stacked layers (--layers),
AdLIF adaptive threshold (--adaptive), learnable time constants (--learn-tau),
SHD augmentation (--augment), spike-frequency regularization (--spike-reg).

    python3 python/train_shd.py --layers 256,256 --adaptive 1 --learn-tau 1 \\
        --augment 1 --delay-max 30 --epochs 80 --device cuda
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
from snc.shd import load_dataset, N_CHANNELS
from snc.shd_model import SHDNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="shd", choices=["shd", "ssc"])
    p.add_argument("--data-dir", default="", help="default: data/<dataset>")
    p.add_argument("--layers", default="256", help="recurrent layer widths, e.g. 256 or 256,256")
    p.add_argument("--rec-structure", default="random-sparse",
                   choices=["dense", "random-sparse", "static-snc"])
    p.add_argument("--rec-budget", type=int, default=8192)
    p.add_argument("--ff-structure", default="random-sparse",
                   choices=["dense", "random-sparse", "static-snc"])
    p.add_argument("--ff-budget", type=int, default=16384)
    p.add_argument("--delay-max", type=int, default=1)
    p.add_argument("--delay-mode", default="random", choices=["random", "distance"])
    p.add_argument("--in-structure", default="dense",
                   choices=["dense", "random-sparse", "static-snc"])
    p.add_argument("--in-budget", type=int, default=16384)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--decay", type=float, default=0.95)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--surrogate-scale", type=float, default=10.0)
    p.add_argument("--w-rec-scale", type=float, default=0.5)
    p.add_argument("--adaptive", type=int, default=0)
    p.add_argument("--learn-tau", type=int, default=0)
    p.add_argument("--readout", default="rate", choices=["rate", "leaky"])
    p.add_argument("--spike-reg", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--augment", type=int, default=0)
    p.add_argument("--aug-tshift", type=int, default=8, help="max time-bin shift")
    p.add_argument("--aug-cshift", type=int, default=8, help="max channel shift")
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
           "--delay-max", str(delay_max), "--delay-mode", a.delay_mode, "--out", tmp.name]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"snc_export failed:\n{r.stderr}")
    g = snc.load_graph(tmp.name)
    pre = torch.from_numpy(g["pre"].astype(np.int64))
    post = torch.from_numpy(g["post"].astype(np.int64)) - A
    delays = torch.from_numpy(g["delays"].astype(np.int64))
    return pre, post, delays, g["S"]


def augment(x, tshift, cshift):
    """Per-sample random time + channel roll of a [B,T,C] spike tensor (on device)."""
    B, T, C = x.shape
    if tshift > 0:
        sh = torch.randint(-tshift, tshift + 1, (B,), device=x.device)
        idx = (torch.arange(T, device=x.device)[None, :] - sh[:, None]) % T
        x = torch.gather(x, 1, idx[:, :, None].expand(-1, -1, C))
    if cshift > 0:
        sh = torch.randint(-cshift, cshift + 1, (B,), device=x.device)
        idx = (torch.arange(C, device=x.device)[None, :] - sh[:, None]) % C
        x = torch.gather(x, 2, idx[:, None, :].expand(-1, T, -1))
    return x


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

    data_dir = a.data_dir or os.path.join(ROOT, "data", a.dataset)
    xtr, ytr, xte, yte, n_class = load_dataset(data_dir, a.dataset, T=a.steps)
    hidden = [int(h) for h in a.layers.split(",")]
    rec_edges, rec_delays, ff_edges, tot_edges = [], [], [None], 0
    for l, H in enumerate(hidden):
        pre, post, delays, S = export_graph(a, a.rec_structure, H, H, a.rec_budget,
                                            sub_seed=l * 3, delay_max=a.delay_max)
        rec_edges.append((pre, post)); tot_edges += S
        rec_delays.append(delays if a.delay_max > 1 else None)
        if l >= 1:
            fp, fq, _, Sf = export_graph(a, a.ff_structure, hidden[l - 1], H, a.ff_budget, l * 3 + 1)
            ff_edges.append((fp, fq)); tot_edges += Sf
    in_edges = None
    if a.in_structure != "dense":
        ip, iq, _, _ = export_graph(a, a.in_structure, N_CHANNELS, hidden[0], a.in_budget, 999)
        in_edges = (ip, iq)

    model = SHDNet(N_CHANNELS, hidden, n_class, rec_edges, ff_edges=ff_edges,
                   rec_delays=rec_delays, in_edges=in_edges, decay=a.decay, thr=a.threshold,
                   surrogate_scale=a.surrogate_scale, w_rec_scale=a.w_rec_scale, seed=a.seed,
                   adaptive=bool(a.adaptive), learn_tau=bool(a.learn_tau), readout=a.readout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"device={device} {a.dataset.upper()} train={len(xtr)} test={len(xte)} classes={n_class} "
          f"steps={a.steps} layers={hidden} rec={a.rec_structure} delay_max={a.delay_max}({a.delay_mode}) "
          f"adaptive={a.adaptive} learn_tau={a.learn_tau} augment={a.augment} "
          f"sparse_edges={tot_edges} params={n_params} chance={1/n_class:.3f}")
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
            if a.augment:
                xb = augment(xb, a.aug_tshift, a.aug_cshift)
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
