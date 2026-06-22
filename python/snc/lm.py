"""Recurrent spiking character-level language model on the SNC substrate.

The "LLM-shaped" task: next-token prediction. An LLM is a sequence model trained
to predict the next token; SNNs are natively temporal, so the SNC-faithful
version is a recurrent spiking net trained by BPTT:

    token --embed--> input current
            │
       LIF hidden pool with an SNC-SPARSE recurrent core (hidden->hidden)
       membrane + spikes carry sequence state across token steps
            │
       low-pass spike trace --linear readout--> next-token logits

Recurrence (per token step t, given previous spikes s):
    rec_t = W_rec · s_{t-1}            # sparse, SNC topology
    v_t   = decay·v_{t-1}(1-s_{t-1}) + Embed(x_t) + rec_t
    s_t   = surrogate_Heaviside(v_t - thr)
    r_t   = ro_decay·r_{t-1} + s_t     # filtered spikes
    logit_t = Readout(r_t)             # predicts x_{t+1}

PyTorch autograd unrolls the whole sequence -> true BPTT. The recurrent core is
sparse with an SNC-generated topology; embedding and readout are small dense
layers (PyTorch §9 split: SNC owns the recurrent structure, torch owns the rest).
"""
import math
import torch
import torch.nn as nn

from .model import SurrogateSpike


class SpikingLM(nn.Module):
    def __init__(self, vocab, hidden, rec_pre, rec_post, decay=0.95, thr=1.0,
                 ro_decay=0.9, surrogate_scale=10.0, w_rec_scale=0.5, seed=0):
        super().__init__()
        SurrogateSpike.scale = surrogate_scale
        self.H, self.vocab = hidden, vocab
        self.decay, self.thr, self.ro_decay = decay, thr, ro_decay
        self.register_buffer("rpre", rec_pre.long())
        self.register_buffer("rpost", rec_post.long())
        # per-edge fan-in for a sane recurrent init
        fanin = torch.zeros(hidden).index_add(
            0, rec_post.long(), torch.ones(rec_post.numel())).clamp_min_(1.0)
        gen = torch.Generator().manual_seed(seed)
        w0 = (torch.rand(rec_pre.numel(), generator=gen) * 2 - 1) * w_rec_scale
        self.w_rec = nn.Parameter(w0 / fanin[rec_post.long()].sqrt())
        self.embed = nn.Embedding(vocab, hidden)
        self.readout = nn.Linear(hidden, vocab)

    def step(self, tok, v, s, r):
        rec = torch.zeros_like(v).index_add(
            1, self.rpost, self.w_rec.unsqueeze(0) * s[:, self.rpre])
        v = self.decay * v + self.embed(tok) + rec
        s = SurrogateSpike.apply(v - self.thr)
        v = v * (1.0 - s)
        r = self.ro_decay * r + s
        return v, s, r

    def forward(self, x):  # x: [B, T] token ids -> logits [B, T, vocab]
        B, T = x.shape
        dev = self.w_rec.device
        v = torch.zeros(B, self.H, device=dev)
        s = torch.zeros(B, self.H, device=dev)
        r = torch.zeros(B, self.H, device=dev)
        logits = []
        for t in range(T):
            v, s, r = self.step(x[:, t], v, s, r)
            logits.append(self.readout(r))
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def generate(self, prefix_ids, n, temperature=0.8):
        """Autoregressively sample `n` tokens after the prefix. Returns id list."""
        dev = self.w_rec.device
        v = torch.zeros(1, self.H, device=dev)
        s = torch.zeros(1, self.H, device=dev)
        r = torch.zeros(1, self.H, device=dev)
        out = list(prefix_ids)
        tok = None
        for tid in prefix_ids:  # warm up the state on the prefix
            tok = torch.tensor([tid], device=dev)
            v, s, r = self.step(tok, v, s, r)
        for _ in range(n):
            logit = self.readout(r)[0] / max(1e-6, temperature)
            p = torch.softmax(logit, dim=-1)
            tid = int(torch.multinomial(p, 1).item())
            out.append(tid)
            tok = torch.tensor([tid], device=dev)
            v, s, r = self.step(tok, v, s, r)
        return out
