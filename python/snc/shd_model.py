"""Recurrent spiking classifier for SHD (and other event/temporal tasks).

The recurrent core is SNC-sparse (topology from snc_export) with optional
heterogeneous conduction delays; trained by surrogate-gradient BPTT. On top of
that structural substrate sit standard, structure-preserving accuracy levers
(all gated by flags so the plain-LIF baseline is reproducible):

  * AdLIF -- adaptive threshold: a per-neuron adaptation variable b raises the
    firing threshold after each spike and decays slowly (Bellec 2020 ALIF /
    Bittar & Garner 2022 AdLIF). Big win on temporal data; also *reduces* firing.
  * Learnable time constants (PLIF, Fang 2021): per-neuron membrane decay and
    adaptation decay/scale are trained, not fixed.
  * Leaky-integrator readout: a non-spiking output layer integrates hidden
    spikes over time, instead of a plain mean spike count.

None of these touch the connectivity -- the SNC-sparse topology, locality, and
delays are unchanged; they are neuron-model / readout / optimizer upgrades.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import SurrogateSpike


def _sparse_init(pre, post, H, scale, gen):
    fanin = torch.zeros(H).index_add(0, post, torch.ones(post.numel())).clamp_min_(1.0)
    w = (torch.rand(pre.numel(), generator=gen) * 2 - 1) * scale
    return w / fanin[post].sqrt()


def _logit(p):
    return torch.log(torch.tensor(p / (1.0 - p)))


class SHDNet(nn.Module):
    def __init__(self, n_in, hidden, n_class, rec_pre, rec_post, in_edges=None,
                 rec_delays=None, decay=0.95, thr=1.0, surrogate_scale=10.0,
                 w_rec_scale=0.5, seed=0, adaptive=True, learn_tau=True,
                 readout="leaky", ro_decay=0.9):
        super().__init__()
        SurrogateSpike.scale = surrogate_scale
        self.H, self.thr = hidden, thr
        self.adaptive, self.readout_mode, self.ro_decay = adaptive, readout, ro_decay
        self.readout = nn.Linear(hidden, n_class)
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("rpre", rec_pre.long())
        self.register_buffer("rpost", rec_post.long())
        self.w_rec = nn.Parameter(_sparse_init(rec_pre.long(), rec_post.long(), hidden, w_rec_scale, gen))

        # Membrane decay: learnable per-neuron (PLIF) or fixed.
        if learn_tau:
            self.decay_logit = nn.Parameter(_logit(decay) * torch.ones(hidden))
        else:
            self.register_buffer("decay_fixed", torch.tensor(float(decay)))
        self.learn_tau = learn_tau
        # AdLIF adaptation: per-neuron decay rho and scale beta (learnable).
        if adaptive:
            self.rho_logit = nn.Parameter(_logit(0.97) * torch.ones(hidden))
            self.beta_raw = nn.Parameter(0.5413 * torch.ones(hidden))  # softplus(0.5413) ~= 1.0

        self.sparse_in = in_edges is not None
        if self.sparse_in:
            ip, iq = in_edges[0].long(), in_edges[1].long()
            self.register_buffer("ipre", ip)
            self.register_buffer("ipost", iq)
            self.w_in = nn.Parameter(_sparse_init(ip, iq, hidden, w_rec_scale, gen))
        else:
            self.win = nn.Linear(n_in, hidden, bias=False)

        self.max_delay = int(rec_delays.max()) if rec_delays is not None else 1
        if self.max_delay > 1:
            self.register_buffer("rdelay", rec_delays.long())
        self.spike_rate = 0.0  # mean firing rate of the last forward (for reg/report)

    def _decay(self):
        return torch.sigmoid(self.decay_logit) if self.learn_tau else self.decay_fixed

    def _input_current(self, xt):
        if self.sparse_in:
            return torch.zeros(xt.shape[0], self.H, device=xt.device).index_add(
                1, self.ipost, self.w_in.unsqueeze(0) * xt[:, self.ipre])
        return self.win(xt)

    def forward(self, x):  # x: [B, T, n_in] -> logits [B, n_class]
        B, T, _ = x.shape
        dev = self.w_rec.device
        decay = self._decay()
        rho = torch.sigmoid(self.rho_logit) if self.adaptive else None
        beta = F.softplus(self.beta_raw) if self.adaptive else None
        v = torch.zeros(B, self.H, device=dev)
        s = torch.zeros(B, self.H, device=dev)
        b = torch.zeros(B, self.H, device=dev)        # adaptation variable
        cnt = torch.zeros(B, self.H, device=dev)      # rate readout accumulator
        out = torch.zeros(B, self.readout.out_features, device=dev)  # leaky readout
        s_hist = (torch.zeros(self.max_delay, B, self.H, device=dev)
                  if self.max_delay > 1 else None)
        spk_total = 0.0
        for t in range(T):
            if self.max_delay > 1:
                src = s_hist[self.rdelay - 1, :, self.rpre]
                rec = torch.zeros(B, self.H, device=dev).index_add(
                    1, self.rpost, (self.w_rec.unsqueeze(1) * src).t())
            else:
                rec = torch.zeros(B, self.H, device=dev).index_add(
                    1, self.rpost, self.w_rec.unsqueeze(0) * s[:, self.rpre])
            v = decay * v + self._input_current(x[:, t]) + rec
            thr_eff = self.thr + (beta * b if self.adaptive else 0.0)
            s = SurrogateSpike.apply(v - thr_eff)
            v = v * (1.0 - s)
            if self.adaptive:
                b = rho * b + (1.0 - rho) * s   # bounded adaptation (~ firing rate)
            spk_total = spk_total + s.sum()
            if self.readout_mode == "leaky":
                out = self.ro_decay * out + self.readout(s)
            else:
                cnt = cnt + s
            if self.max_delay > 1:
                s_hist = torch.cat([s.unsqueeze(0), s_hist[:-1]], dim=0)
        self.spike_rate = float(spk_total.detach()) / (B * self.H * T)
        self.spk_reg = spk_total / (B * self.H * T)  # in-graph, for optional reg
        return out / T if self.readout_mode == "leaky" else self.readout(cnt / T)
