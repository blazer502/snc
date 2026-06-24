"""Recurrent spiking classifier for SHD (and other event/temporal tasks).

A stack of recurrent spiking layers; every recurrent and inter-layer weight is
SNC-sparse (topology from snc_export), with optional heterogeneous conduction
delays on the recurrent cores. Trained by surrogate-gradient BPTT. On top of the
structural substrate sit standard, structure-preserving accuracy levers (gated
by flags so the plain single-layer LIF baseline is reproducible):

  * stacked layers -- depth (2-layer adaptive RSNN is the SHD-competitive shape);
  * AdLIF -- per-neuron adaptive threshold (Bellec 2020 / Bittar & Garner 2022);
  * learnable time constants (PLIF, Fang 2021);
  * leaky-integrator readout from the top layer.

None of these touch the connectivity -- the SNC-sparse topology, locality, and
delays are unchanged; they are neuron-model / readout / depth upgrades.
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
    def __init__(self, n_in, hidden, n_class, rec_edges, ff_edges=None,
                 rec_delays=None, in_edges=None, decay=0.95, thr=1.0,
                 surrogate_scale=10.0, w_rec_scale=0.5, seed=0, adaptive=True,
                 learn_tau=True, readout="leaky", ro_decay=0.9):
        """hidden: list of recurrent-layer widths [H0, H1, ...].
        rec_edges[l]: (pre,post) for the H_l x H_l recurrent core.
        ff_edges[l] (l>=1): (pre,post) for the H_{l-1} x H_l feedforward.
        rec_delays[l]: per-synapse delays for layer l's recurrent core (or None).
        in_edges: sparse 700->H0 input projection (else dense Linear)."""
        super().__init__()
        SurrogateSpike.scale = surrogate_scale
        self.H = list(hidden)
        self.L = len(hidden)
        self.thr, self.adaptive, self.readout_mode, self.ro_decay = thr, adaptive, readout, ro_decay
        self.learn_tau = learn_tau
        self.readout = nn.Linear(hidden[-1], n_class)
        gen = torch.Generator().manual_seed(seed)
        ff_edges = ff_edges or [None] * self.L
        rec_delays = rec_delays or [None] * self.L

        self.w_rec = nn.ParameterList()
        self.w_ff = nn.ParameterList()           # index l-1 holds layer l's feedforward
        self.decay_logit = nn.ParameterList() if learn_tau else None
        self.rho_logit = nn.ParameterList() if adaptive else None
        self.beta_raw = nn.ParameterList() if adaptive else None
        self.max_delay = []
        for l in range(self.L):
            pre, post = rec_edges[l][0].long(), rec_edges[l][1].long()
            self.register_buffer(f"rpre{l}", pre)
            self.register_buffer(f"rpost{l}", post)
            self.w_rec.append(nn.Parameter(_sparse_init(pre, post, self.H[l], w_rec_scale, gen)))
            if learn_tau:
                self.decay_logit.append(nn.Parameter(_logit(decay) * torch.ones(self.H[l])))
            if adaptive:
                self.rho_logit.append(nn.Parameter(_logit(0.97) * torch.ones(self.H[l])))
                self.beta_raw.append(nn.Parameter(0.5413 * torch.ones(self.H[l])))
            md = int(rec_delays[l].max()) if rec_delays[l] is not None else 1
            self.max_delay.append(md)
            if md > 1:
                self.register_buffer(f"rdelay{l}", rec_delays[l].long())
            if l >= 1:
                fp, fq = ff_edges[l][0].long(), ff_edges[l][1].long()
                self.register_buffer(f"fpre{l}", fp)
                self.register_buffer(f"fpost{l}", fq)
                self.w_ff.append(nn.Parameter(_sparse_init(fp, fq, self.H[l], w_rec_scale, gen)))
        if not learn_tau:
            self.register_buffer("decay_fixed", torch.tensor(float(decay)))

        self.sparse_in = in_edges is not None
        if self.sparse_in:
            ip, iq = in_edges[0].long(), in_edges[1].long()
            self.register_buffer("ipre", ip)
            self.register_buffer("ipost", iq)
            self.w_in = nn.Parameter(_sparse_init(ip, iq, self.H[0], w_rec_scale, gen))
        else:
            self.win = nn.Linear(n_in, self.H[0], bias=False)
        self.spike_rate = 0.0

    def _input_current(self, xt):
        if self.sparse_in:
            return torch.zeros(xt.shape[0], self.H[0], device=xt.device).index_add(
                1, self.ipost, self.w_in.unsqueeze(0) * xt[:, self.ipre])
        return self.win(xt)

    def forward(self, x):  # x: [B, T, n_in] -> logits [B, n_class]
        B, T, _ = x.shape
        dev = self.w_rec[0].device
        decay = [torch.sigmoid(self.decay_logit[l]) if self.learn_tau else self.decay_fixed
                 for l in range(self.L)]
        rho = [torch.sigmoid(self.rho_logit[l]) for l in range(self.L)] if self.adaptive else None
        beta = [F.softplus(self.beta_raw[l]) for l in range(self.L)] if self.adaptive else None
        v = [torch.zeros(B, h, device=dev) for h in self.H]
        s = [torch.zeros(B, h, device=dev) for h in self.H]
        bb = [torch.zeros(B, h, device=dev) for h in self.H]
        s_hist = [torch.zeros(self.max_delay[l], B, self.H[l], device=dev)
                  if self.max_delay[l] > 1 else None for l in range(self.L)]
        cnt = torch.zeros(B, self.H[-1], device=dev)
        out = torch.zeros(B, self.readout.out_features, device=dev)
        spk_total = 0.0
        n_units = sum(self.H)
        for t in range(T):
            new_s = []
            for l in range(self.L):
                if self.max_delay[l] > 1:
                    src = s_hist[l][getattr(self, f"rdelay{l}") - 1, :, getattr(self, f"rpre{l}")]
                    rec = torch.zeros(B, self.H[l], device=dev).index_add(
                        1, getattr(self, f"rpost{l}"), (self.w_rec[l].unsqueeze(1) * src).t())
                else:
                    rec = torch.zeros(B, self.H[l], device=dev).index_add(
                        1, getattr(self, f"rpost{l}"), self.w_rec[l].unsqueeze(0) * s[l][:, getattr(self, f"rpre{l}")])
                if l == 0:
                    inp = self._input_current(x[:, t])
                else:
                    inp = torch.zeros(B, self.H[l], device=dev).index_add(
                        1, getattr(self, f"fpost{l}"), self.w_ff[l - 1].unsqueeze(0) * new_s[l - 1][:, getattr(self, f"fpre{l}")])
                vl = decay[l] * v[l] + inp + rec
                thr_eff = self.thr + (beta[l] * bb[l] if self.adaptive else 0.0)
                sp = SurrogateSpike.apply(vl - thr_eff)
                v[l] = vl * (1.0 - sp)
                if self.adaptive:
                    bb[l] = rho[l] * bb[l] + (1.0 - rho[l]) * sp
                new_s.append(sp)
                spk_total = spk_total + sp.sum()
            s = new_s
            if self.readout_mode == "leaky":
                out = self.ro_decay * out + self.readout(s[-1])
            else:
                cnt = cnt + s[-1]
            for l in range(self.L):
                if self.max_delay[l] > 1:
                    s_hist[l] = torch.cat([s[l].unsqueeze(0), s_hist[l][:-1]], dim=0)
        self.spike_rate = float(spk_total.detach()) / (B * n_units * T)
        self.spk_reg = spk_total / (B * n_units * T)
        return out / T if self.readout_mode == "leaky" else self.readout(cnt / T)
