"""Recurrent spiking classifier for SHD (and other event/temporal tasks).

A leaky-integrate-and-fire recurrent layer whose recurrent core is SNC-sparse
(topology from snc_export), trained by surrogate-gradient BPTT. Input spikes are
projected in by a dense layer; a rate readout (mean hidden spike count over time)
feeds a linear classifier. This is the standard RSNN shape for SHD, with the
recurrent connectivity supplied by SNC's structure generators so we can ask
whether a morphology/locality prior helps on genuinely temporal data.

    v_t = decay·v_{t-1} + W_in·x_t + W_rec·s_{t-1}
    s_t = surrogate_Heaviside(v_t - thr) ;  reset
    logits = Readout( mean_t s_t )
"""
import torch
import torch.nn as nn

from .model import SurrogateSpike


def _sparse_init(pre, post, H, scale, gen):
    fanin = torch.zeros(H).index_add(0, post, torch.ones(post.numel())).clamp_min_(1.0)
    w = (torch.rand(pre.numel(), generator=gen) * 2 - 1) * scale
    return w / fanin[post].sqrt()


class SHDNet(nn.Module):
    def __init__(self, n_in, hidden, n_class, rec_pre, rec_post, in_edges=None,
                 decay=0.95, thr=1.0, surrogate_scale=10.0, w_rec_scale=0.5, seed=0):
        """Recurrent core is always SNC-sparse (rec_pre/rec_post). The input
        projection is dense unless `in_edges=(pre,post)` over [n_in, hidden] is
        given, in which case it is SNC-sparse (e.g. frequency-local for SHD)."""
        super().__init__()
        SurrogateSpike.scale = surrogate_scale
        self.H, self.decay, self.thr = hidden, decay, thr
        self.readout = nn.Linear(hidden, n_class)
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("rpre", rec_pre.long())
        self.register_buffer("rpost", rec_post.long())
        self.w_rec = nn.Parameter(_sparse_init(rec_pre.long(), rec_post.long(), hidden, w_rec_scale, gen))
        self.sparse_in = in_edges is not None
        if self.sparse_in:
            ip, iq = in_edges[0].long(), in_edges[1].long()
            self.register_buffer("ipre", ip)
            self.register_buffer("ipost", iq)
            self.w_in = nn.Parameter(_sparse_init(ip, iq, hidden, w_rec_scale, gen))
        else:
            self.win = nn.Linear(n_in, hidden, bias=False)

    def _input_current(self, xt):  # xt: [B, n_in] -> [B, H]
        if self.sparse_in:
            return torch.zeros(xt.shape[0], self.H, device=xt.device).index_add(
                1, self.ipost, self.w_in.unsqueeze(0) * xt[:, self.ipre])
        return self.win(xt)

    def forward(self, x):  # x: [B, T, n_in] -> logits [B, n_class]
        B, T, _ = x.shape
        dev = self.w_rec.device
        v = torch.zeros(B, self.H, device=dev)
        s = torch.zeros(B, self.H, device=dev)
        cnt = torch.zeros(B, self.H, device=dev)
        for t in range(T):
            rec = torch.zeros(B, self.H, device=dev).index_add(
                1, self.rpost, self.w_rec.unsqueeze(0) * s[:, self.rpre])
            v = self.decay * v + self._input_current(x[:, t]) + rec
            s = SurrogateSpike.apply(v - self.thr)
            v = v * (1.0 - s)
            cnt = cnt + s
        return self.readout(cnt / T)
