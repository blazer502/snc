"""Recurrent spiking character-level language model on the SNC substrate.

The "LLM-shaped" task is next-token prediction. SNNs are natively temporal, so
the SNC-faithful model is a *stack* of recurrent spiking layers trained by BPTT:

    token --Embed--> input current to layer 0
              │
        layer 0: LIF pool, SNC-sparse recurrent core (H0->H0)
              │ SNC-sparse feedforward (H0->H1)
        layer 1: LIF pool, SNC-sparse recurrent core (H1->H1)
              │ ...
        top layer spikes --low-pass trace--> linear readout --> next-token logits

Per token step t, layers update bottom-up; each layer's recurrence reads its own
previous-step spikes, its feedforward reads the current step of the layer below:

    rec_l   = W_rec[l] · s_l(t-1)
    ff_l    = Embed(x_t)            if l == 0   else  W_ff[l] · s_{l-1}(t)
    v_l(t)  = decay·v_l(t-1) + ff_l + rec_l ;  s_l(t) = H(v_l - thr) ;  reset
    r(t)    = ro_decay·r(t-1) + s_top(t) ;  logit(t) = Readout(r(t))

PyTorch autograd unrolls the token loop -> true surrogate-gradient BPTT. Every
recurrent and feedforward weight is SNC-sparse (edges from snc_export); only the
token embedding and the readout are dense. Single-layer is the L==1 case.
"""
import torch
import torch.nn as nn

from .model import SurrogateSpike


def _sparse_init(pre, post, H, scale, gen):
    """Signed weights ~ U[-scale,scale] / sqrt(fan-in), per-edge for a sparse
    [*, H] graph addressed by (pre, post)."""
    fanin = torch.zeros(H).index_add(0, post, torch.ones(post.numel())).clamp_min_(1.0)
    w = (torch.rand(pre.numel(), generator=gen) * 2 - 1) * scale
    return w / fanin[post].sqrt()


class SpikingLM(nn.Module):
    def __init__(self, vocab, hidden, rec_edges, ff_edges, decay=0.95, thr=1.0,
                 ro_decay=0.9, surrogate_scale=10.0, w_scale=0.5, seed=0):
        """hidden: list of layer widths [H0, H1, ...].
        rec_edges[l]: (pre, post) for the H_l x H_l recurrent core.
        ff_edges[l]:  (pre, post) for the H_{l-1} x H_l feedforward (l >= 1)."""
        super().__init__()
        SurrogateSpike.scale = surrogate_scale
        self.H = list(hidden)
        self.L = len(hidden)
        self.vocab = vocab
        self.decay, self.thr, self.ro_decay = decay, thr, ro_decay
        self.embed = nn.Embedding(vocab, hidden[0])
        self.readout = nn.Linear(hidden[-1], vocab)
        gen = torch.Generator().manual_seed(seed)
        self.w_rec = nn.ParameterList()
        self.w_ff = nn.ParameterList()  # index l-1 holds layer l's feedforward
        for l in range(self.L):
            pre, post = rec_edges[l][0].long(), rec_edges[l][1].long()
            self.register_buffer(f"rpre{l}", pre)
            self.register_buffer(f"rpost{l}", post)
            self.w_rec.append(nn.Parameter(_sparse_init(pre, post, self.H[l], w_scale, gen)))
            if l >= 1:
                fp, fq = ff_edges[l][0].long(), ff_edges[l][1].long()
                self.register_buffer(f"fpre{l}", fp)
                self.register_buffer(f"fpost{l}", fq)
                self.w_ff.append(nn.Parameter(_sparse_init(fp, fq, self.H[l], w_scale, gen)))

    def _zero_state(self, B, dev):
        v = [torch.zeros(B, h, device=dev) for h in self.H]
        s = [torch.zeros(B, h, device=dev) for h in self.H]
        r = torch.zeros(B, self.H[-1], device=dev)
        return v, s, r

    def step(self, tok, v, s, r):
        new_s = []
        for l in range(self.L):
            rpre, rpost = getattr(self, f"rpre{l}"), getattr(self, f"rpost{l}")
            rec = torch.zeros_like(v[l]).index_add(
                1, rpost, self.w_rec[l].unsqueeze(0) * s[l][:, rpre])
            if l == 0:
                ff = self.embed(tok)
            else:
                fpre, fpost = getattr(self, f"fpre{l}"), getattr(self, f"fpost{l}")
                ff = torch.zeros_like(v[l]).index_add(
                    1, fpost, self.w_ff[l - 1].unsqueeze(0) * new_s[l - 1][:, fpre])
            vl = self.decay * v[l] + ff + rec
            sp = SurrogateSpike.apply(vl - self.thr)
            v[l] = vl * (1.0 - sp)
            new_s.append(sp)
        r = self.ro_decay * r + new_s[-1]
        return v, new_s, r

    def forward(self, x):  # x: [B, T] token ids -> logits [B, T, vocab]
        B, T = x.shape
        v, s, r = self._zero_state(B, self.w_rec[0].device)
        logits = []
        for t in range(T):
            v, s, r = self.step(x[:, t], v, s, r)
            logits.append(self.readout(r))
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def generate(self, prefix_ids, n, temperature=0.8):
        dev = self.w_rec[0].device
        v, s, r = self._zero_state(1, dev)
        out = list(prefix_ids)
        for tid in prefix_ids:
            v, s, r = self.step(torch.tensor([tid], device=dev), v, s, r)
        for _ in range(n):
            logit = self.readout(r)[0] / max(1e-6, temperature)
            tid = int(torch.multinomial(torch.softmax(logit, -1), 1).item())
            out.append(tid)
            v, s, r = self.step(torch.tensor([tid], device=dev), v, s, r)
        return out
