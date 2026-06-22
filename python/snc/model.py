"""Surrogate-gradient spiking network over an SNC sparse topology (Phase 5).

The SNN is a plain torch.nn.Module: it runs `num_steps` of LIF dynamics with a
non-differentiable Heaviside spike whose backward pass uses a smooth surrogate
(fast-sigmoid). PyTorch autograd unrolls the timestep loop, giving true
backprop-through-time. Connectivity comes from the exported SNC graph as fixed
pre/post index buffers; the signed per-synapse weight is the trainable parameter.

Delay-1 feedforward topologies only (the SNC generators' default): step t reads
the previous step's spikes. Encoding is Poisson or direct, on-device.
"""
import numpy as np
import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    """Heaviside forward; fast-sigmoid surrogate derivative on backward."""
    scale = 10.0

    @staticmethod
    def forward(ctx, x):  # x = v - threshold
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        sg = 1.0 / (1.0 + SurrogateSpike.scale * x.abs()) ** 2
        return grad_out * sg


class SNN(nn.Module):
    def __init__(self, g, num_steps=20, thr=1.0, decay=0.9, gain=1.0, max_rate=0.5,
                 w_init=0.1, surrogate_scale=10.0, encoder="poisson", seed=0):
        super().__init__()
        if (g["delays"] != 1).any():
            raise ValueError("torch SNN path supports delay==1 graphs only")
        SurrogateSpike.scale = surrogate_scale
        self.N, self.dim, self.n_out = g["N"], g["n_in"], g["n_out"]
        self.T, self.thr, self.decay = num_steps, thr, decay
        self.gain, self.max_rate, self.encoder = gain, max_rate, encoder
        self.register_buffer("pre", torch.as_tensor(g["pre"], dtype=torch.long))
        self.register_buffer("post", torch.as_tensor(g["post"], dtype=torch.long))
        out_ids = np.where(g["role"] == 2)[0]
        self.register_buffer("out_ids", torch.as_tensor(out_ids, dtype=torch.long))
        self.register_buffer("out_ch",
                             torch.as_tensor(g["channel"][out_ids], dtype=torch.long))
        gen = torch.Generator().manual_seed(seed)
        self.w = nn.Parameter((torch.rand(g["S"], generator=gen) * 2 - 1) * w_init)

    def forward(self, x):  # x: [B, dim] in [0,1] -> logits [B, n_out]
        B, dev = x.shape[0], self.w.device
        v = torch.zeros(B, self.N, device=dev)
        s = torch.zeros(B, self.N, device=dev)
        out = torch.zeros(B, self.n_out, device=dev)
        for _ in range(self.T):
            # synaptic current from the previous step's spikes (delay 1)
            msg = self.w.unsqueeze(0) * s[:, self.pre]                # [B, S]
            cur = torch.zeros(B, self.N, device=dev).index_add(1, self.post, msg)
            # external drive into the input neurons (ids 0..dim-1)
            if self.encoder == "poisson":
                inp = (torch.rand(B, self.dim, device=dev) < x * self.max_rate).float()
            else:  # direct current
                inp = x
            ext = torch.zeros(B, self.N, device=dev)
            ext[:, :self.dim] = inp * self.gain
            v = self.decay * v + cur + ext
            s = SurrogateSpike.apply(v - self.thr)
            v = v * (1.0 - s)  # reset neurons that fired
            out = out.index_add(1, self.out_ch, s[:, self.out_ids])
        return out
