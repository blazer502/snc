"""Load an SNC graph exported by the C++ `snc_export` tool (the PyTorch bridge).

Returns the CSR connectivity (pre/post/delays) and per-neuron role/channel so the
torch SNN can train the *exact* topology produced by SNC's structure generators.
"""
import numpy as np

MAGIC = 0x534E4347  # 'SNCG'


def load_graph(path):
    with open(path, "rb") as f:
        magic, N, S, n_in, n_out = np.fromfile(f, dtype="<i4", count=5)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic 0x{magic:08x}")
        pre = np.fromfile(f, dtype="<i4", count=S)
        post = np.fromfile(f, dtype="<i4", count=S)
        delays = np.fromfile(f, dtype="<i4", count=S)
        role = np.fromfile(f, dtype="<i4", count=N)
        channel = np.fromfile(f, dtype="<i4", count=N)
    return {
        "N": int(N), "S": int(S), "n_in": int(n_in), "n_out": int(n_out),
        "pre": pre, "post": post, "delays": delays, "role": role, "channel": channel,
    }
