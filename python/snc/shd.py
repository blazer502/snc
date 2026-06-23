"""Spiking Heidelberg Digits (SHD) loader.

SHD (Cramer et al. 2019) is spoken digits 0-9 in English + German (20 classes),
converted to spike trains by a cochlear model: each sample is a list of
(time, unit) events over 700 frequency channels and ~1 s. A genuinely temporal,
neuromorphic benchmark -- the regime where SNC's locality + timing prior should
matter most.

This bins the variable-length event lists into dense [N, T, 700] binary spike
tensors and caches them as .npy for fast reload. Fetch the HDF5 files first:
    curl -L https://zenkelab.org/datasets/shd_train.h5.gz | gunzip > data/shd/shd_train.h5
    curl -L https://zenkelab.org/datasets/shd_test.h5.gz  | gunzip > data/shd/shd_test.h5
"""
import os
import numpy as np

N_CHANNELS = 700
N_CLASSES = 20


def _bin_split(h5_path, T, t_max):
    import h5py
    dt = t_max / T
    with h5py.File(h5_path, "r") as f:
        times = f["spikes"]["times"]
        units = f["spikes"]["units"]
        labels = np.asarray(f["labels"], dtype=np.int64)
        n = len(labels)
        x = np.zeros((n, T, N_CHANNELS), dtype=np.uint8)
        for i in range(n):
            ti = (np.asarray(times[i]) / dt).astype(np.int64)
            ui = np.asarray(units[i], dtype=np.int64)
            keep = (ti >= 0) & (ti < T) & (ui >= 0) & (ui < N_CHANNELS)
            x[i, ti[keep], ui[keep]] = 1
    return x, labels


def load_shd(data_dir, T=100, t_max=1.0, cache=True):
    """Returns (xtr, ytr, xte, yte); x: [N, T, 700] uint8, y: [N] int64."""
    cache_path = os.path.join(data_dir, f"shd_binned_T{T}.npz")
    if cache and os.path.exists(cache_path):
        d = np.load(cache_path)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]
    xtr, ytr = _bin_split(os.path.join(data_dir, "shd_train.h5"), T, t_max)
    xte, yte = _bin_split(os.path.join(data_dir, "shd_test.h5"), T, t_max)
    if cache:
        np.savez(cache_path, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    return xtr, ytr, xte, yte
