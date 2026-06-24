"""Loaders for spiking-audio benchmarks (Cramer et al. 2019): SHD and SSC.

Both convert spoken audio to spike trains via a cochlear model -- each sample is
a list of (time, unit) events over 700 frequency channels and ~1 s.
  * SHD -- Spiking Heidelberg Digits: 20 classes (digits 0-9, EN+DE).
  * SSC -- Spiking Speech Commands: 35 classes (Google Speech Commands words),
           ~10x larger; harder, the standard "scale up" sibling of SHD.

This bins the variable-length event lists into dense [N, T, 700] binary spike
tensors and caches them as .npz. Fetch the HDF5 files first (gitignored):
    ./scripts/fetch_shd.sh        # data/shd/{shd_train,shd_test}.h5
    ./scripts/fetch_ssc.sh        # data/ssc/{ssc_train,ssc_test}.h5
"""
import os
import numpy as np

N_CHANNELS = 700
CLASSES = {"shd": 20, "ssc": 35}
N_CLASSES = CLASSES["shd"]  # back-compat default


def _bin_split(h5_path, T, t_max):
    import h5py
    dt = t_max / T
    with h5py.File(h5_path, "r") as f:
        times, units = f["spikes"]["times"], f["spikes"]["units"]
        labels = np.asarray(f["labels"], dtype=np.int64)
        n = len(labels)
        x = np.zeros((n, T, N_CHANNELS), dtype=np.uint8)
        for i in range(n):
            ti = (np.asarray(times[i]) / dt).astype(np.int64)
            ui = np.asarray(units[i], dtype=np.int64)
            keep = (ti >= 0) & (ti < T) & (ui >= 0) & (ui < N_CHANNELS)
            x[i, ti[keep], ui[keep]] = 1
    return x, labels


def load_dataset(data_dir, dataset="shd", T=100, t_max=1.0, cache=True):
    """Returns (xtr, ytr, xte, yte, n_classes). x: [N,T,700] uint8, y: [N] int64."""
    cache_path = os.path.join(data_dir, f"{dataset}_binned_T{T}.npz")
    if cache and os.path.exists(cache_path):
        d = np.load(cache_path)
        n = int(d["n_classes"]) if "n_classes" in d.files \
            else CLASSES.get(dataset, int(d["ytr"].max()) + 1)
        return d["xtr"], d["ytr"], d["xte"], d["yte"], n
    xtr, ytr = _bin_split(os.path.join(data_dir, f"{dataset}_train.h5"), T, t_max)
    xte, yte = _bin_split(os.path.join(data_dir, f"{dataset}_test.h5"), T, t_max)
    n_classes = CLASSES.get(dataset, int(max(ytr.max(), yte.max())) + 1)
    if cache:
        np.savez(cache_path, xtr=xtr, ytr=ytr, xte=xte, yte=yte, n_classes=n_classes)
    return xtr, ytr, xte, yte, n_classes


def load_shd(data_dir, T=100, t_max=1.0, cache=True):
    xtr, ytr, xte, yte, _ = load_dataset(data_dir, "shd", T, t_max, cache)
    return xtr, ytr, xte, yte
