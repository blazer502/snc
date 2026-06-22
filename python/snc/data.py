"""MNIST IDX loader (numpy) -- reads the uncompressed files fetched by
scripts/fetch_mnist.sh into data/mnist."""
import os
import numpy as np


def _images(path):
    with open(path, "rb") as f:
        _, n, r, c = np.frombuffer(f.read(16), dtype=">i4")
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(int(n), int(r) * int(c))
    return (data.astype(np.float32) / 255.0)


def _labels(path):
    with open(path, "rb") as f:
        _, n = np.frombuffer(f.read(8), dtype=">i4")
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def load_mnist(data_dir, n_train, n_test):
    xtr = _images(os.path.join(data_dir, "train-images-idx3-ubyte"))[:n_train]
    ytr = _labels(os.path.join(data_dir, "train-labels-idx1-ubyte"))[:n_train]
    xte = _images(os.path.join(data_dir, "t10k-images-idx3-ubyte"))[:n_test]
    yte = _labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte"))[:n_test]
    return xtr, ytr, xte, yte
