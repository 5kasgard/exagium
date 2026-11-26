"""Drifting-stream label task generator: fixed seeds, no download."""
import numpy as np


def generate(seed, n, dim=16, segments=4):
    """Return (X, y, W): a stream split into `segments` equal blocks.

    Block s follows unit rule W[s]; y = 1 iff x . W[s] > 0. X is the
    per-column standard-normal x quantized to uint8 bytes.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(segments, dim))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    X = rng.normal(size=(n, dim))
    y = np.empty(n, dtype=np.uint8)
    bounds = np.linspace(0, n, segments + 1, dtype=int)
    for s in range(segments):
        block = slice(bounds[s], bounds[s + 1])
        y[block] = (X[block] @ W[s] > 0).astype(np.uint8)
    X = np.clip(np.round((X + 4.0) / 8.0 * 255.0), 0, 255).astype(np.uint8)
    return X, y, W
