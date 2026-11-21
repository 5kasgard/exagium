```hypothesis
Family: label-localquad (local quadratic logistic regression). Adds 210 pairwise products of adjacent pixels (horizontal, vertical, both diagonals) to the 64 linear features, capturing stroke curvature in 8×8 digits. Trains with same Adam/L2 regimen as label-linear. Expected to beat 0.9613 at ~2× instruction cost (still <10k Ir/item). If dev accuracy ≤ 0.965, the spatial inductive bias fails to help.
```
```python
"""label-localquad: local quadratic logistic regression for 8x8 optdigits.
Adds 210 adjacent-pixel products to 64 linear features. Trains with Adam/L2.
Weights format: magic 0x4C4F514C, d=64, P=210, C=10; float32 mean[d], inv_std[d], W_lin[d*C], W_quad[P*C], bias[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gen_pairs(d=64, h=8, w=8):
    """Return list of (j,k) for adjacent pixels in 8x8 grid."""
    pairs = []
    # horizontal
    for i in range(h):
        for j in range(w-1):
            pairs.append((i*w + j, i*w + j + 1))
    # vertical
    for i in range(h-1):
        for j in range(w):
            pairs.append((i*w + j, (i+1)*w + j))
    # diagonal down-right
    for i in range(h-1):
        for j in range(w-1):
            pairs.append((i*w + j, (i+1)*w + j + 1))
    # diagonal down-left
    for i in range(h-1):
        for j in range(1, w):
            pairs.append((i*w + j, (i+1)*w + j - 1))
    assert len(pairs) == 210
    return pairs

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    assert d == 64
    rng = np.random.default_rng(SEED)
    
    # Standardize linear features
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    inv_sd = 1.0 / sd
    Z_lin = ((X - mu) * inv_sd).astype(np.float32)  # (n, 64)
    
    # Generate quadratic features
    pairs = gen_pairs()
    P = len(pairs)
    Z_quad = np.empty((X.shape[0], P), dtype=np.float32)
    for idx, (j, k) in enumerate(pairs):
        Z_quad[:, idx] = Z_lin[:, j] * Z_lin[:, k]
    
    Z = np.concatenate([Z_lin, Z_quad], axis=1)  # (n, 64+P)
    n, k = Z.shape
    assert k == 64 + P
    
    # Initialize weights
    scale = math.sqrt(6.0 / k)
    W_lin = (rng.uniform(-1, 1, (64, C)) * scale).astype(np.float32)
    W_quad = (rng.uniform(-1, 1, (P, C)) * scale).astype(np.float32)
    bias = np.zeros(C,