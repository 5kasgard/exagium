```hypothesis
Family: label-onehot-mlp. Each byte feature is one-hot encoded over its training-set unique values, then a one-hidden-layer ReLU MLP is trained with Adam (lr=3e-3, L2=1e-4). This captures non-linear per-pixel responses that linear models miss. On optdigits, pixel intensities 0-16 have non-monotonic class relationships, so one-hot + MLP should exceed label-linear's 0.9613 dev accuracy. If dev accuracy stays below 0.965, the added complexity is not justified.
```

```python
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


def build_encoding(X, unique_vals_per_feature):
    d = X.shape[1]
    enc = []
    for j in range(d):
        vals = unique_vals_per_feature[j]
        dim = len(vals)
        E = np.zeros((256, dim), dtype=np.float32)
        E[vals, np.arange(dim)] = 1.0
        enc.append(E)
    return enc


def train_model(X, y, H, enc, C, rng):
    n, d = X.shape
    total_dim = sum(E.shape[1] for E in enc)
    W = H if H > 0 else C
    if H > 0:
        W1 = (rng.uniform(-1, 1, (total_dim, H)) * math.sqrt(6.0 / total_dim)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np