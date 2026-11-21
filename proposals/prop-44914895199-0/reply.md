```hypothesis
Family: label-fm (Factorization Machine with K=8). 
Why: Adult has strong categorical interactions (e.g., occupation × education) that linear models miss; FM captures all pairwise interactions via low-rank embeddings with minimal overhead. 
Failure signal: dev accuracy ≤ label-linear (0.8524) or inference cost > 2× label-linear.
```

```python
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 30


def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d_in = X_bytes.shape
    types = card["input"]["types"]
    C = card["classes"]

    # Build expanded feature mapping
    exp_idx = []
    exp_type = []
    ordinal_params = []
    cat_maps = []

    d_exp = 0
    for j, t in enumerate(types):
        col = X_bytes[:, j]
        if t == "ordinal":
            mu = float(col.mean())
            sd = float(col.std() or 1.0)
            ordinal_params.append((mu, sd))
            exp_idx.append([d_exp])
            exp_type.append('ordinal')
            cat_maps.append(None)
            d_exp += 1
        else:
            vals = np.unique(col)
            mapping = {int(v): d_exp + i for i, v in enumerate(vals)}
            cat_maps.append(mapping)
            exp_idx.append(list(mapping.values()))
            exp_type.append('categorical')
            ordinal_params.append(None)
            d_exp += len(vals)

    # Build expanded feature matrix X (n, d_exp) as float32
    X = np.zeros((n, d_exp), dtype=np.float32)
    for j in range(d_in):
        col = X_bytes[:, j]
        if exp_type[j] == 'ordinal':
            mu, sd = ordinal_params[j]
            idx = exp_idx[j][0]
            X[:, idx] = (col.astype(np.float32) - mu) / sd
        else:
            mapping = cat_maps[j]
            for v, idx in mapping.items():
                X[col == v, idx] = 1.0

    # Initialize FM parameters
    rng = np.random.default_rng(SEED)
    w0 = np.float32(0.0)
    w = np.zeros(d_exp, dtype=np.float32)
    V = rng.normal(0, 0.01, (d_exp, K)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
