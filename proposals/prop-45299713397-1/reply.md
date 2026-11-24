```hypothesis
label-mlp-dense: A 2-layer MLP (64→16→10) on densely encoded ordinal features (z-scored per pixel) instead of 256-way one-hot tables. Avoids sparse expansion, fits in cache, learns nonlinearities. Should beat label-linear (0.9613) at lower CPU cost. Fails if dev accuracy ≤ 0.9613 or Ir/item ≥ 5000.
```
```python
"""label-mlp-dense: Dense MLP on z-scored ordinal / one-hot categorical bytes.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

H = 16
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    d_orig = card["input"]["bytes"]
    n = X_bytes.shape[0]

    # Encode features in original order
    feat_info = []  # (type, params) type 0=ordinal, 1=categorical
    Z_parts = []
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X_bytes[:, j].astype(np.float32)
            mu = float(v.mean())
            sd = float(v.std())
            if sd == 0.0:
                sd = 1.0
            feat_info.append((0, (mu, sd)))
            Z_parts.append(((v - mu) / sd)[:, None])
        else:
            vals = np.unique(X_bytes[:, j])
            mapping = np.full(256, 255, dtype=np.uint8)
            mapping[vals] = np.arange(len(vals), dtype=np.uint8)
            dim = len(vals)
            feat_info.append((1, (mapping, dim)))
            oh = np.zeros((n, dim), dtype=np.float32)
            oh[np.arange(n), mapping[X_bytes[:, j]]] = 1.0
            Z_parts.append(oh)

    Z = np.concatenate(Z_parts, axis=1) if Z_parts else np.zeros((n, 0), dtype=np.float32)
    d_eff = Z.shape[1]

    rng = np.random.default_rng(SEED)
    # Xavier init for ReLU
    W1 = (rng.uniform(-1, 1, (d_eff, H)) * math.sqrt(6.0 / d_eff)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # Forward
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            # Softmax gradient
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            # Backward
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(0)
            gh = (g @ W2.T) * (a > 0)
            grad_W1 = z.T @ gh + L2 * W1
            grad_b1 = gh.sum(0)
            grads = [grad_W1, grad_b1, grad_W2, grad_b2]
            # Adam
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                m_hat = m / (1 - 0