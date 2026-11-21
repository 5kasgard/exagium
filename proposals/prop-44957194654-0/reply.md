```hypothesis
Family: label-fm (Factorization Machine on expanded features).
Why: Adult has strong feature interactions (education×occupation, marital×sex) that linear models miss. FM captures all pairwise interactions with only O(k·D) params (k=8, D≈100) and O(k·D) inference, far cheaper than GBDT/MLP.
Failure: If dev accuracy ≤0.8524 (label-linear) or Ir/item >2000.
```
```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Encodes ordinal as z-scored, categorical as one-hot.
FM with k=8 factors, Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Exports: magic 0x464D3100, d, k, C, bias, T_linear[d][256], T_factor[d][256][k] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

K = 8
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    C = card["classes"]
    assert C == 2
    rng = np.random.default_rng(SEED)

    enc = []
    d_expanded = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None].astype(np.float32))
            d_expanded.append(1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
            d_expanded.append(len(vals))

    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, D = Z.shape

    w = np.zeros(D, np.float32)
    V = (rng.uniform(-1, 1, (D, K)) * math.sqrt(6.0 / D)).astype(np.float32)
    bias = np.float32(0.0)

    mw, mV, mb = np.zeros_like(w), np.zeros_like(V), np.float32(0.0)
    vw, vV, vb = np.zeros_like(w), np.zeros_like(V), np.float32(0.0)
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]
            yy = y[idx].astype(np.float32)
            B = len(yy)

            linear = z @ w
            S = z @ V
            Q = (z * z) @ (V * V)
            inter = 0.5 * np.sum(S * S - Q, axis=1)
            score = bias + linear + inter
            p = 1.0 / (1.0 + np.exp(-score))
            grad = p - yy

            dbias = grad.mean()
            dw = (z.T @ grad) / B + L2 * w
            A = z.T @ (grad[:, None] * S)
            B_vec = (z * z).T @ grad
            dV = (A - B_vec[:, None] * V) / B + L2 * V

            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            mb = beta1 * mb + (1 - beta1) * dbias
            vb = beta2 * vb + (1 - beta2) * dbias * dbias
            bias -= LR * (mb / (1 - beta1 ** t)) / (np.sqrt(vb / (1 - beta2 ** t)) + eps)

            mw = beta1 * mw + (1 - beta1) * dw
            vw = beta2 * vw + (1 - beta2) * dw * dw
            w -= LR * (mw / (1 - beta1 ** t)) / (np.sqrt(vw / (1 - beta2 ** t)) + eps)

            mV = beta1 * mV + (1 - beta1) * dV
            vV = beta2 * vV + (1 - beta2) * dV * dV
            V -= LR * (mV / (1 - beta1 ** t)) / (np.sqrt(vV / (1 - beta2 ** t)) + eps)

    T_linear = []
    T_factor = []
    off = 0
    for j, E in enumerate(enc):
        Dj = d_expanded[j]
        T_linear.append((E @ w[off:off + Dj]).astype(np.float32))
        T_factor.append((E @ V[off:off + Dj, :]).astype(np.float32))
        off += Dj

    out = struct.pack("<4i",