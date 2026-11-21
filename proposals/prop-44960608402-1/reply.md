```hypothesis
Family: label-fm (Factorization Machine with ordinal z-scored features).  
Why: Optdigits pixels have strong spatial correlations (strokes, loops) that a linear model misses. A degree-2 FM captures all pairwise interactions via k latent factors (k=8) with only O(d·k) extra parameters, and inference stays fast because per-byte contributions (linear + interaction terms) are precomputed into three lookup tables (T1, V1, V2).  
Failure signal: dev accuracy ≤ 0.9613 (label-linear) or Ir/item ≥ 10 000 (2× label-linear).
```
```python
import json, math, struct, sys

import numpy as np

K_CHOICES = [4, 8, 12, 16]
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, d, C, k, rng, n_epochs):
    mu = X.mean(axis=0).astype(np.float64)
    sigma = X.std(axis=0).astype(np.float64)
    sigma[sigma == 0] = 1.0
    E = ((np.arange(256, dtype=np.float64)[None, :] - mu[:, None]) / sigma[:, None]).astype(np.float32)

    scale_w = math.sqrt(2.0 / d)
    scale_v = math.sqrt(2.0 / d)
    scale_u = math.sqrt(2.0 / k)
    w = rng.normal(0, scale_w, (d, C)).astype(np.float32)
    v = rng.normal(0, scale_v, (d, k)).astype(np.float32)
    u = rng.normal(0, scale_u, (k, C)).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [w, v, u, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]

    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            B = len(yb)
            E_xb = E[np.arange(d), xb.T].T

            scores = E_xb @ w + b
            S = E_xb @ v
            S2 = (E_xb**2) @ (v**2)
            scores += (S**2 - S2) @ u

            max_s = scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grads_out = probs / B

            grad_w = E_xb.T @ grads_out + L2 * w
            grad_b = grads_out.sum(axis=0) + L2 * b
            grad_u = (S**2 - S2).T @ grads_out + L2 * u

            G = grads_out @ u.T
            term1 = E_xb.T @ (G * S)
            term2 = (E_xb**2).T @ G
            grad_v = 2.0 * (term1 - v * term2) + L2 * v

            grads = [grad_w, grad_v, grad_u, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v_state = state[i]
                m *= 0.9; m += 0.1 * gr
                v_state *= 0.999; v_state += 0.001 * gr * gr
                m_hat = m / (1 - 0.9**t)
                v_hat = v_state / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    return w, v, u, b, mu, sigma


def evaluate(X, y, w, v, u, b, mu, sigma):
    d = X.shape[1]
    E = ((np.arange(256, dtype=np.float64)[None, :] - mu[:, None]) / sigma[:, None]).astype(np.float32)
    E_xb = E[np.arange(d), X.T].T
    scores = E_xb @ w + b
    S = E_xb @ v
    S2 = (E_xb**2) @ (v**2)
    scores += (S**2 - S2) @ u
    return (scores.argmax(axis=1) == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
