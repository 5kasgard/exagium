```hypothesis
Family: label-elm (Extreme Learning Machine). A single hidden layer with fixed random weights (ReLU) and a trained linear output layer. The random projection non-linearly expands the 64-dimensional pixel space, enabling a linear classifier to capture more complex boundaries than plain logistic regression. Training is fast (only output layer) and deterministic via fixed seeds. Hyperparameter H (hidden units) is chosen on the last 20% of train.in.
Why it should beat label-linear: optdigits is known to benefit from non-linear features; random ReLU projections approximate an RBF kernel. The fixed hidden layer avoids overfitting on the small dataset (2867 samples) while adding capacity.
Failure condition: dev accuracy ≤ 0.9613 (label-linear) or inference cost > 8685 Ir/item (label-mlp16) without accuracy improvement.
```

```python
import json, math, struct, sys
import numpy as np

H_CANDIDATES = [8, 16, 32, 64, 128]
SEED_ELM = 42
SEED_TRAIN = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_ordinal(X, j):
    v = X[:, j].astype(np.float64)
    mu, sd = v.mean(), v.std() or 1.0
    return mu, sd


def apply_encoding(X, enc_params, types):
    n, d = X.shape
    Z = np.zeros((n, d), dtype=np.float32)
    for j, t in enumerate(types):
        if t == "ordinal":
            mu, sd = enc_params[j]
            Z[:, j] = (X[:, j].astype(np.float32) - mu) / sd
        else:
            # categorical not expected for this task
            pass
    return Z


def train_output(A, y, C, seed=SEED_TRAIN, epochs=MAX_EPOCHS, batch=BATCH, lr=LR, l2=L2, samples=SAMPLES):
    n, H = A.shape
    rng = np.random.default_rng(seed)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    max_epochs = min(epochs, math.ceil(samples / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            a, yy = A[idx], y[idx]
            scores = a @ W2 + b2
            scores = scores - scores.max(axis=1, keepdims=True)
            P = np.exp(scores)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grad_W2 = a.T @ g + l2 * W2
            grad_b2 = g.sum(0)
            grads = [grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W2, b2


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    types = card["input"]["types"]

    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_all = len(X_all)
    n_tune = n_all // 5
    n_train = n_all - n_tune
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_tune, y_tune = X_all[n_train:], y_all[n_train:]

    max_H = max(H_CANDIDATES)
    rng_elm = np.random.default_rng(SEED_ELM)
    W1_full = (rng_elm.uniform(-1, 1, (d, max_H)) * math.sqrt(6.0 / d)).astype(np.float32)
    b1_full = (rng_elm.uniform(-1, 1, max_H) * 0.1).astype(np.float32)

    best_H = H_CANDIDATES[0]
    best_acc = -1.0
    for H in H_CANDIDATES:
        enc_params = [encode_ordinal(X_train, j) for j in range(d)]
        Z_train = apply_encoding(X_train, enc_params, types)
        Z_tune = apply_encoding(X_tune, enc_params, types)
        W1 = W1_full[:, :H]
        b1 = b1_full[:H]
        A_train = np.maximum(Z_train @ W1 + b1, 0)
        A_tune = np.maximum(Z_tune @ W1 + b1, 0)
        W2, b2 = train_output(A_train, y_train, C)
        scores = A_tune @ W2 + b2
        pred = scores.argmax(axis=1)
        acc = (pred == y_tune).mean()
        if acc > best_acc:
            best_acc = acc
            best_H = H

    enc_params = [encode_ordinal(X_all, j) for j in range(d)]
    Z_all = apply_encoding(X_all, enc_params, types)
    W1 = W1_full[:, :best_H]
    b1 = b1_full[:best_H]
    A_all = np.maximum(Z_all @ W1 + b1, 0)
    W2, b2 = train_output(A_all, y_all, C)

    T = []
    for j in range(d):
        mu, sd = enc_params[j]
        encoded = (np.arange(256, dtype=np.float32) - mu) / sd
        Tj = encoded[:, None] * W1[j, :]
        T.append(Tj)
    T = np.stack(T)

    out = struct.pack("<4i", 0x54424C31, d, best_H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(3); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n ? *n : 1);
    if (fread(b, 1, *n, f) != (size_t)*n) { fprintf(stderr, "short read %s\n", p); exit(3); }
    fclose(f);
   