```hypothesis
We propose a new family "label-mlp-emb" that uses per-position byte embeddings (dimension D) followed by a one-hidden-layer ReLU MLP (size H). This reduces parameters vs. one-hot linear models while capturing nonlinear interactions, and should beat label-linear's 0.3375 dev accuracy with similar inference cost (≈10k Ir/item). If dev accuracy ≤ 0.3375 or inference cost > 2× label-linear, the hypothesis fails.
```

```python
"""label-mlp-emb: per-position embeddings + one-hidden-layer MLP for next-char prediction.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid to search on the tuning split (last 20% of train.in)
GRID = [
    {"D": 32, "H": 64, "lr": 1e-3, "batch": 64, "epochs": 20, "l2": 1e-4},
    {"D": 16, "H": 32, "lr": 3e-3, "batch": 64, "epochs": 30, "l2": 1e-4},
    {"D": 64, "H": 128, "lr": 5e-4, "batch": 64, "epochs": 15, "l2": 1e-4},
]
SEED = 1
FINAL_EPOCH_MULT = 1.5  # retrain longer on full data


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return np.mean(np.argmax(logits, axis=1) == y)


def forward(X, params, d, D, H, C):
    """X: (n, d) uint8. Returns logits (n, C)."""
    n = X.shape[0]
    # Embedding sum
    emb_sum = np.zeros((n, D), dtype=np.float32)
    for j in range(d):
        E = params["E"][j]  # (256, D)
        emb_sum += E[X[:, j]]
    # Hidden layer
    hidden = np.maximum(emb_sum @ params["W1"] + params["b1"], 0, dtype=np.float32)
    # Output
    logits = hidden @ params["W2"] + params["b2"]
    return logits, hidden, emb_sum


def backward(X, y, hidden, emb_sum, params, d, D, H, C, l2):
    n = X.shape[0]
    # Softmax gradient
    logits = hidden @ params["W2"] + params["b2"]
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n  # dL/dlogits

    # Output layer gradients
    dW2 = hidden.T @ probs + l2 * params["W2"]
    db2 = probs.sum(axis=0)

    # Hidden layer gradients
    dhidden = probs @ params["W2"].T
    dhidden[hidden <= 0] = 0

    dW1 = emb_sum.T @ dhidden + l2 * params["W1"]
    db1 = dhidden.sum(axis=0)

    # Embedding gradients
    demb_sum = dhidden @ params["W1"].T  # (n, D)
    dE = [np.zeros((256, D), dtype=np.float32) for _ in range(d)]
    for j in range(d):
        np.add.at(dE[j], X[:, j], demb_sum)

    return {"E": dE, "W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def init_params(d, D, H, C, rng):
    params = {}
    params["E"] = [rng.normal(0, 0.01, (256, D)).astype(np.float32) for _ in range(d)]
    params["W1"] = rng.normal(0, math.sqrt(2.0 / D), (D, H)).astype(np.float32)
    params["b1"] = np.zeros(H, dtype=np.float32)
    params["W2"] = rng.normal(0, math.sqrt(2.0 / H), (H, C)).astype(np.float32)
    params["b2"] = np.zeros(C, dtype=np.float32)
    return params


def adam_step(params, grads, state, t, lr):
    for key in params:
        if key == "E":
            for j in range(len(params["E"])):
                g = grads["E"][j]
                m, v = state["E"][j]
                m *= 0.9; m += 0.1 * g
                v *= 0.999; v += 0.001 * g * g
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                params["E"][j] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        else:
            g = grads[key]
            m, v = state[key]
            m *= 0.9; m += 0.1 * g
            v *= 0.999; v += 0.001 * g * g
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            params[key] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def train_loop(X, y, params, d, D, H, C, cfg, rng, epochs):
    n = X.shape[0]
    batch = cfg["batch"]
    lr = cfg["lr"]
    l2 = cfg["l2"]
    # Adam state
    state = {}
    state["E"] = [(np.zeros((256, D), np.float32), np.zeros((256, D), np.float32)) for _ in range(d)]
    state["W1"] = (np.zeros((D, H), np.float32), np.zeros((D, H), np.float32))
    state["b1"] = (np.zeros(H, np.float32), np.zeros(H, np.float32))
    state["W2"] = (np.zeros((H, C), np.float32), np.zeros((H, C), np.float32))
    state["b2"] = (np.zeros(C, np.float32), np.zeros(C, np.float32))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            Xb, yb = X[idx], y[idx]
            logits, hidden, emb_sum = forward(Xb, params, d, D, H, C)
            grads = backward(Xb, yb, hidden, emb_sum, params, d, D, H, C, l2)
            t += 1
            adam_step(params, grads, state, t, lr)


def evaluate(X, y, params, d, D, H, C):
    logits, _, _ = forward(X, params, d, D, H, C)
    return accuracy(logits, y)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    best_cfg = None
    best_acc = -1.0
    best_params = None

    for cfg in GRID:
        D, H = cfg["D"], cfg["H"]
        rng = np.random.default_rng(SEED)
        params = init_params(d, D, H, C, rng)
        train_loop(X_train, y_train, params, d, D, H, C, cfg, rng, cfg["epochs"])
        acc = evaluate(X_val, y_val, params, d, D, H, C)
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_params = {k: (v.copy() if k != "E" else [e.copy() for e in v]) for k, v in params.items()}

    print(f"Best config: {best_cfg}, val acc: {best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    D, H = best_cfg["D"], best_cfg["H"]
    rng = np.random.default_rng(SEED)
    params = init_params(d, D, H, C, rng)
    final_epochs = int(best_cfg["epochs"] * FINAL_EPOCH_MULT)
    train_loop(X, y, params, d, D, H, C, best_cfg, rng, final_epochs)

    # Pack weights: magic, d, D, H, C, then E[d][256][D], W1[D][H], b1[H], W2[H][C], b2[C]
    magic = 0x4D4C5045  # "MLPE"
    header = struct.pack("<5i", magic, d, D, H, C)
    # Embeddings
    emb_bytes = b"".join(params["E"][j].astype("<f4").tobytes() for j in range(d))
    W1_bytes = params["W1"].astype("<f4").tobytes()
    b1_bytes = params["b1"].astype("<f4").tobytes()
    W2_bytes = params["W2"].astype("<f4").tobytes()
    b2_bytes = params["b2"].astype("<f4").tobytes()
    out = header + emb_bytes + W1_bytes + b1_bytes + W2_bytes + b2_bytes
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-mlp-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index. */
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
    return b;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = buf[*pos] | buf[*pos + 1] << 8 | buf[*pos + 2] << 16 | (uint32_t)buf[*pos + 3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x4D4C5045) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20
        + 4L * d * 256 * D
        + 4L * D * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    const float *E = (const float *)(hd + 5);
    const float *W1 = E + (long)d * 256 * D;
    const float *b1 = W1 + (long)D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = calloc(D, sizeof(float));
    float *hidden = malloc(H * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(emb_sum, 0, D * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *row = E + ((long)j * 256 + xi[j]) * D;
            for (int v = 0; v < D; v++) emb_sum[v] += row[v];
        }
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int v = 0; v < D; v++) sum += emb_sum[v] * W1[(long)v * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```