```hypothesis
Family: label-emb-mlp (per-position byte embeddings + one ReLU hidden layer).
Why: The linear one-hot model (label-linear) ignores semantic similarity between bytes (e.g., 'a' vs 'b'). Learned embeddings capture such similarities, and a hidden layer models interactions between positions. This should improve accuracy over label-linear (0.3375) while keeping inference cost moderate (embeddings sum + small MLP). Failure: if dev accuracy ≤ 0.3375 or inference cost exceeds label-linear by >2×.
```

```python
"""label-emb-mlp: per-position embeddings + one hidden ReLU layer.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only."""
import json, math, struct, sys, os

import numpy as np

# Fixed hyperparameter grid (tuned on last 20% of train.in)
GRID = [
    # (E, H, lr, epochs, batch, L2)
    (16, 0, 3e-3, 25, 64, 1e-4),   # linear on embeddings
    (16, 32, 3e-3, 25, 64, 1e-4),
    (16, 64, 3e-3, 25, 64, 1e-4),
    (32, 0, 3e-3, 25, 64, 1e-4),
    (32, 32, 3e-3, 25, 64, 1e-4),
    (32, 64, 3e-3, 25, 64, 1e-4),
    (64, 0, 3e-3, 25, 64, 1e-4),
    (64, 32, 3e-3, 25, 64, 1e-4),
    (64, 64, 3e-3, 25, 64, 1e-4),
]
SEED = 1
MAGIC = 0x4C424D45  # "EMBL" little-endian

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def split_last_20(X, y):
    n = X.shape[0]
    cut = n - n // 5
    return X[:cut], y[:cut], X[cut:], y[cut:]

def init_params(d, E, H, C, rng):
    # embeddings: list of d arrays (256, E)
    emb = [(rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)
           for _ in range(d)]
    if H:
        W1 = (rng.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / (E + H))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / (H + C))).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        return emb, W1, b1, W2, b2
    else:
        W2 = (rng.uniform(-1, 1, (E, C)) * math.sqrt(6.0 / (E + C))).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        return emb, None, None, W2, b2

def forward(X, emb, W1, b1, W2, b2, H):
    # X: (n, d) uint8
    n, d = X.shape
    # sum embeddings
    sums = np.zeros((n, emb[0].shape[1]), np.float32)
    for j in range(d):
        sums += emb[j][X[:, j]]
    if H:
        h = np.maximum(sums @ W1 + b1, 0)
        logits = h @ W2 + b2
    else:
        logits = sums @ W2 + b2
    return logits, sums if H else None

def softmax_cross_entropy_grad(logits, y):
    # logits: (n, C), y: (n,)
    n = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return probs / n

def backward(X, emb, W1, b1, W2, b2, H, logits, sums, y, grad_logits):
    n, d = X.shape
    E = emb[0].shape[1]
    # grad for output layer
    if H:
        grad_W2 = sums.T @ grad_logits  # (H, C)
        grad_b2 = grad_logits.sum(0)
        grad_h = grad_logits @ W2.T  # (n, H)
        grad_h[sums @ W1 + b1 <= 0] = 0  # ReLU grad
        grad_W1 = sums.T @ grad_h  # (E, H)
        grad_b1 = grad_h.sum(0)
        grad_sums = grad_h @ W1.T  # (n, E)
    else:
        grad_W2 = sums.T @ grad_logits  # (E, C)
        grad_b2 = grad_logits.sum(0)
        grad_sums = grad_logits @ W2.T  # (n, E)
    # grad for embeddings
    grad_emb = [np.zeros((256, E), np.float32) for _ in range(d)]
    for j in range(d):
        np.add.at(grad_emb[j], X[:, j], grad_sums)
    if H:
        return grad_emb, grad_W1, grad_b1, grad_W2, grad_b2
    else:
        return grad_emb, None, None, grad_W2, grad_b2

def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()

def train_config(Xtr, ytr, Xval, yval, d, C, E, H, lr, epochs, batch, L2, rng):
    n = Xtr.shape[0]
    emb, W1, b1, W2, b2 = init_params(d, E, H, C, rng)
    params = [emb, W1, b1, W2, b2] if H else [emb, W2, b2]
    # Adam state
    state = []
    for p in params:
        if isinstance(p, list):
            state.append([(np.zeros_like(e), np.zeros_like(e)) for e in p])
        elif p is not None:
            state.append((np.zeros_like(p), np.zeros_like(p)))
        else:
            state.append(None)
    t = 0
    best_val = -1
    best_params = None
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            xb, yb = Xtr[idx], ytr[idx]
            logits, sums = forward(xb, emb, W1, b1, W2, b2, H)
            grad_logits = softmax_cross_entropy_grad(logits, yb)
            grads = backward(xb, emb, W1, b1, W2, b2, H, logits, sums, yb, grad_logits)
            t += 1
            # Adam update
            for i, (p, gr) in enumerate(zip(params, grads)):
                if isinstance(p, list):
                    for j, (pj, grj) in enumerate(zip(p, gr)):
                        m, v = state[i][j]
                        m *= 0.9; m += 0.1 * grj; v *= 0.999; v += 0.001 * grj * grj
                        m_hat = m / (1 - 0.9 ** t)
                        v_hat = v / (1 - 0.999 ** t)
                        pj -= lr * (m_hat / (np.sqrt(v_hat) + 1e-8) + L2 * pj)
                elif p is not None:
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= lr * (m_hat / (np.sqrt(v_hat) + 1e-8) + L2 * p)
        # validation
        val_logits, _ = forward(Xval, emb, W1, b1, W2, b2, H)
        val_acc = accuracy(val_logits, yval)
        if val_acc > best_val:
            best_val = val_acc
            # deep copy
            best_params = ([e.copy() for e in emb],
                           W1.copy() if W1 is not None else None,
                           b1.copy() if b1 is not None else None,
                           W2.copy(), b2.copy())
    return best_val, best_params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    # split
    Xtr, ytr, Xval, yval = split_last_20(X, y)
    # grid search
    rng = np.random.default_rng(SEED)
    best_overall = -1
    best_cfg = None
    best_params = None
    for idx, (E, H, lr, epochs, batch, L2) in enumerate(GRID):
        cfg_rng = np.random.default_rng(SEED + 1000 + idx)
        val_acc, params = train_config(Xtr, ytr, Xval, yval, d, C, E, H, lr, epochs, batch, L2, cfg_rng)
        if val_acc > best_overall:
            best_overall = val_acc
            best_cfg = (E, H, lr, epochs, batch, L2)
            best_params = params
    # retrain on full data with best config
    E, H, lr, epochs, batch, L2 = best_cfg
    final_rng = np.random.default_rng(SEED)
    _, final_params = train_config(X, y, X[:1], y[:1], d, C, E, H, lr, epochs, batch, L2, final_rng)
    emb, W1, b1, W2, b2 = final_params
    # pack weights
    # header: magic, d, E, H, C
    header = struct.pack("<5i", MAGIC, d, E, H, C)
    # embeddings: d * 256 * E floats
    emb_bytes = b"".join(e.astype("<f4").tobytes() for e in emb)
    if H:
        rest = (W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes() +
                W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes())
    else:
        rest = W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + rest)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    if (nw < 20 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H < 0 || C <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    // expected size
    long expected = 20 + 4L * d * 256 * E;
    if (H) expected += 4L * (E * H + H + H * C + C);
    else expected += 4L * (E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    if (!H) {
        W2 = W1; // W1/b1 not used
        b2 = b1;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sums = malloc(sizeof(float) * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // sum embeddings
        for (int e = 0; e < E; e++) sums[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) sums[e] += eptr[e];
        }
        if (H) {
            // hidden layer: ReLU(sums @ W1 + b1)
            for (int h = 0; h < H; h++) {
                float v = b1[h];
                for (int e = 0; e < E; e++) v += sums[e] * W1[(long)e * H + h];
                hidden[h] = v > 0.0f ? v : 0.0f;
            }
            // output layer
            for (int c = 0; c < C; c++) {
                float v = b2[c];
                for (int h = 0; h < H; h++) v += hidden[h] * W2[(long)h * C + c];
                logits[c] = v;
            }
        } else {
            // linear on embeddings
            for (int c = 0; c < C; c++) {
                float v = b2[c];
                for (int e = 0; e < E; e++) v += sums[e] * W2[(long)e * C + c];
                logits[c] = v;
            }
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```