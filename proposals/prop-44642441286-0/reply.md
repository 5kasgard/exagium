```hypothesis
Propose label-emblin: a linear model on learned per-feature embeddings (dimension D tuned on the last 20% of train). Each of the 14 bytes (ordinal or categorical) maps to a D-vector; concatenated embeddings feed a linear output layer. This captures category similarities that one-hot linear misses, while keeping inference a single table lookup and sum per feature (like label-linear). If dev accuracy ≤ 0.8524 or instructions/item ≥ label-linear, the hypothesis fails.
```

```python
"""label-emblin: linear model on learned embeddings for tabular byte features.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Hyperparameter D (embedding dim) chosen on last 20% of train, then retrained on all."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
D_CANDIDATES = [4, 8, 16]  # embedding dimensions to try


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_eval(X_tr, y_tr, X_va, y_va, d, C, D, rng):
    """Train on (X_tr,y_tr), return validation accuracy."""
    n_tr = X_tr.shape[0]
    # parameters: embeddings[d][256, D], W_out[d*D, C], b_out[C]
    emb = [rng.normal(0, 0.01, (256, D)).astype(np.float32) for _ in range(d)]
    W_out = rng.normal(0, 0.01, (d * D, C)).astype(np.float32)
    b_out = np.zeros(C, np.float32)
    params = emb + [W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X):
        # X: (n, d) uint8
        n = X.shape[0]
        # concatenate embeddings
        emb_cat = np.empty((n, d * D), np.float32)
        for j in range(d):
            emb_cat[:, j * D:(j + 1) * D] = emb[j][X[:, j]]
        return emb_cat @ W_out + b_out  # (n, C)

    def softmax_cross_entropy_grad(logits, y):
        # logits (n,C), y (n,)
        n = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits, dtype=np.float32)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(n), y] -= 1.0
        return probs / n  # (n,C)

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
    for _ in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            # forward
            n = Xb.shape[0]
            emb_cat = np.empty((n, d * D), np.float32)
            for j in range(d):
                emb_cat[:, j * D:(j + 1) * D] = emb[j][Xb[:, j]]
            logits = emb_cat @ W_out + b_out
            # gradient
            g = softmax_cross_entropy_grad(logits, yb)  # (n,C)
            # grad W_out, b_out
            grad_W = emb_cat.T @ g + L2 * W_out
            grad_b = g.sum(axis=0)
            # grad embeddings
            g_emb_cat = g @ W_out.T  # (n, d*D)
            grad_emb = [np.zeros((256, D), np.float32) for _ in range(d)]
            for j in range(d):
                vals = Xb[:, j]
                np.add.at(grad_emb[j], vals, g_emb_cat[:, j * D:(j + 1) * D])
                grad_emb[j] += L2 * emb[j]
            # Adam update
            grads = grad_emb + [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # validation accuracy
    logits_va = forward(X_va)
    pred = logits_va.argmax(axis=1)
    return (pred == y_va).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    best_D, best_acc = D_CANDIDATES[0], -1.0
    for D in D_CANDIDATES:
        rng = np.random.default_rng(SEED)
        acc = train_eval(X_tr, y_tr, X_va, y_va, d, C, D, rng)
        if acc > best_acc:
            best_acc, best_D = acc, D
    # retrain on all data with best_D
    rng = np.random.default_rng(SEED)
    n_all = X.shape[0]
    emb = [rng.normal(0, 0.01, (256, best_D)).astype(np.float32) for _ in range(d)]
    W_out = rng.normal(0, 0.01, (d * best_D, C)).astype(np.float32)
    b_out = np.zeros(C, np.float32)
    params = emb + [W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_cross_entropy_grad(logits, y):
        n = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits, dtype=np.float32)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(n), y] -= 1.0
        return probs / n

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_all))
    for _ in range(epochs):
        order = rng.permutation(n_all)
        for s in range(0, n_all, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            n = Xb.shape[0]
            emb_cat = np.empty((n, d * best_D), np.float32)
            for j in range(d):
                emb_cat[:, j * best_D:(j + 1) * best_D] = emb[j][Xb[:, j]]
            logits = emb_cat @ W_out + b_out
            g = softmax_cross_entropy_grad(logits, yb)
            grad_W = emb_cat.T @ g + L2 * W_out
            grad_b = g.sum(axis=0)
            g_emb_cat = g @ W_out.T
            grad_emb = [np.zeros((256, best_D), np.float32) for _ in range(d)]
            for j in range(d):
                vals = Xb[:, j]
                np.add.at(grad_emb[j], vals, g_emb_cat[:, j * best_D:(j + 1) * best_D])
                grad_emb[j] += L2 * emb[j]
            grads = grad_emb + [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Precompute T[j][v] = emb[j][v] @ W_out[j] for inference
    # T shape: d x 256 x C
    T = np.empty((d, 256, C), np.float32)
    for j in range(d):
        Wj = W_out[j * best_D:(j + 1) * best_D]  # (D, C)
        T[j] = emb[j] @ Wj  # (256, C)

    # Write weights: magic, d, D, C, T (d*256*C float32), b_out (C float32)
    magic = 0x454D424C  # 'EMBL'
    out = struct.pack("<4i", magic, d, best_D, C)
    out += T.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emblin inference: ./infer weights.bin in.rec N
 * Weights: magic(0x454D424C), d, D, C; float32 T[d][256][C]; float32 b_out[C].
 * Input record: d bytes. Output: one byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 16 + 4L * d * 256 * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *b_out = T + (long)d * 256 * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = b_out[c];
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += tj[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```