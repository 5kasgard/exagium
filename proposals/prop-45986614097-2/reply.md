```hypothesis
Family: label-emblinear (positional embeddings + linear output).
Why: The linear model uses 16*256 one-hot features (4096 dim). Factorizing the weight matrix via low-rank embeddings (E=32..128) shares statistical strength across positions and bytes, reducing parameters and improving generalization on this small dataset.
Failure: If dev accuracy <= 0.3375 (label-linear) or inference cost exceeds label-linear by >2x.
```
```python
"""label-emblinear: positional embeddings + linear output for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Tunes embedding size E on last 20% of train, then trains on all data.
Weights format: magic 0x454D424C, int32 d, E, C; float32 emb[d][256][E], W[E][C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS_TUNE = 20
MAX_EPOCHS_FULL = 80
E_CANDIDATES = [16, 32, 64, 128]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    # logits: (B, C), y: (B,)
    B = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B  # (B, C)

def evaluate_acc(emb, W, b, X, y):
    n = X.shape[0]
    correct = 0
    for s in range(0, n, 1024):
        idx = slice(s, min(s+1024, n))
        xb = X[idx]
        yb = y[idx]
        h = np.zeros((xb.shape[0], emb.shape[2]), dtype=np.float32)
        for j in range(xb.shape[1]):
            h += emb[j, xb[:, j]]
        logits = h @ W + b
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n

def train_loop(emb, W, b, X, y, rng, epochs, batch_size):
    n, d = X.shape
    C = b.shape[0]
    E = emb.shape[2]
    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s+batch_size]
            xb = X[idx]
            yb = y[idx]
            B = xb.shape[0]
            # forward
            h = np.zeros((B, E), dtype=np.float32)
            for j in range(d):
                h += emb[j, xb[:, j]]
            logits = h @ W + b
            # backward
            g = softmax_cross_entropy_grad(logits, yb)  # (B, C)
            # gradients
            gW = h.T @ g + L2 * W
            gb = g.sum(axis=0) + L2 * b
            gh = g @ W.T  # (B, E)
            # embedding gradients: for each position j, add gh to emb[j, xb[:, j]]
            gemb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(gemb[j], xb[:, j], gh)
            gemb += L2 * emb
            # Adam update
            t += 1
            for param, grad, m, v in [
                (emb, gemb, m_emb, v_emb),
                (W, gW, m_W, v_W),
                (b, gb, m_b, v_b)
            ]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(0.8 * n)
    idx_train = np.arange(split)
    idx_tune = np.arange(split, n)
    X_train, y_train = X[idx_train], y[idx_train]
    X_tune, y_tune = X[idx_tune], y[idx_tune]

    rng = np.random.default_rng(SEED)
    best_E = E_CANDIDATES[0]
    best_acc = -1.0

    for E in E_CANDIDATES:
        # initialize
        emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
        W = (rng.normal(0, math.sqrt(6.0 / (E + C)), (E, C))).astype(np.float32)
        b = np.zeros(C, dtype=np.float32)
        # train on 80%
        train_loop(emb, W, b, X_train, y_train, rng, MAX_EPOCHS_TUNE, BATCH)
        # evaluate on 20%
        acc = evaluate_acc(emb, W, b, X_tune, y_tune)
        if acc > best_acc:
            best_acc = acc
            best_E = E
            # keep best model? we'll retrain anyway

    # retrain on full data with best_E
    E = best_E
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    W = (rng.normal(0, math.sqrt(6.0 / (E + C)), (E, C))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    train_loop(emb, W, b, X, y, rng, MAX_EPOCHS_FULL, BATCH)

    # save weights
    MAGIC = 0x454D424C  # "LBME"
    out = struct.pack("<4i", MAGIC, d, E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-emblinear inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte class index.
 * Weights format: magic 0x454D424C, int32 d, E, C; float32 emb[d][256][E], W[E][C], b[C]. */
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
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = (float *)malloc(sizeof(float) * E);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, sizeof(float) * E);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int u = 0; u < E; u++) h[u] += e[u];
        }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int u = 0; u < E; u++) sum += h[u] * W[(long)u * C + c];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```