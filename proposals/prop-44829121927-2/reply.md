```hypothesis
Family: label-embed. Learns per-position byte embeddings (dimension D) concatenated into a single vector, followed by a linear classifier. This captures character similarities and positional context more efficiently than one-hot linear models, improving generalization on the small tinyshakespeare dataset. If dev accuracy does not exceed label-linear's 0.3375, the dense embedding approach fails to beat the sparse baseline.
```
```python
"""label-embed: per-position byte embeddings + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Tunes embedding dimension D on last 20% of train.in, then trains on all.
Weights format: magic 0x454D4244, d, D, C; float32 E[d][256][D], W[d*D][C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
D_CANDIDATES = [8, 16, 32, 64]  # tuning grid


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_cross_entropy_grad(logits, y):
    # logits: (B, C), y: (B,) int
    B = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B  # (B, C)


def evaluate(E, W, b, X, y):
    n = X.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        Xb = X[s:s+BATCH]
        yb = y[s:s+BATCH]
        B = Xb.shape[0]
        emb = np.empty((B, d, D), dtype=np.float32)
        for j in range(d):
            emb[:, j] = E[j, Xb[:, j]]
        emb_flat = emb.reshape(B, -1)
        logits = emb_flat @ W + b
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def train_split(E, W, b, X, y, rng, steps):
    n = X.shape[0]
    params = [E, W, b]
    # Adam state: m, v for each param
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(steps):
        idx = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch_idx = idx[s:s+BATCH]
            Xb = X[batch_idx]
            yb = y[batch_idx]
            B = Xb.shape[0]

            # forward
            emb = np.empty((B, d, D), dtype=np.float32)
            for j in range(d):
                emb[:, j] = E[j, Xb[:, j]]
            emb_flat = emb.reshape(B, -1)  # (B, d*D)
            logits = emb_flat @ W + b      # (B, C)

            # backward
            dlogits = softmax_cross_entropy_grad(logits, yb)  # (B, C)

            # gradients
            dW = emb_flat.T @ dlogits + L2 * W
            db = dlogits.sum(axis=0)
            demb_flat = dlogits @ W.T  # (B, d*D)
            demb = demb_flat.reshape(B, d, D)

            # scatter-add to E gradients
            dE = np.zeros_like(E)
            for j in range(d):
                np.add.at(dE[j], Xb[:, j], demb[:, j])
            dE += L2 * E

            grads = [dE, dW, db]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)


def main():
    card = json.load(open(sys.argv[1]))
    global d, D, C
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # tuning split: last 20% in file order
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_D, best_acc = D_CANDIDATES[0], -1.0

    for D in D_CANDIDATES:
        # initialize
        E = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / 256)).astype(np.float32)
        W = (rng.uniform(-1, 1, (d * D, C)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
        b = np.zeros(C, dtype=np.float32)

        steps = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))
        train_split(E, W, b, X_tr, y_tr, rng, steps)
        acc = evaluate(E, W, b, X_val, y_val)
        if acc > best_acc:
            best_acc, best_D = acc, D

    # final training on all data with best_D
    D = best_D
    E = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W = (rng.uniform(-1, 1, (d * D, C)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    steps = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    train_split(E, W, b, X, y, rng, steps)

    # write weights
    magic = 0x454D4244  # 'EMBD'
    header = struct.pack("<4I", magic, d, D, C)
    out = header + E.astype("<f4").tobytes() + W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D < 1 || D > 256) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16L
        + (long)d * 256 * D * 4
        + (long)d * D * C * 4
        + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)d * 256 * D;
    const float *b = W + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb = malloc(sizeof(float) * d * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *src = E + ((long)j * 256 + xi[j]) * D;
            float *dst = emb + (long)j * D;
            memcpy(dst, src, D * sizeof(float));
        }

        // linear layer: scores = emb * W + b
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < d * D; f++) {
            float v = emb[f];
            if (v == 0.0f) continue;
            const float *wcol = W + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += v * wcol[c];
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```