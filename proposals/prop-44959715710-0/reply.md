```hypothesis
We propose "label-emb", an embedding-based linear model. Each of the 14 byte features is mapped to a learned dense vector of size E (tuned on the last 20% of train.in), summed across features, then passed to a linear output layer. This captures non-linear feature effects and category similarities better than one-hot linear, while being faster than an MLP due to the low-dimensional bottleneck. If dev accuracy does not exceed 0.8524 or cost exceeds label-linear, the hypothesis fails.
```

```python
"""label-emb: embedding-based linear model for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Weights format: magic 0x454D424C, d, E, C; float32 embeddings[d][256][E], W_out[E][C], b_out[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
# Candidate embedding sizes to try on the tuning split
E_CANDIDATES = [4, 8, 12, 16]
TUNE_EPOCHS = 20  # quick evaluation epochs for tuning


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, d, C, E, epochs, rng, lr=LR, l2=L2, batch=BATCH):
    # embeddings: d x 256 x E
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 * E))).astype(np.float32)
    W_out = (rng.uniform(-1, 1, (E, C)) * math.sqrt(6.0 / E)).astype(np.float32)
    b_out = np.zeros(C, np.float32)
    params = [emb, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            # forward
            acc = np.zeros((len(idx), E), np.float32)
            for j in range(d):
                acc += emb[j, xb[:, j]]
            logits = acc @ W_out + b_out
            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            probs /= len(yb)
            # gradients
            g_W_out = acc.T @ probs + l2 * W_out
            g_b_out = probs.sum(0)
            g_acc = probs @ W_out.T
            g_emb = np.zeros_like(emb)
            for j in range(d):
                # accumulate gradient for each feature value
                np.add.at(g_emb[j], xb[:, j], g_acc)
            g_emb += l2 * emb
            grads = [g_emb, g_W_out, g_b_out]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return emb, W_out, b_out


def evaluate(X, y, emb, W_out, b_out):
    n = X.shape[0]
    acc = np.zeros((n, emb.shape[2]), np.float32)
    for j in range(emb.shape[0]):
        acc += emb[j, X[:, j]]
    logits = acc @ W_out + b_out
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    split = int(0.8 * n)  # last 20% for tuning
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    # Tune embedding size E
    best_E, best_acc = E_CANDIDATES[0], -1.0
    for E in E_CANDIDATES:
        rng = np.random.default_rng(SEED + E)
        emb, W_out, b_out = train_model(X_train, y_train, d, C, E, TUNE_EPOCHS, rng)
        acc = evaluate(X_tune, y_tune, emb, W_out, b_out)
        if acc > best_acc:
            best_acc, best_E = acc, E

    # Retrain on all data with best_E
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng = np.random.default_rng(SEED)
    emb, W_out, b_out = train_model(X, y, d, C, best_E, epochs, rng)

    # Write weights
    magic = 0x454D424C  # "EMBL"
    out = struct.pack("<4i", magic, d, best_E, C)
    out += emb.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from weights). Output: one byte class index.
 * Weights format: magic 0x454D424C, int32 d, E, C; float32 emb[d][256][E], W_out[E][C], b_out[C]. */
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
    if (C < 1 || C > 256 || E < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W_out = emb + (long)d * 256 * E;
    const float *b_out = W_out + (long)E * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * E);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(acc, 0, sizeof(float) * E);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int e_idx = 0; e_idx < E; e_idx++) acc[e_idx] += e[e_idx];
        }
        for (int c = 0; c < C; c++) scores[c] = b_out[c];
        for (int e_idx = 0; e_idx < E; e_idx++) {
            float a = acc[e_idx];
            for (int c = 0; c < C; c++) scores[c] += a * W_out[(long)e_idx * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```