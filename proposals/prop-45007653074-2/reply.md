```hypothesis
Factorized linear model (per-position character embeddings summed then linear classifier) captures character similarities and positional effects with far fewer parameters than full one-hot linear. It should exceed label-linear accuracy (33.75%) while keeping inference cost low (O(d*D + D*C) per item). If dev accuracy ≤ 0.34 or inference cost > 2× label-linear, the approach fails.
```

```python
"""Factorized linear model for next-byte prediction: per-position embeddings summed then linear.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only. Chooses embedding dim D on tuning split (last 20% of train)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH = 1e-2, 1e-4, 64
SAMPLES, MAX_EPOCHS = 600000, 100
CANDIDATE_D = [16, 32, 64]  # embedding dimensions to try on tuning split
TUNE_EPOCHS = 10


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(d, D, C, rng):
    """Return list of parameter arrays: embeddings[d][256][D], V[D][C], b[C]."""
    embeddings = [rng.normal(0, 1.0 / math.sqrt(256), size=(256, D)).astype(np.float32) for _ in range(d)]
    V = rng.normal(0, 1.0 / math.sqrt(D), size=(D, C)).astype(np.float32)
    b = np.zeros(C, np.float32)
    return embeddings + [V, b]


def adam_update(params, grads, state, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    for i, (p, g) in enumerate(zip(params, grads)):
        m, v = state[i]
        m *= beta1; m += (1 - beta1) * g
        v *= beta2; v += (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train_epoch(X, y, params, state, t, rng, l2):
    """One epoch of minibatch SGD. Returns updated t."""
    n = X.shape[0]
    d = X.shape[1]
    C = params[-1].shape[0]
    D = params[0].shape[1]
    embeddings = params[:-2]
    V = params[-2]
    b = params[-1]
    order = rng.permutation(n)
    for s in range(0, n, BATCH):
        idx = order[s:s + BATCH]
        xb = X[idx]          # (B, d)
        yb = y[idx]          # (B,)
        B = len(idx)
        # forward: accumulate embeddings
        h = np.zeros((B, D), np.float32)
        for p in range(d):
            h += embeddings[p][xb[:, p]]
        logits = h @ V + b    # (B, C)
        # softmax + cross-entropy gradient
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        grad_logits = probs
        grad_logits[np.arange(B), yb] -= 1.0
        grad_logits /= B
        # gradients
        grad_V = h.T @ grad_logits + l2 * V
        grad_b = grad_logits.sum(axis=0)
        grad_h = grad_logits @ V.T          # (B, D)
        grad_embeddings = []
        for p in range(d):
            g = np.zeros((256, D), np.float32)
            np.add.at(g, xb[:, p], grad_h)
            g += l2 * embeddings[p]
            grad_embeddings.append(g)
        grads = grad_embeddings + [grad_V, grad_b]
        t += 1
        adam_update(params, grads, state, t, LR)
    return t


def evaluate(X, y, params):
    d = X.shape[1]
    D = params[0].shape[1]
    C = params[-1].shape[0]
    embeddings = params[:-2]
    V = params[-2]
    b = params[-1]
    h = np.zeros((X.shape[0], D), np.float32)
    for p in range(d):
        h += embeddings[p][X[:, p]]
    logits = h @ V + b
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # tuning split: last 20% in file order
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    rng = np.random.default_rng(SEED)
    best_D, best_acc = CANDIDATE_D[0], -1.0

    for D in CANDIDATE_D:
        params = init_params(d, D, C, rng)
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0
        for _ in range(TUNE_EPOCHS):
            t = train_epoch(X_train, y_train, params, state, t, rng, L2)
        acc = evaluate(X_tune, y_tune, params)
        if acc > best_acc:
            best_acc, best_D = acc, D

    # final training on all data with best_D
    params = init_params(d, best_D, C, rng)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    max_epochs = min(MAX_EPOCHS, max(1, math.ceil(SAMPLES / n)))
    for _ in range(max_epochs):
        t = train_epoch(X, y, params, state, t, rng, L2)

    # pack weights: magic, d, D, C; embeddings[d][256][D]; V[D][C]; b[C]
    embeddings = params[:-2]
    V = params[-2]
    b = params[-1]
    magic = 0x464C4531  # "FLE1"
    header = struct.pack("<4i", magic, d, best_D, C)
    emb_bytes = b"".join(e.astype("<f4").tobytes() for e in embeddings)
    out = header + emb_bytes + V.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorized linear inference: ./infer weights.bin in.rec N
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
    if (nw < 16 || hd[0] != 0x464C4531) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || D < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + (long)d * 256 * D * 4 + (long)D * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *V = emb + (long)d * 256 * D;
    const float *b = V + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < D; j++) acc[j] = 0.0f;
        for (int p = 0; p < d; p++) {
            const float *e = emb + ((long)p * 256 + xi[p]) * D;
            for (int j = 0; j < D; j++) acc[j] += e[j];
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < D; j++) {
            float a = acc[j];
            for (int c = 0; c < C; c++) scores[c] += a * V[(long)j * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(acc); free(scores);
    return 0;
}
```