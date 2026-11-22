```hypothesis
We propose a shared embedding model with learned positional encodings (family "label-embed"). By sharing a single 256×D embedding matrix across all 16 positions and adding position-specific vectors, the model captures character-level statistics more efficiently than per-position one-hot encoding, reducing parameters and improving generalization. It should beat label-linear (0.3375) on dev accuracy with similar inference cost (O(16·D + D·C) vs O(16·C)). If dev accuracy ≤ 0.3375, the hypothesis fails.
```

```python
"""Shared embedding + positional encoding for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Architecture: embedding[256, D] + pos_emb[16, D] -> sum -> linear[D, C].
Hyperparameter D chosen on tuning split (last 20% of train.in), then retrain on all.
Exported as: int32 magic 0x44424D45, d, D, C; float32 emb[256*D], pos[16*D], W[D*C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, BETA1, BETA2, EPS = 0.01, 0.9, 0.999, 1e-8
BATCH = 64
EPOCHS_TUNE = 10
EPOCHS_FULL = 20
D_CANDIDATES = [32, 64, 128]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


def train_step(X, y, emb, pos, W, b, opt_state, rng):
    n = X.shape[0]
    order = rng.permutation(n)
    for s in range(0, n, BATCH):
        idx = order[s:s + BATCH]
        xb, yb = X[idx], y[idx]
        bsz = xb.shape[0]

        # forward
        hid = np.zeros((bsz, D), np.float32)
        for j in range(d):
            hid += emb[xb[:, j]] + pos[j]
        logits = hid @ W + b

        # softmax + cross-entropy gradient
        max_log = logits.max(axis=1, keepdims=True)
        exp_log = np.exp(logits - max_log)
        probs = exp_log / exp_log.sum(axis=1, keepdims=True)
        probs[np.arange(bsz), yb] -= 1.0
        probs /= bsz

        # gradients
        gW = hid.T @ probs
        gb = probs.sum(axis=0)
        ghid = probs @ W.T

        gemb = np.zeros_like(emb)
        gpos = np.zeros_like(pos)
        for j in range(d):
            np.add.at(gemb, xb[:, j], ghid)
            gpos[j] += ghid.sum(axis=0)

        # Adam update
        for param, grad, (m, v) in zip(
            [emb, pos, W, b], [gemb, gpos, gW, gb], opt_state
        ):
            m *= BETA1; m += (1 - BETA1) * grad
            v *= BETA2; v += (1 - BETA2) * grad * grad
            m_hat = m / (1 - BETA1 ** opt_state[0][2])
            v_hat = v / (1 - BETA2 ** opt_state[0][2])
            param -= LR * m_hat / (np.sqrt(v_hat) + EPS)
        opt_state[0] = (m, v, opt_state[0][2] + 1)


def evaluate(X, y, emb, pos, W, b):
    hid = np.zeros((X.shape[0], D), np.float32)
    for j in range(d):
        hid += emb[X[:, j]] + pos[j]
    logits = hid @ W + b
    return accuracy(logits, y)


def main():
    card = json.load(open(sys.argv[1]))
    global d, D, C
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_tu, y_tu = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_D, best_acc = D_CANDIDATES[0], -1.0

    for D in D_CANDIDATES:
        # initialize
        emb = rng.normal(0, 1/math.sqrt(D), (256, D)).astype(np.float32)
        pos = rng.normal(0, 1/math.sqrt(D), (d, D)).astype(np.float32)
        W = rng.normal(0, 1/math.sqrt(D), (D, C)).astype(np.float32)
        b = np.zeros(C, np.float32)
        opt_state = [
            (np.zeros_like(emb), np.zeros_like(emb), 1),
            (np.zeros_like(pos), np.zeros_like(pos), 1),
            (np.zeros_like(W), np.zeros_like(W), 1),
            (np.zeros_like(b), np.zeros_like(b), 1),
        ]

        for _ in range(EPOCHS_TUNE):
            train_step(X_tr, y_tr, emb, pos, W, b, opt_state, rng)

        acc = evaluate(X_tu, y_tu, emb, pos, W, b)
        if acc > best_acc:
            best_acc, best_D = acc, D

    # retrain on full data with best_D
    D = best_D
    rng = np.random.default_rng(SEED)
    emb = rng.normal(0, 1/math.sqrt(D), (256, D)).astype(np.float32)
    pos = rng.normal(0, 1/math.sqrt(D), (d, D)).astype(np.float32)
    W = rng.normal(0, 1/math.sqrt(D), (D, C)).astype(np.float32)
    b = np.zeros(C, np.float32)
    opt_state = [
        (np.zeros_like(emb), np.zeros_like(emb), 1),
        (np.zeros_like(pos), np.zeros_like(pos), 1),
        (np.zeros_like(W), np.zeros_like(W), 1),
        (np.zeros_like(b), np.zeros_like(b), 1),
    ]

    for _ in range(EPOCHS_FULL):
        train_step(X, y, emb, pos, W, b, opt_state, rng)

    # pack weights
    magic = 0x44424D45  # 'EMBD' little-endian
    header = struct.pack("<4i", magic, d, D, C)
    body = emb.astype("<f4").tobytes() + pos.astype("<f4").tobytes() + \
           W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* Shared embedding + positional encoding inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 16 || hd[0] != 0x44424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4L * (256L * D + 16L * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *pos_emb = emb + 256L * D;
    const float *W = pos_emb + 16L * D;
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hid = calloc(D, sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(hid, 0, D * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * D;
            const float *p = pos_emb + (long)j * D;
            for (int k = 0; k < D; k++) hid[k] += e[k] + p[k];
        }

        for (int c = 0; c < C; c++) logits[c] = b[c];
        for (int k = 0; k < D; k++) {
            float h = hid[k];
            const float *w = W + (long)k * C;
            for (int c = 0; c < C; c++) logits[c] += h * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```