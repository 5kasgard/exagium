```hypothesis
Family: label-elm (Extreme Learning Machine: fixed random projection + ReLU + ridge regression).
Why: Optdigits benefits from non-linear features; a random projection (fixed seed) expands 64 pixels into a higher-dimensional space where classes are more linearly separable. Training only the output layer with ridge regression is fast, deterministic, and avoids vanishing gradients. Tuning hidden size H on the 20% tuning split balances accuracy and inference cost.
Failure: Dev accuracy ≤ 0.9613 (label-linear) or inference cost ≥ label-mlp16 (8685 Ir/item).
```
```python
"""ELM family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal features as z-scores, categorical as one-hot.
Random projection W1 (k x H) and bias b1 fixed by seed 123.
Hidden activations: ReLU(Z @ W1 + b1). Output weights W2, b2 trained by ridge regression on one-hot labels.
Exported as table-family format (magic 0x54424C31) compatible with the provided infer.c."""
import json, math, struct, sys

import numpy as np

SEED = 1
RIDGE_LAMBDA = 1e-3
CANDIDATE_H = [32, 64, 128]  # tuned on last 20% of train.in


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoding(card, X):
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def make_Z(X, enc):
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def ridge_fit(H_act, Y_onehot, lam):
    # H_act: n x H, Y_onehot: n x C
    n, H = H_act.shape
    Hb = np.hstack([H_act, np.ones((n, 1), dtype=np.float32)])  # n x (H+1)
    A = Hb.T @ Hb + lam * np.eye(H + 1, dtype=np.float32)
    B = Hb.T @ Y_onehot
    Wb = np.linalg.solve(A, B)  # (H+1) x C
    return Wb[:-1], Wb[-1]  # W2 (H x C), b2 (C)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]

    # Split: last 20% for tuning
    split = int(0.8 * n)
    idx_train, idx_tune = np.arange(split), np.arange(split, n)
    X_train, y_train = X[idx_train], y[idx_train]
    X_tune, y_tune = X[idx_tune], y[idx_tune]

    enc = build_encoding(card, X_train)
    Z_train = make_Z(X_train, enc)
    Z_tune = make_Z(X_tune, enc)

    # One-hot labels
    Y_train_oh = np.eye(C, dtype=np.float32)[y_train]
    Y_tune_oh = np.eye(C, dtype=np.float32)[y_tune]

    best_H, best_acc = CANDIDATE_H[0], -1.0
    for H in CANDIDATE_H:
        rng = np.random.default_rng(123 + H)
        k = Z_train.shape[1]
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = rng.uniform(-0.1, 0.1, H).astype(np.float32)

        H_train = np.maximum(Z_train @ W1 + b1, 0)
        H_tune = np.maximum(Z_tune @ W1 + b1, 0)

        W2, b2 = ridge_fit(H_train, Y_train_oh, RIDGE_LAMBDA)

        logits = H_tune @ W2 + b2
        pred = logits.argmax(axis=1)
        acc = (pred == y_tune).mean()
        if acc > best_acc:
            best_acc, best_H = acc, H

    # Retrain on all data with best_H
    enc_all = build_encoding(card, X)
    Z_all = make_Z(X, enc_all)
    Y_all_oh = np.eye(C, dtype=np.float32)[y]

    rng = np.random.default_rng(123 + best_H)
    k = Z_all.shape[1]
    W1 = (rng.uniform(-1, 1, (k, best_H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = rng.uniform(-0.1, 0.1, best_H).astype(np.float32)

    H_all = np.maximum(Z_all @ W1 + b1, 0)
    W2, b2 = ridge_fit(H_all, Y_all_oh, RIDGE_LAMBDA)

    # Build lookup tables T for export (same format as label-linear)
    off, T = 0, []
    for E in enc_all:
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]

    out = struct.pack("<4i", 0x54424C31, X.shape[1], best_H, C) + np.stack(T).astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
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
    return b;
}

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```