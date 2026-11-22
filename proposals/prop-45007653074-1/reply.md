```hypothesis
Family: label-mlp2h (two-hidden-layer MLP with ReLU, 64 and 32 units).
Why: The 8×8 digit images have spatial correlations; a deeper MLP can learn hierarchical features (edges → shapes) that a single-layer MLP or linear model cannot. With H1=64, H2=32 it has more capacity than label-mlp16 (16 units) and should approach label-gbdt accuracy while remaining faster than label-knn.
Failure: If dev accuracy ≤ 0.9613 (label-linear) or CPU cost per item exceeds label-mlp16 by >2× without a corresponding accuracy gain.
```

```python
"""label-mlp2h: two-hidden-layer MLP with ReLU for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports weights as: magic(0x4D4C5032), d, H1, H2, C (int32),
then float32 arrays: T1[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C]."""
import json, math, struct, sys

import numpy as np

H1, H2 = 64, 32
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # per-pixel normalization (ordinal only)
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Xn = ((X - mu) / sd).astype(np.float32)  # (n, d)

    # He initialization for ReLU
    W1 = (rng.normal(0, math.sqrt(2.0 / d), (d, H1))).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H1), (H1, H2))).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.normal(0, math.sqrt(2.0 / H2), (H2, C))).astype(np.float32)
    b3 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = Xn.shape[0]
    steps = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0
    for _ in range(steps):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x, yy = Xn[idx], y[idx]
            m = x.shape[0]

            # forward
            z1 = x @ W1 + b1
            a1 = np.maximum(z1, 0)
            z2 = a1 @ W2 + b2
            a2 = np.maximum(z2, 0)
            z3 = a2 @ W3 + b3

            # softmax cross-entropy gradient
            z3 -= z3.max(axis=1, keepdims=True)
            exp_z3 = np.exp(z3)
            probs = exp_z3 / exp_z3.sum(axis=1, keepdims=True)
            probs[np.arange(m), yy] -= 1.0
            probs /= m

            # backward
            dW3 = a2.T @ probs + L2 * W3
            db3 = probs.sum(axis=0)
            da2 = probs @ W3.T
            dz2 = da2 * (z2 > 0)
            dW2 = a1.T @ dz2 + L2 * W2
            db2 = dz2.sum(axis=0)
            da1 = dz2 @ W2.T
            dz1 = da1 * (z1 > 0)
            dW1 = x.T @ dz1 + L2 * W1
            db1 = dz1.sum(axis=0)

            grads = [dW1, db1, dW2, db2, dW3, db3]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m1, v1 = state[i]
                m1 *= 0.9; m1 += 0.1 * gr
                v1 *= 0.999; v1 += 0.001 * gr * gr
                m_hat = m1 / (1 - 0.9 ** t)
                v_hat = v1 / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # build per-byte lookup tables for first layer
    T1 = np.empty((d, 256, H1), dtype=np.float32)
    for j in range(d):
        norm_vals = (np.arange(256, dtype=np.float32) - mu[j]) / sd[j]
        T1[j] = norm_vals[:, None] * W1[j]  # (256, H1)

    # pack weights
    magic = 0x4D4C5032  # '2PLM' little-endian
    hdr = struct.pack("<5i", magic, d, H1, H2, C)
    out = hdr
    out += T1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    out += W3.astype("<f4").tobytes()
    out += b3.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-mlp2h inference: ./infer weights.bin in.rec N
 * Two-hidden-layer MLP with ReLU. Input: d bytes. Output: one byte class. */
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
    if (nw < 20 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d < 1) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 20 + 4L * ((long)d * 256 * H1 + H1 + (long)H1 * H2 + H2 + (long)H2 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    const float *T1 = (const float *)(hd + 5);
    const float *b1 = T1 + (long)d * 256 * H1;
    const float *W2 = b1 + H1;
    const float *b2 = W2 + (long)H1 * H2;
    const float *W3 = b2 + H2;
    const float *b3 = W3 + (long)H2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc1 = malloc(sizeof(float) * H1);
    float *acc2 = malloc(sizeof(float) * H2);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(acc1, b1, H1 * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *t = T1 + ((long)j * 256 + xi[j]) * H1;
            for (int h = 0; h < H1; h++) acc1[h] += t[h];
        }
        for (int h = 0; h < H1; h++) if (acc1[h] < 0.0f) acc1[h] = 0.0f;

        memcpy(acc2, b2, H2 * sizeof(float));
        for (int h1 = 0; h1 < H1; h1++) {
            float a = acc1[h1];
            const float *w = W2 + (long)h1 * H2;
            for (int h2 = 0; h2 < H2; h2++) acc2[h2] += a * w[h2];
        }
        for (int h2 = 0; h2 < H2; h2++) if (acc2[h2] < 0.0f) acc2[h2] = 0.0f;

        memcpy(scores, b3, C * sizeof(float));
        for (int h2 = 0; h2 < H2; h2++) {
            float a = acc2[h2];
            const float *w = W3 + (long)h2 * C;
            for (int c = 0; c < C; c++) scores[c] += a * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```