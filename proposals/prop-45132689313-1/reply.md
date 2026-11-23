```hypothesis
Family: label-elm (Extreme Learning Machine). Fixed random projection (H=32) with ReLU, then ridge-regression output layer. 
Why: Random features give non-linearity at near-linear cost; ridge regression is deterministic and fast. 
Fails if: dev accuracy ≤ label-linear (0.9613) or inference cost exceeds label-mlp16.
```
```python
"""label-elm: fixed random projection + ridge regression.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

H = 32
SEED = 1
LAMBDA = 1e-4

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

    # per-feature z-score tables (256 values each)
    rng = np.random.default_rng(SEED)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd).astype(np.float32))
        else:
            vals = np.unique(X[:, j])
            E = np.zeros(256, np.float32)
            E[vals] = 1.0
            enc.append(E)
    Z = np.stack([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)  # (n, d)

    # fixed random projection: W1 (d, H), b1 (H)
    W1 = rng.uniform(-math.sqrt(6.0 / d), math.sqrt(6.0 / d), (d, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)

    # hidden activations for training
    H_train = np.maximum(Z @ W1 + b1, 0, dtype=np.float32)  # (n, H)

    # ridge regression on [H_train, 1] -> one-hot labels
    n = H_train.shape[0]
    H_aug = np.hstack([H_train, np.ones((n, 1), np.float32)])  # (n, H+1)
    Y = np.eye(C, dtype=np.float32)[y]  # (n, C)
    A = H_aug.T @ H_aug + LAMBDA * np.eye(H + 1, dtype=np.float32)
    B = H_aug.T @ Y
    W_aug = np.linalg.solve(A, B)  # (H+1, C)
    W2 = W_aug[:H, :]  # (H, C)
    b2 = W_aug[H, :]   # (C,)

    # fuse encoding and W1 into U[j, v, h] = enc[j][v] * W1[j, h]
    U = np.empty((d, 256, H), dtype=np.float32)
    for j in range(d):
        U[j] = enc[j][:, None] * W1[j, :]

    # write weights: magic, d, H, C; U (d*256*H); b1 (H); W2 (H*C); b2 (C)
    magic = 0x314D4C45  # 'ELM1'
    out = struct.pack("<4i", magic, d, H, C)
    out += U.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-elm inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index). */
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
    if (nw < 16 || hd[0] != 0x314D4C45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *U = (const float *)(hd + 4);
    const float *b1 = U + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = (float *)malloc(sizeof(float) * H);
    float *sc = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(acc, b1, H * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *u = U + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += u[h];
        }
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;
        memcpy(sc, b2, C * sizeof(float));
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) sc[c] += ah * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```