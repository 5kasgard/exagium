```hypothesis
Family: label-poly2 (polynomial degree-2 expansion + linear model).
Why: Ordinal pixel values 0..16 likely have non-linear relationships with the digit class; adding a squared feature per pixel captures curvature while keeping the model linear in the expanded features. Inference uses the same per-byte lookup tables as label-linear (256×C per byte), so CPU cost should be nearly identical. The tuning split (last 20% of train) will confirm that the extra features do not overfit.
Failure signal: Dev accuracy ≤ label-linear (0.9613) or instruction count significantly higher than label-linear.
```

```python
"""Polynomial degree-2 expansion + linear model for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: ordinal -> z-scored and squared (2 features). Categorical -> one-hot over training values.
Linear model (H=0) trained with Adam. Exported as per-byte lookup tables (same format as label-linear but different magic)."""
import json, math, struct, sys

import numpy as np

H = 0
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x504F4C32  # "POL2"

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
    rng = np.random.default_rng(SEED)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            z = (np.arange(256) - mu) / sd
            # Two features: z and z^2
            E = np.column_stack([z, z*z]).astype(np.float32)  # shape (256, 2)
            enc.append(E)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape
    # Linear model (H=0)
    W1 = np.zeros((k, C), np.float32)
    b1 = np.zeros(C, np.float32)
    params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            def sg(S):
                S = S - S.max(axis=1, keepdims=True)
                P = np.exp(S)
                P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                return (P / len(yy)).astype(np.float32)
            g = sg(z @ W1 + b1)
            grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    # Build per-byte tables
    off = 0
    T = []
    for E in enc:
        d_j = E.shape[1]
        T.append((E @ W1[off:off + d_j]).astype(np.float32))
        off += d_j
    out = struct.pack("<4i", MAGIC, X.shape[1], H, C) + np.stack(T).astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Polynomial degree-2 + linear model for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weights format: int32 magic 0x504F4C32, d, H, C; float32 T[d][256][C], b1[C]. H must be 0. */
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
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || H != 0) { fprintf(stderr, "only H=0 supported\n"); return 3; }
    int W = C;
    long expected = 16 + 4L * ((long)d * 256 * W + W);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * W;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```