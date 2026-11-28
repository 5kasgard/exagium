```hypothesis
Family: label-poly2hash (quadratic polynomial features with feature hashing to 256 buckets).
Why: Optdigits pixels have local correlations; degree-2 interactions capture strokes and curves. Hashing to 256 buckets keeps inference at ~5k ops/item (vs 640 for linear, 1184 for mlp16) while adding non-linearity. Deterministic fixed hash needs no extra storage.
Failure: If dev accuracy ≤ 0.9613 (label-linear) or Ir/item > 8000, the trade-off isn't justified.
```

```python
"""label-poly2hash: quadratic polynomial features hashed to 256 buckets, then linear softmax.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x504F4C32, d, D, C; float32 W[D][C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
D = 256          # hash buckets
MAGIC = 0x504F4C32  # "POL2"

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def hash_bucket(i, j):
    """Deterministic hash of pair (i,j) with i<=j into 0..D-1."""
    return (i * 64 + j) % D

def featurize(X):
    """X: (n, d) uint8 -> (n, D) float32 hashed quadratic features."""
    n, d = X.shape
    assert d == 64
    F = np.zeros((n, D), dtype=np.float32)
    # vectorized over samples, loop over pairs
    for i in range(d):
        xi = X[:, i].astype(np.float32)
        for j in range(i, d):
            b = hash_bucket(i, j)
            F[:, b] += xi * X[:, j].astype(np.float32)
    return F

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    assert C == 10 and X.shape[1] == 64

    F = featurize(X)          # (n, D)
    n, _ = F.shape

    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits, dtype=np.float32)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            f, yy = F[idx], y[idx]
            logits = f @ W + b
            g = softmax_grad(logits, yy)
            grad_W = f.T @ g + L2 * W
            grad_b = g.sum(0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Export
    out = struct.pack("<4i", MAGIC, 64, D, C)
    out += W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-poly2hash infer: ./infer weights.bin in.rec N
 * Input: 64-byte records. Output: 1-byte class (argmax).
 * Weights format: magic 0x504F4C32, d, D, C; float32 W[D][C], b[C]. */
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

static inline int hash_bucket(int i, int j) { return (i * 64 + j) % 256; }

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 64 || D != 256 || C != 10) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 4L * D * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(hd + 4);
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = calloc(D, sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(feat, 0, D * sizeof(float));
        for (int p = 0; p < d; p++) {
            float xp = xi[p];
            for (int q = p; q < d; q++) {
                feat[hash_bucket(p, q)] += xp * xi[q];
            }
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int bucket = 0; bucket < D; bucket++) {
            float f = feat[bucket];
            if (f != 0.0f) {
                const float *w = W + (long)bucket * C;
                for (int c = 0; c < C; c++) scores[c] += f * w[c];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```