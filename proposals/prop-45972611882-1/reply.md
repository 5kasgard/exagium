```hypothesis
The family "label-poly2" expands each 8x8 digit into 2144 features (64 linear + 2080 quadratic pixel products) and trains a multinomial logistic regression with Adam. Pairwise interactions capture stroke junctions and loops crucial for digit discrimination, improving over linear while staying fast (~20k FLOPs/item). If dev accuracy ≤ 0.9613 (label-linear) or latency exceeds label-mlp16, the hypothesis fails.
```

```python
"""label-poly2: degree-2 polynomial expansion + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
MAGIC = 0x504F4C32  # "POL2"

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def expand(X):
    """X: (n, 64) uint8 -> (n, 2144) float32 normalized to [0,1] with quadratic terms."""
    n, d = X.shape
    assert d == 64
    x = X.astype(np.float32) / 16.0  # 0..1
    exp_dim = d + d * (d + 1) // 2  # 64 + 2080 = 2144
    Z = np.empty((n, exp_dim), dtype=np.float32)
    Z[:, :d] = x
    col = d
    for i in range(d):
        xi = x[:, i:i+1]
        for j in range(i, d):
            Z[:, col] = (xi * x[:, j]).ravel()
            col += 1
    assert col == exp_dim
    return Z

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 64

    Z = expand(X)
    n, k = Z.shape
    assert k == 2144

    rng = np.random.default_rng(SEED)
    # Xavier init for linear layer
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # softmax gradient
            scores = z @ W + b
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = (probs / len(yy)).astype(np.float32)
            # gradients
            gW = z.T @ grad + L2 * W
            gb = grad.sum(axis=0)
            grads = [gW, gb]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights.bin
    out = struct.pack("<4i", MAGIC, d, k, C)
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-poly2 infer: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixels 0..16). Output: one byte class 0..9.
 * Weights format: magic(0x504F4C32), d, exp_dim, C; float32 W[exp_dim][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], exp_dim = hd[2], C = hd[3];
    if (d != 64 || exp_dim != 2144 || C != 10) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 16 + 4L * ((long)exp_dim * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W = (const float *)(hd + 4);
    const float *b = W + (long)exp_dim * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * exp_dim);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // linear terms (normalized)
        for (int j = 0; j < d; j++) feat[j] = xi[j] / 16.0f;
        // quadratic terms
        int col = d;
        for (int i1 = 0; i1 < d; i1++) {
            float xi1 = feat[i1];
            for (int i2 = i1; i2 < d; i2++) {
                feat[col++] = xi1 * feat[i2];
            }
        }
        // compute scores
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int k_idx = 0; k_idx < exp_dim; k_idx++) {
            float f = feat[k_idx];
            const float *wrow = W + (long)k_idx * C;
            for (int c = 0; c < C; c++) scores[c] += f * wrow[c];
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```