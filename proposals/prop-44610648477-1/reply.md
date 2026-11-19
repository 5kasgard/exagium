```hypothesis
Family: label-poly2 (degree-2 polynomial logistic regression on normalized pixels).
Why: Opendigits pixels have strong pairwise interactions (strokes, loops) that a linear model misses. A full quadratic expansion (2145 features) captures these while staying a convex problem, so deterministic Adam trains fast. The model is still a single linear layer on expanded features, so inference is a tight loop of ~2k multiply-adds per item.
Failure signal: dev accuracy ≤ 0.961 (label-linear) or CPU cost > 3× label-linear.
```

```python
"""label-poly2: polynomial logistic regression (degree 2) on z-scored ordinal bytes.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format (little-endian):
  uint32 magic 0x504F4C32 ("POL2")
  int32  d (input bytes, 64)
  int32  C (classes, 10)
  int32  D (expanded features = 1 + d + d*(d+1)/2 = 2145)
  float32 mean[d], std[d]
  float32 W[D*C] (row-major: D rows, C cols)
  float32 b[C]
"""
import json, math, struct, sys
import numpy as np

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
    d = X.shape[1]
    C = card["classes"]
    assert d == 64 and C == 10

    rng = np.random.default_rng(SEED)

    # z-score each ordinal feature
    Xf = X.astype(np.float64)
    mean = Xf.mean(axis=0)
    std = Xf.std(axis=0)
    std[std == 0] = 1.0
    Xn = (Xf - mean) / std  # (n, d)

    # Build expanded feature matrix Z: [1, x_i, x_i*x_j (i<=j)]
    n = Xn.shape[0]
    D = 1 + d + d * (d + 1) // 2  # 2145
    Z = np.empty((n, D), dtype=np.float32)
    Z[:, 0] = 1.0
    Z[:, 1:1+d] = Xn.astype(np.float32)
    col = 1 + d
    for i in range(d):
        xi = Xn[:, i:i+1].astype(np.float32)  # (n,1)
        Z[:, col:col + d - i] = xi * Xn[:, i:].astype(np.float32)
        col += d - i
    assert col == D

    # Initialize weights
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
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
            grad = probs / len(yy)  # (batch, C)
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

    # Pack weights
    out = struct.pack("<4i", MAGIC, d, C, D)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-poly2 inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixels). Output record: 1 byte (class 0..9).
 * Weights format: see train.py. */
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], D = hd[3];
    if (d != 64 || C != 10 || D != 2145) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    size_t expected = 16 + (size_t)d * 4 * 2 + (size_t)D * C * 4 + (size_t)C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std  = mean + d;
    const float *W    = std + d;
    const float *b    = W + (size_t)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xn = malloc(d * sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // normalize
        for (int j = 0; j < d; j++) {
            xn[j] = (xi[j] - mean[j]) / std[j];
        }
        // compute scores = b + W^T * phi(xn)
        memcpy(scores, b, C * sizeof(float));
        // bias term (index 0)
        for (int c = 0; c < C; c++) scores[c] += W[c];
        // linear terms (indices 1..d)
        for (int j = 0; j < d; j++) {
            const float *wj = W + (1 + j) * C;
            float xv = xn[j];
            for (int c = 0; c < C; c++) scores[c] += xv * wj[c];
        }
        // quadratic terms (i<=j)
        size_t base = 1 + d;
        for (int i1 = 0; i1 < d; i1++) {
            for (int i2 = i1; i2 < d; i2++) {
                const float *wij = W + (base++) * C;
                float prod = xn[i1] * xn[i2];
                for (int c = 0; c < C; c++) scores[c] += prod * wij[c];
            }
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```