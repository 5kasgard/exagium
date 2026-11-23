```hypothesis
Family: label-poly2 (degree-2 polynomial features + linear model). The 8x8 digit pixels have local correlations; pairwise products capture stroke intersections and curvatures that a linear model misses. With 64 inputs -> 2144 features, a single linear layer should beat label-linear (0.9613) while keeping inference fast (no hidden layer). Failure: dev accuracy <= 0.9613 or CPU cost > 2x label-linear.
```
```python
"""label-poly2: polynomial degree-2 features + linear classifier for optdigits.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x504F4C32, d, D, C; float32 mean[d], std[d], W[D][C], b[C].
D = d + d*(d+1)//2 = 2144 for d=64."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def expand(X, mean, std):
    """X: (n, d) normalized float32. Return (n, D) with linear + quadratic terms."""
    n, d = X.shape
    D = d + d * (d + 1) // 2
    Z = np.empty((n, D), dtype=np.float32)
    Z[:, :d] = X
    col = d
    for i in range(d):
        xi = X[:, i:i+1]          # (n,1)
        Z[:, col:col + d - i] = xi * X[:, i:]  # (n, d-i)
        col += d - i
    return Z


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X_bytes.shape[1]
    C = card["classes"]

    # Normalize per feature (ordinal z-score)
    X = X_bytes.astype(np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_norm = ((X - mean) / std).astype(np.float32)

    # Expand to polynomial features
    Z = expand(X_norm, mean, std)  # (n, D)
    n, D = Z.shape

    rng = np.random.default_rng(SEED)
    # Linear weights
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
            # Softmax gradient
            scores = z @ W + b
            scores -= scores.max(axis=1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)   # (B, C)
            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(axis=0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, D, C; mean[d], std[d]; W[D][C] row-major; b[C]
    magic = 0x504F4C32  # "POL2"
    out = struct.pack("<4i", magic, d, D, C)
    out += np.array(mean, dtype=np.float32).tobytes()
    out += np.array(std, dtype=np.float32).tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-poly2 infer: ./infer weights.bin in.rec N
 * Input record: d bytes (d=64). Output: 1 byte class (argmax).
 * Weights format: magic 0x504F4C32, int32 d, D, C; float32 mean[d], std[d], W[D*C], b[C]. */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 64 || C < 1 || C > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + (long)d * 4 * 2 + (long)D * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std  = mean + d;
    const float *W    = std + d;
    const float *b    = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * D);
    float *score = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize
        for (int j = 0; j < d; j++) {
            float v = (float)xi[j];
            float s = std[j];
            feat[j] = (v - mean[j]) / (s == 0.0f ? 1.0f : s);
        }
        // Quadratic terms (upper triangular including diagonal)
        int idx = d;
        for (int u = 0; u < d; u++) {
            float fu = feat[u];
            for (int v = u; v < d; v++) {
                feat[idx++] = fu * feat[v];
            }
        }
        // Linear classification
        for (int c = 0; c < C; c++) score[c] = b[c];
        for (int f = 0; f < D; f++) {
            float val = feat[f];
            if (val == 0.0f) continue;
            const float *wf = W + (long)f * C;
            for (int c = 0; c < C; c++) score[c] += val * wf[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(feat); free(score);
    return 0;
}
```