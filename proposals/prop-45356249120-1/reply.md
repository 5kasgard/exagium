```hypothesis
Family: label-rff (Random Fourier Features + linear classifier).  
Why: Optdigits pixels have non-linear boundaries; RFF approximates an RBF kernel, giving non-linear power with only two matmuls and cos/sin at inference. Fixed random projection (seed=1) ensures determinism.  
Failure: Dev accuracy ≤ label-linear (0.9613) or inference cost > label-mlp16 (8685 Ir/item).
```

```python
"""label-rff: Random Fourier Features (Gaussian kernel approx.) + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic(0x52464631), d, D, C; float32 mean[d], std[d], W_rff[D][d], b_rff[D], W_clf[2D][C], b_clf[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
D = 64                 # number of random frequencies (=> 2D features)
SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100


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
    n, d = X.shape

    rng = np.random.default_rng(SEED)

    # Per-feature standardization (ordinal types)
    mean = X.mean(axis=0).astype(np.float64)
    std = X.std(axis=0).astype(np.float64)
    std[std == 0] = 1.0
    Xn = ((X - mean) / std).astype(np.float32)          # (n, d)

    # Random Fourier Features: W_rff ~ N(0,1), b_rff ~ Uniform(0, 2π)
    W_rff = rng.normal(0, 1, (D, d)).astype(np.float32)
    b_rff = rng.uniform(0, 2*math.pi, D).astype(np.float32)

    # Project: Z = Xn @ W_rff.T + b_rff  -> (n, D)
    Z = Xn @ W_rff.T + b_rff
    # Features: [cos(Z), sin(Z)] -> (n, 2D)
    Feat = np.concatenate([np.cos(Z), np.sin(Z)], axis=1).astype(np.float32)
    k = Feat.shape[1]          # 2D

    # Linear classifier on features: weights W_clf (k, C), bias b_clf (C)
    W_clf = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b_clf = np.zeros(C, np.float32)
    params = [W_clf, b_clf]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Feat[idx], y[idx]
            # Softmax gradient
            S = z @ W_clf + b_clf
            S = S - S.max(axis=1, keepdims=True)
            P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            # Gradients
            grad_W = z.T @ g + L2 * W_clf
            grad_b = g.sum(0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)

    # Pack weights
    # header: magic, d, D, C
    hdr = struct.pack("<4i", 0x52464631, d, D, C)
    # mean[d], std[d]
    body = mean.astype("<f4").tobytes() + std.astype("<f4").tobytes()
    # W_rff[D][d], b_rff[D]
    body += W_rff.astype("<f4").tobytes() + b_rff.astype("<f4").tobytes()
    # W_clf[2D][C], b_clf[C]
    body += W_clf.astype("<f4").tobytes() + b_clf.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(hdr + body)


if __name__ == "__main__":
    main()
```

```c
/* label-rff inference: ./infer weights.bin in.rec N
 * Input record: d bytes (uint8). Output record: 1 byte (class 0..9).
 * Weights format (little-endian):
 *   int32 magic(0x52464631), d, D, C
 *   float32 mean[d], std[d]
 *   float32 W_rff[D][d], b_rff[D]
 *   float32 W_clf[2*D][C], b_clf[C] */
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
    if (nw < 16 || hd[0] != 0x52464631) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad header dims\n"); return 3; }
    int K = 2 * D;                     // feature dimension
    long expected = 16L
        + 4L * (d + d                 // mean, std
                + (long)D * d + D      // W_rff, b_rff
                + (long)K * C + C);    // W_clf, b_clf
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *ptr = (const float *)(hd + 4);
    const float *mean = ptr; ptr += d;
    const float *std  = ptr; ptr += d;
    const float *W_rff = ptr; ptr += (long)D * d;
    const float *b_rff = ptr; ptr += D;
    const float *W_clf = ptr; ptr += (long)K * C;
    const float *b_clf = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * D);
    float *feat = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize: (xi - mean) / std
        // Compute Z = W_rff * x_norm + b_rff
        for (int u = 0; u < D; u++) {
            float sum = b_rff[u];
            const float *w = W_rff + (long)u * d;
            for (int j = 0; j < d; j++) {
                float xn = (xi[j] - mean[j]) / std[j];
                sum += w[j] * xn;
            }
            z[u] = sum;
        }
        // Features: cos(z), sin(z)
        for (int u = 0; u < D; u++) {
            feat[u] = cosf(z[u]);
            feat[D + u] = sinf(z[u]);
        }
        // Linear classifier: scores = feat @ W_clf + b_clf
        for (int c = 0; c < C; c++) scores[c] = b_clf[c];
        for (int f = 0; f < K; f++) {
            float val = feat[f];
            if (val == 0.0f) continue;
            const float *w = W_clf + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += val * w[c];
        }
        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```