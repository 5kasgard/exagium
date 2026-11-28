```hypothesis
Family: label-adjlinear (linear model with adjacent pixel products).
Why: Optdigits 8x8 pixels have strong local correlation (strokes); pairwise products of 4-connected neighbors capture orientation/continuity missed by per-pixel linear. Adds only 112 features (vs 64), keeping inference fast (~3× linear cost).
Failure signal: Dev accuracy ≤ label-linear (0.9613) or instruction count > 3× label-linear.
```
```python
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

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n, d = X_bytes.shape
    assert d == 64

    # Define adjacent pairs (right, down) in fixed order
    pairs = []
    for r in range(8):
        for c in range(8):
            idx = r * 8 + c
            if c < 7:
                pairs.append((idx, idx + 1))  # right
            if r < 7:
                pairs.append((idx, idx + 8))  # down
    n_pairs = len(pairs)  # 112
    n_feat = d + n_pairs  # 176

    # Build feature matrix: original + products
    X = X_bytes.astype(np.float32)
    Z = np.zeros((n, n_feat), dtype=np.float32)
    Z[:, :d] = X
    for i, (a, b) in enumerate(pairs):
        Z[:, d + i] = X[:, a] * X[:, b]

    # Normalize per feature
    rng = np.random.default_rng(SEED)
    mean = Z.mean(axis=0)
    std = Z.std(axis=0)
    std[std == 0] = 1.0
    Z = (Z - mean) / std

    # Logistic regression with Adam
    k = n_feat
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W + b
            g = softmax_grad(logits, yy)
            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Save weights: magic, d, n_pairs, C, mean[176], std[176], W[176,10], b[10] (all float32 LE)
    magic = 0x41444A4C  # "ADJL"
    out = struct.pack("<4i", magic, d, n_pairs, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
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
    if (nw < 16 || hd[0] != 0x41444A4C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], n_pairs = hd[2], C = hd[3];
    if (d != 64 || n_pairs != 112 || C != 10) { fprintf(stderr, "bad weights dims\n"); return 3; }
    int n_feat = d + n_pairs;
    size_t expected = 16 + (size_t)n_feat * 4 * 2 + (size_t)n_feat * C * 4 + (size_t)C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld vs %zu\n", nw, expected); return 3; }

    float *mean = (float *)(hd + 4);
    float *std = mean + n_feat;
    float *W = std + n_feat;
    float *b = W + (size_t)n_feat * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * n_feat);
    float *scores = malloc(sizeof(float) * C);

    // Precompute pair indices (must match train.py exactly)
    int pair_a[112], pair_b[112];
    int pidx = 0;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            int idx = r * 8 + c;
            if (c < 7) { pair_a[pidx] = idx; pair_b[pidx] = idx + 1; pidx++; }
            if (r < 7) { pair_a[pidx] = idx; pair_b[pidx] = idx + 8; pidx++; }
        }
    }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Original features (z-scored)
        for (int j = 0; j < d; j++) {
            feat[j] = ((float)xi[j] - mean[j]) / std[j];
        }
        // Pair features (z-scored products)
        for (int p = 0; p < n_pairs; p++) {
            float prod = (float)xi[pair_a[p]] * (float)xi[pair_b[p]];
            feat[d + p] = (prod - mean[d + p]) / std[d + p];
        }

        // Compute scores
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < n_feat; f++) {
            float fv = feat[f];
            if (fv != 0.0f) {
                const float *w = W + (size_t)f * C;
                for (int c = 0; c < C; c++) scores[c] += fv * w[c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```