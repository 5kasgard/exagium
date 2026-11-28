```hypothesis
Family: label-linpair (Linear with Spatial Pairwise Interactions).
Why: 8x8 digit pixels have strong local spatial structure (strokes, corners). Adding products of horizontally and vertically adjacent pixels (112 pairs) gives a linear model explicit access to 2nd-order features that capture edges and curves, which a pure linear model misses. This should lift accuracy toward MLP levels (0.977) while keeping inference extremely fast (just 176 features × 10 classes). The model stays deterministic, single-threaded, and uses only stdlib+numpy.
Failure signal: Dev accuracy ≤ 0.965 or instruction cost > 20k Ir/item (≈4× label-linear).
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
    X_raw = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n, d = X_raw.shape
    assert d == 64

    # Horizontal and vertical adjacent pairs (8x8 grid)
    h_pairs = [(r * 8 + c, r * 8 + c + 1) for r in range(8) for c in range(7)]  # 56
    v_pairs = [(r * 8 + c, (r + 1) * 8 + c) for r in range(7) for c in range(8)]  # 56
    pairs = h_pairs + v_pairs  # 112
    F = d + len(pairs)  # 176

    X = np.zeros((n, F), dtype=np.float32)
    X[:, :d] = X_raw.astype(np.float32)
    for idx, (i, j) in enumerate(pairs):
        X[:, d + idx] = X_raw[:, i] * X_raw[:, j]

    # Per-feature z-score on training set
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    inv_std = (1.0 / std).astype(np.float32)
    X = (X - mean) * inv_std

    # Linear classifier with Adam (same recipe as label-linear)
    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (F, C)) * math.sqrt(6.0 / F)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x, yy = X[idx], y[idx]
            logits = x @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_logits = (probs / len(yy)).astype(np.float32)
            grad_W = x.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export: magic, d, F, C; mean[F], inv_std[F], W[F][C], b[C] (all float32 LE)
    magic = 0x4C504950  # "LPIP"
    out = struct.pack("<4i", magic, d, F, C)
    out += mean.tobytes()
    out += inv_std.tobytes()
    out += W.tobytes()
    out += b.tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4C504950) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], F = hd[2], C = hd[3];
    if (d != 64 || F != 176 || C != 10) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16 + 4L * (F + F + (long)F * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *inv_std = mean + F;
    const float *W = inv_std + F;
    const float *b = W + (long)F * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * F);
    float *scores = malloc(sizeof(float) * C);

    /* Pair indices (precomputed constants) */
    int h_pairs[56][2], v_pairs[56][2];
    for (int r = 0, p = 0; r < 8; r++) for (int c = 0; c < 7; c++, p++) {
        h_pairs[p][0] = r * 8 + c; h_pairs[p][1] = r * 8 + c + 1;
    }
    for (int r = 0, p = 0; r < 7; r++) for (int c = 0; c < 8; c++, p++) {
        v_pairs[p][0] = r * 8 + c; v_pairs[p][1] = (r + 1) * 8 + c;
    }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* Raw features 0..63 */
        for (int j = 0; j < d; j++) {
            feat[j] = (xi[j] - mean[j]) * inv_std[j];
        }
        /* Horizontal pairwise 64..119 */
        for (int p = 0; p < 56; p++) {
            int a = h_pairs[p][0], b_idx = h_pairs[p][1];
            int f = 64 + p;
            float val = (float)xi[a] * (float)xi[b_idx];
            feat[f] = (val - mean[f]) * inv_std[f];
        }
        /* Vertical pairwise 120..175 */
        for (int p = 0; p < 56; p++) {
            int a = v_pairs[p][0], b_idx = v_pairs[p][1];
            int f = 120 + p;
            float val = (float)xi[a] * (float)xi[b_idx];
            feat[f] = (val - mean[f]) * inv_std[f];
        }

        /* Linear classification */
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < F; f++) {
            float fv = feat[f];
            if (fv == 0.0f) continue;
            const float *w = W + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += fv * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```