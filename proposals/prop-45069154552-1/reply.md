```hypothesis
Family: label-localpoly (linear model with local pairwise interactions).
Why: 8x8 digit pixels have strong spatial correlations; products of adjacent pixels capture edges/strokes.
A linear model on 64 raw + 112 adjacent products (176 features) adds non-linearity at ~3x linear cost.
Failure: dev accuracy < 0.965 (does not beat label-linear's 0.9613 meaningfully).
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

    # Build interaction features: right and down neighbors in 8x8 grid
    interactions = []
    for r in range(8):
        for c in range(8):
            idx = r * 8 + c
            if c < 7:  # right neighbor
                interactions.append((idx, idx + 1))
            if r < 7:  # down neighbor
                interactions.append((idx, idx + 8))
    n_inter = len(interactions)  # 56 + 56 = 112

    # Features: 64 raw + 112 interactions = 176
    X = np.zeros((n, 176), dtype=np.float32)
    X[:, :64] = X_bytes.astype(np.float32)
    for k, (i, j) in enumerate(interactions):
        X[:, 64 + k] = X_bytes[:, i].astype(np.float32) * X_bytes[:, j].astype(np.float32)

    # Z-score per feature
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    Z = (X - means) / stds

    rng = np.random.default_rng(SEED)
    W = np.zeros((176, C), np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W + b
            g = softmax_grad(logits, yy)
            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(axis=0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Save weights: magic, n_feat, C, n_inter; means, stds, W, b (all float32 LE)
    magic = 0x4C504C31  # 'LPL1'
    out = struct.pack("<4i", magic, 176, C, 112)
    out += means.astype("<f4").tobytes()
    out += stds.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4C504C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int n_feat = hd[1], C = hd[2], n_inter = hd[3];
    if (n_feat != 176 || C != 10 || n_inter != 112) {
        fprintf(stderr, "bad weights dims\n"); return 3;
    }
    long expected = 16 + 4L * (n_feat + n_feat + (long)n_feat * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *means = (const float *)(hd + 4);
    const float *stds = means + n_feat;
    const float *W = stds + n_feat;
    const float *b = W + (long)n_feat * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * n_feat);
    float *scores = malloc(sizeof(float) * C);

    int inter_idx[112][2];
    int k = 0;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            int idx = r * 8 + c;
            if (c < 7) { inter_idx[k][0] = idx; inter_idx[k][1] = idx + 1; k++; }
            if (r < 7) { inter_idx[k][0] = idx; inter_idx[k][1] = idx + 8; k++; }
        }
    }

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, len); return 3; }

        for (int j = 0; j < 64; j++) {
            feat[j] = (xi[j] - means[j]) / stds[j];
        }
        for (int m = 0; m < 112; m++) {
            int a = inter_idx[m][0];
            int b_idx = inter_idx[m][1];
            float val = (float)xi[a] * (float)xi[b_idx];
            feat[64 + m] = (val - means[64 + m]) / stds[64 + m];
        }

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < n_feat; f++) {
            float fv = feat[f];
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