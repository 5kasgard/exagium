```hypothesis
Family: label-poly2local (local degree-2 polynomial logistic regression).
Why: 8x8 digits have strong local spatial correlations (strokes, curves). A linear model misses these; a full MLP is overkill. Using only 4-neighbor pairwise interactions (right, down, down-right, down-left) yields 274 features—capturing local structure with minimal parameters. SGD with Adam (same recipe as label-linear) trains fast and deterministically. Inference is ~2740 multiply-adds/item, cheaper than label-mlp16.
Failure signal: dev accuracy ≤ 0.9613 (label-linear) or inference cost ≥ label-mlp16.
```

```python
"""label-poly2local: local degree-2 polynomial logistic regression for 8x8 digit images.
Features: 64 linear (pixel values) + 210 local quadratic (4 neighbor directions per pixel).
Train: Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)), seed=1.
Weights file: magic(0x504F4C32), nfeat, nclass; float32 mean[nfeat], std[nfeat], W[nfeat*nclass], b[nclass]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x504F4C32  # 'POL2'

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_features(X):
    """X: (n, 64) uint8. Returns (n, 274) float32."""
    n = X.shape[0]
    feats = np.zeros((n, 274), dtype=np.float32)
    # Linear: 0..63
    feats[:, :64] = X.astype(np.float32)
    # Quadratic: 4 neighbor directions
    col = 64
    for r in range(8):
        for c in range(8):
            i = r * 8 + c
            xi = X[:, i].astype(np.float32)
            # right
            if c < 7:
                j = r * 8 + (c + 1)
                feats[:, col] = xi * X[:, j].astype(np.float32)
                col += 1
            # down
            if r < 7:
                j = (r + 1) * 8 + c
                feats[:, col] = xi * X[:, j].astype(np.float32)
                col += 1
            # down-right
            if r < 7 and c < 7:
                j = (r + 1) * 8 + (c + 1)
                feats[:, col] = xi * X[:, j].astype(np.float32)
                col += 1
            # down-left
            if r < 7 and c > 0:
                j = (r + 1) * 8 + (c - 1)
                feats[:, col] = xi * X[:, j].astype(np.float32)
                col += 1
    assert col == 274
    return feats

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    assert X.shape[1] == 64

    Z = build_features(X)  # (n, 274)
    n, D = Z.shape

    # Standardize features on training set
    mean = Z.mean(axis=0).astype(np.float32)
    std = Z.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    Z = (Z - mean) / std

    rng = np.random.default_rng(SEED)
    # Logistic regression weights
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            scores = z @ W + b
            g = softmax_grad(scores, yy)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Pack weights: magic, D, C; mean[D], std[D], W[D*C], b[C]
    out = struct.pack("<3i", MAGIC, D, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-poly2local inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (8x8 pixels 0..16). Output record: 1 byte class 0..9.
 * Weights format: int32 magic(0x504F4C32), D, C; float32 mean[D], std[D], W[D*C], b[C].
 * Features: 64 linear + 210 local quadratic (right, down, down-right, down-left). */
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
    if (nw < 12 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int D = hd[1], C = hd[2];
    if (D != 274 || C != 10) { fprintf(stderr, "unexpected dims %d %d\n", D, C); return 3; }
    long expected = 12L + 4L * (2L * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 3);
    const float *std  = mean + D;
    const float *W    = std + D;
    const float *b    = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * D);
    float *score = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }

        // Linear features (0..63)
        for (int j = 0; j < 64; j++) feat[j] = (float)xi[j];

        // Quadratic features (64..273)
        int col = 64;
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                int idx = r * 8 + c;
                float xv = (float)xi[idx];
                // right
                if (c < 7) {
                    int j = r * 8 + (c + 1);
                    feat[col++] = xv * (float)xi[j];
                }
                // down
                if (r < 7) {
                    int j = (r + 1) * 8 + c;
                    feat[col++] = xv * (float)xi[j];
                }
                // down-right
                if (r < 7 && c < 7) {
                    int j = (r + 1) * 8 + (c + 1);
                    feat[col++] = xv * (float)xi[j];
                }
                // down-left
                if (r < 7 && c > 0) {
                    int j = (r + 1) * 8 + (c - 1);
                    feat[col++] = xv * (float)xi[j];
                }
            }
        }

        // Normalize
        for (int j = 0; j < D; j++) feat[j] = (feat[j] - mean[j]) / std[j];

        // Scores = feat @ W + b
        for (int c = 0; c < C; c++) score[c] = b[c];
        for (int j = 0; j < D; j++) {
            float fj = feat[j];
            const float *wj = W + (long)j * C;
            for (int c = 0; c < C; c++) score[c] += fj * wj[c];
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```