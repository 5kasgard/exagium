```hypothesis
Family: label-quadratic (quadratic feature expansion + multinomial logistic regression).
Why: Optdigits pixels exhibit spatial correlations (strokes); pairwise products capture 2nd-order interactions
without the cost of a hidden layer. The expanded dimension (2080) is still small enough for fast inference.
Failure: Dev accuracy ≤ 0.9613 (label-linear) or CPU cost ≥ 5058 Ir/item.
```

```python
"""label-quadratic: quadratic feature expansion + linear model for optdigits.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x51554144  # "QUAD" in little-endian


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
    d = X.shape[1]                     # 64
    D = d + d * (d - 1) // 2           # 2080

    rng = np.random.default_rng(SEED)

    # Normalize each original feature
    Xf = X.astype(np.float64)
    mean = Xf.mean(axis=0)
    std = Xf.std(axis=0)
    std[std == 0] = 1.0
    Znorm = (Xf - mean) / std          # (n, d)

    # Build quadratic features: first d are linear, then all pairwise products
    n = X.shape[0]
    Z = np.empty((n, D), dtype=np.float32)
    Z[:, :d] = Znorm.astype(np.float32)
    idx = d
    for i in range(d):
        for j in range(i + 1, d):
            Z[:, idx] = Znorm[:, i] * Znorm[:, j]
            idx += 1

    # Linear model weights
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx_batch = order[s:s + BATCH]
            z, yy = Z[idx_batch], y[idx_batch]
            # Softmax gradient
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = (probs / len(yy)).astype(np.float32)
            # Gradients
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

    # Save weights.bin: magic, d, C, D, mean[d], std[d], W[D*C], b[C] (all float32 little-endian)
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
/* label-quadratic infer.c: quadratic expansion + linear classification.
 * ./infer weights.bin in.rec N  -> writes N one-byte class records to stdout. */
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
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }

    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x51554144) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], C = hd[2], D = hd[3];
    if (d != 64 || C != 10 || D != 2080) { fprintf(stderr, "unexpected dimensions\n"); return 3; }

    size_t expected = 16 + (size_t)d * 4 * 2 + (size_t)D * C * 4 + (size_t)C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld, expected %zu\n", nw, expected); return 3; }

    float *mean = (float *)(wb + 16);
    float *std  = mean + d;
    float *W    = std + d;
    float *b    = W + (size_t)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *znorm = malloc(d * sizeof(float));
    float *zexp  = malloc(D * sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize
        for (int j = 0; j < d; j++) {
            znorm[j] = (xi[j] - mean[j]) / std[j];
        }

        // Quadratic expansion
        memcpy(zexp, znorm, d * sizeof(float));
        int idx = d;
        for (int u = 0; u < d; u++) {
            for (int v = u + 1; v < d; v++) {
                zexp[idx++] = znorm[u] * znorm[v];
            }
        }

        // Linear classification
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < D; f++) {
            float zf = zexp[f];
            if (zf == 0.0f) continue;
            const float *wf = W + (size_t)f * C;
            for (int c = 0; c < C; c++) scores[c] += zf * wf[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(znorm); free(zexp); free(scores);
    free(wb); if (x) free(x);
    return 0;
}
```