```hypothesis
Family: label-poly2 (degree-2 polynomial logistic regression).
Why: Optdigits pixels exhibit local correlations; quadratic features capture pairwise interactions without a hidden layer, adding expressive power over linear while keeping inference fast (single dot product over expanded features). Training uses the same deterministic Adam recipe as label-linear.
Failure: If dev accuracy ≤ 0.9613 (label-linear) or CPU cost per item exceeds label-mlp16 (≈776 ns/item), the model does not improve the frontier.
```

```python
"""Polynomial logistic regression (degree 2) for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes each ordinal byte as z-scoped value, then expands to all monomials up to degree 2.
Trains multinomial logistic regression with Adam (lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n))).
Exports: magic 0x504F4C32, d, C; float32 mean[d], std[d]; float32 weight[C][D], bias[C] (D = d + d*(d+1)/2)."""
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
    D = d + d * (d + 1) // 2

    rng = np.random.default_rng(SEED)

    # Normalize each feature (ordinal)
    mean = X.mean(axis=0).astype(np.float64)
    std = X.std(axis=0).astype(np.float64)
    std[std == 0] = 1.0
    Xn = (X.astype(np.float64) - mean) / std  # (n, d)

    # Polynomial expansion: linear + quadratic (i <= j)
    n = Xn.shape[0]
    Z = np.empty((n, D), dtype=np.float32)
    Z[:, :d] = Xn.astype(np.float32)
    col = d
    for i in range(d):
        for j in range(i, d):
            Z[:, col] = Xn[:, i] * Xn[:, j]
            col += 1

    # Initialize weights
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, yy):
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
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
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, C; mean[d], std[d]; weight[C][D] (row-major), bias[C]
    mean_f32 = mean.astype("<f4")
    std_f32 = std.astype("<f4")
    W_f32 = W.T.astype("<f4")  # (C, D)
    b_f32 = b.astype("<f4")
    header = struct.pack("<3i", MAGIC, d, C)
    out = header + mean_f32.tobytes() + std_f32.tobytes() + W_f32.tobytes() + b_f32.tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Polynomial logistic regression (degree 2) inference: ./infer weights.bin in.rec N
 * Input record: d bytes (ordinal). Output: one byte class index (argmax).
 * Weights format: magic 0x504F4C32, int32 d, int32 C; float32 mean[d], std[d];
 * float32 weight[C][D] (row-major, D = d + d*(d+1)/2); float32 bias[C]. */
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
    if (nw < 12) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hdr = (int32_t *)wb;
    if (hdr[0] != 0x504F4C32) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], C = hdr[2];
    if (d <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    int D = d + d * (d + 1) / 2;
    long expected = 12L + 4L * d * 2 + 4L * C * D + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(wb + 12);
    const float *std = mean + d;
    const float *weight = std + d;           // [C][D] row-major
    const float *bias = weight + (long)C * D;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xnorm = malloc(d * sizeof(float));
    float *quad = malloc((D - d) * sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize
        for (int j = 0; j < d; j++) {
            float v = xi[j];
            float s = std[j];
            xnorm[j] = (s != 0.0f) ? (v - mean[j]) / s : 0.0f;
        }

        // Quadratic features (i <= j)
        int qidx = 0;
        for (int i1 = 0; i1 < d; i1++) {
            for (int i2 = i1; i2 < d; i2++) {
                quad[qidx++] = xnorm[i1] * xnorm[i2];
            }
        }

        // Compute scores
        for (int c = 0; c < C; c++) {
            const float *wc = weight + (long)c * D;
            float sum = bias[c];
            // Linear part
            for (int j = 0; j < d; j++) sum += xnorm[j] * wc[j];
            // Quadratic part
            for (int q = 0; q < D - d; q++) sum += quad[q] * wc[d + q];
            scores[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```