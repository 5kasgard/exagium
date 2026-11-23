```hypothesis
Family: label-quad (per-class rank-1 quadratic model). The 8x8 digit pixels have strong pairwise interactions (strokes, loops) that a linear model misses. Adding a single squared projection per class captures key non-linearities with only 2× the parameters of label-linear. Inference stays fast: two dot products per class plus a square. If dev accuracy ≤ 96.13% (label-linear), the quadratic terms do not help or overfit on this small dataset.
```
```python
"""label-quad: per-class rank-1 quadratic model for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Model: score_c = b_c + z^T w_c + (z^T u_c)^2, where z is per-feature z-scored byte value.
Exported as lookup tables: int32 magic 0x51554144, d, 0, C; float32 T[d][256][2*C], bias[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x51554144  # "QUAD"

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
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # Per-feature z-scoring (ordinal only per card)
    mu = X.mean(axis=0).astype(np.float64)
    sd = X.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    # Precompute z-scored value for each byte 0..255 per feature
    Z_table = np.empty((d, 256), dtype=np.float32)
    for j in range(d):
        Z_table[j] = ((np.arange(256, dtype=np.float64) - mu[j]) / sd[j]).astype(np.float32)

    # Encode training data
    Z = Z_table[np.arange(d), X.T].T  # (n, d)

    # Initialize weights
    W = (rng.uniform(-1, 1, (d, C)) * math.sqrt(6.0 / d)).astype(np.float32)
    U = (rng.uniform(-1, 1, (d, C)) * math.sqrt(6.0 / d)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, U, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    n = Z.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            m = len(yy)
            # Forward
            linear = z @ W + b          # (m, C)
            quad = z @ U                # (m, C)
            scores = linear + quad * quad
            # Softmax + gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores, dtype=np.float32)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(m), yy] -= 1.0
            probs /= m  # (m, C)
            # Gradients
            dL_dlinear = probs.astype(np.float32)
            dL_dquad = (2.0 * quad * probs).astype(np.float32)
            grad_W = z.T @ dL_dlinear + L2 * W
            grad_U = z.T @ dL_dquad + L2 * U
            grad_b = dL_dlinear.sum(axis=0)
            grads = [grad_W, grad_U, grad_b]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m1, v1 = state[i]
                m1 *= 0.9; m1 += 0.1 * gr
                v1 *= 0.999; v1 += 0.001 * gr * gr
                m_hat = m1 / (1 - 0.9 ** t)
                v_hat = v1 / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build lookup tables T[j][v][2*c] = W[j,c]*Z_table[j,v], T[j][v][2*c+1] = U[j,c]*Z_table[j,v]
    T = np.empty((d, 256, 2 * C), dtype=np.float32)
    for j in range(d):
        zj = Z_table[j][:, None]  # (256,1)
        T[j, :, 0::2] = zj * W[j, :]  # broadcast (256,1) * (C,) -> (256,C) placed at even indices
        T[j, :, 1::2] = zj * U[j, :]

    # Pack weights
    out = struct.pack("<4i", MAGIC, d, 0, C)
    out += T.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-quad inference: ./infer weights.bin in.rec N
 * Model: score_c = bias_c + sum_j (W[j,c]*z_j) + (sum_j (U[j,c]*z_j))^2
 * Weights format: int32 magic(0x51554144), d, 0, C; float32 T[d][256][2*C]; float32 bias[C].
 * Each input record: d bytes. Output: one byte class index per record. */
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
    if (nw < 16 || hd[0] != 0x51554144) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (H != 0 || C < 1 || C > 256) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16L + 4L * d * 256 * 2 * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *bias = T + (long)d * 256 * 2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lin = calloc(C, sizeof(float));
    float *quad = calloc(C, sizeof(float));
    if (!lin || !quad) { fprintf(stderr, "oom\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) { lin[c] = bias[c]; quad[c] = 0.0f; }
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((long)j * 256 + xi[j]) * 2 * C;
            for (int c = 0; c < C; c++) {
                lin[c]  += tj[2*c];
                quad[c] += tj[2*c + 1];
            }
        }
        unsigned char best = 0;
        float best_score = -1e30f;
        for (int c = 0; c < C; c++) {
            float score = lin[c] + quad[c] * quad[c];
            if (score > best_score) { best_score = score; best = (unsigned char)c; }
        }
        put_rec(stdout, &best, 1);
    }
    free(lin); free(quad);
    return 0;
}
```