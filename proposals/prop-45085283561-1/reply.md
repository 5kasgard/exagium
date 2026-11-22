```hypothesis
Family: label-fm (factorization machine with shared interactions).
Why: Optdigits pixels have local correlations; pairwise interactions via low-rank FM capture this better than linear model, with only ~900 params. Shared FM across classes keeps inference fast (O(d*k + d*C) per item).
Failure: If dev accuracy <= 0.9613 (label-linear) or CPU cost > 2x label-linear.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
K = 4  # latent dimension

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

    # Preprocessing: ordinal z-score
    means = X.mean(axis=0).astype(np.float64)
    stds = X.std(axis=0).astype(np.float64)
    stds[stds == 0] = 1.0
    # Normalized training data
    Z = ((X.astype(np.float64) - means) / stds).astype(np.float32)

    # Initialize parameters
    rng = np.random.default_rng(SEED)
    # Linear weights: (d, C)
    W = (rng.uniform(-1, 1, (d, C)) * math.sqrt(6.0 / d)).astype(np.float32)
    # Latent vectors: (d, K)
    V = (rng.uniform(-1, 1, (d, K)) * math.sqrt(6.0 / d)).astype(np.float32)
    # Bias: (C,)
    b = np.zeros(C, np.float32)

    params = [W, V, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = Z.shape[0]
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]          # (batch, d)
            yy = y[idx]         # (batch,)

            # Forward
            linear_scores = z @ W + b   # (batch, C)
            # FM interaction (shared)
            interaction = np.zeros(z.shape[0], np.float32)
            for f in range(K):
                vf = V[:, f]            # (d,)
                sum_vx = z @ vf         # (batch,)
                sum_vx2 = (z * z) @ (vf * vf)  # (batch,)
                interaction += 0.5 * (sum_vx * sum_vx - sum_vx2)
            scores = linear_scores + interaction[:, None]  # (batch, C)

            # Softmax gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_scores = (probs / len(yy)).astype(np.float32)  # (batch, C)

            # Gradients
            grad_W = z.T @ grad_scores + L2 * W
            grad_b = grad_scores.sum(axis=0)
            # Gradient for V: interaction gradient is sum over classes of grad_scores
            g_inter = grad_scores.sum(axis=1)  # (batch,)
            grad_V = np.zeros_like(V)
            for f in range(K):
                vf = V[:, f]
                sum_vx = z @ vf
                for i in range(d):
                    zi = z[:, i]
                    grad_V[i, f] = np.dot(g_inter, zi * (sum_vx - zi * vf[i]))
                grad_V[:, f] += L2 * vf

            grads = [grad_W, grad_V, grad_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build lookup tables for inference (little-endian float32)
    byte_vals = np.arange(256, dtype=np.float64)
    linear_table = np.zeros((d, 256, C), dtype=np.float32)
    fm_table = np.zeros((d, 256, K), dtype=np.float32)
    for i in range(d):
        x_norm = ((byte_vals - means[i]) / stds[i]).astype(np.float32)  # (256,)
        linear_table[i] = (x_norm[:, None] * W[i]).astype("<f4")   # (256, C)
        fm_table[i] = (x_norm[:, None] * V[i]).astype("<f4")       # (256, K)

    # Pack weights file
    magic = 0x464D4C31  # "FML1"
    header = struct.pack("<4i", magic, d, K, C)
    out = header + linear_table.tobytes() + fm_table.tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine family for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weights format: magic 0x464D4C31 ("FML1"), int32 d, K, C;
 * float32 linear_table[d][256][C], fm_table[d][256][K], bias[C]. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C + (long)d * 256 * K + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *linear_table = (const float *)(hd + 4);
    const float *fm_table = linear_table + (long)d * 256 * C;
    const float *bias = fm_table + (long)d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_vx = malloc(sizeof(float) * K);
    float *sum_vx2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Initialize scores with bias
        for (int c = 0; c < C; c++) scores[c] = bias[c];

        // Linear part
        for (int j = 0; j < d; j++) {
            const float *lin = linear_table + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += lin[c];
        }

        // FM interaction (shared)
        for (int f = 0; f < K; f++) {
            sum_vx[f] = 0.0f;
            sum_vx2[f] = 0.0f;
        }
        for (int j = 0; j < d; j++) {
            const float *fm = fm_table + ((long)j * 256 + xi[j]) * K;
            for (int f = 0; f < K; f++) {
                float v = fm[f];
                sum_vx[f] += v;
                sum_vx2[f] += v * v;
            }
        }
        float interaction = 0.0f;
        for (int f = 0; f < K; f++) {
            interaction += 0.5f * (sum_vx[f] * sum_vx[f] - sum_vx2[f]);
        }
        for (int c = 0; c < C; c++) scores[c] += interaction;

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```