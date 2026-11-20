```hypothesis
We propose a Factorization Machine family "label-fm" that models pairwise pixel interactions via low-rank latent vectors (k=4 per class). On 8x8 digit images, local pixel correlations are important; FM captures these with only ~3300 parameters, staying fast. If dev accuracy does not beat label-linear (0.9613) or reach near label-mlp16 (0.9770), the added interaction complexity is not justified.
```

```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Model: score_c = bias_c + sum_i W_lin[i,c] * x_i
             + 0.5 * (sum_f (sum_i V[i,f,c] * x_i)^2 - sum_i sum_f (V[i,f,c] * x_i)^2)
Features standardized (z-score) using training mean/std.
Adam lr=0.01, L2=1e-4, batch=64, epochs=min(200, ceil(600000/n)).
Weights format: magic 0x464D4C31, d, C, k; float32 mean[d], std[d], bias[C], W_lin[d*C], V[d*k*C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
K = 4
MAX_EPOCHS = 200
SAMPLES = 600000


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
    d = X_bytes.shape[1]
    n = X_bytes.shape[0]

    rng = np.random.default_rng(SEED)

    # Standardize features
    X = X_bytes.astype(np.float32)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    X = (X - mean) / std

    # Initialize parameters
    bias = np.zeros(C, np.float32)
    W_lin = rng.normal(0, 0.01, (d, C)).astype(np.float32)
    V = rng.normal(0, 0.01, (d, K, C)).astype(np.float32)

    params = [bias, W_lin, V]
    # Adam state: m, v for each param
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X[idx]          # (B, d)
            yy = y[idx]         # (B,)
            B = len(yy)

            # Forward
            linear = x @ W_lin                     # (B, C)
            # S = x @ V per factor: (B, K, C)
            S = np.empty((B, K, C), np.float32)
            for f in range(K):
                S[:, f, :] = x @ V[:, f, :]
            # Q = sum_i sum_f (V[i,f,c] * x_i)^2
            V_sq = V ** 2
            x_sq = x ** 2
            Q = np.zeros((B, C), np.float32)
            for f in range(K):
                Q += x_sq @ V_sq[:, f, :]
            interaction = 0.5 * ((S ** 2).sum(axis=1) - Q)  # (B, C)
            scores = bias + linear + interaction              # (B, C)

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            grad_scores = probs / B   # (B, C)

            # Gradients
            dbias = grad_scores.sum(axis=0).astype(np.float32)
            dW_lin = (x.T @ grad_scores).astype(np.float32)
            dV = np.zeros_like(V)
            for f in range(K):
                S_f = S[:, f, :]                 # (B, C)
                V_f = V[:, f, :]                 # (d, C)
                # termA = x.T @ (grad_scores * S_f)
                termA = (x.T @ (grad_scores * S_f)).astype(np.float32)
                # termB = V_f * (x**3).T @ grad_scores
                x_cube = x ** 3
                termB = (V_f * (x_cube.T @ grad_scores)).astype(np.float32)
                dV[:, f, :] = termA - termB

            grads = [dbias, dW_lin, dV]

            # L2 regularization
            grads[1] += L2 * W_lin
            grads[2] += L2 * V

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Pack weights
    # magic, d, C, k
    header = struct.pack("<4i", 0x464D4C31, d, C, K)
    body = (
        mean.astype("<f4").tobytes() +
        std.astype("<f4").tobytes() +
        bias.astype("<f4").tobytes() +
        W_lin.astype("<f4").tobytes() +
        V.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class (argmax).
 * Weights format: magic 0x464D4C31, int32 d, C, k; float32 mean[d], std[d], bias[C], W_lin[d*C], V[d*k*C]. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], K = hd[3];
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16 + 4L * (2L * d + C + (long)d * C + (long)d * K * C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *bias = std + d;
    const float *W_lin = bias + C;
    const float *V = W_lin + (long)d * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *scores = malloc(sizeof(float) * C);
    float *S = malloc(sizeof(float) * K * C);
    float *Q = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Standardize and compute linear term
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int j = 0; j < d; j++) {
            float xj = (xi[j] - mean[j]) / std[j];
            for (int c = 0; c < C; c++) scores[c] += xj * W_lin[(long)j * C + c];
        }

        // Interaction term
        for (int c = 0; c < C; c++) {
            for (int f = 0; f < K; f++) S[f * C + c] = 0.0f;
            Q[c] = 0.0f;
        }
        for (int j = 0; j < d; j++) {
            float xj = (xi[j] - mean[j]) / std[j];
            float xj2 = xj * xj;
            for (int f = 0; f < K; f++) {
                for (int c = 0; c < C; c++) {
                    float v = V[((long)j * K + f) * C + c];
                    float t = v * xj;
                    S[f * C + c] += t;
                    Q[c] += t * t;
                }
            }
        }
        for (int c = 0; c < C; c++) {
            float sum_sq = 0.0f;
            for (int f = 0; f < K; f++) {
                float s = S[f * C + c];
                sum_sq += s * s;
            }
            scores[c] += 0.5f * (sum_sq - Q[c]);
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    return 0;
}
```