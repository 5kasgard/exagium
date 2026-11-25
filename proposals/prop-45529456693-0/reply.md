```hypothesis
Family: label-fm (Factorization Machine with per-field embeddings).
Why: Adult benefits from pairwise feature interactions (e.g., education×occupation). A factorization machine models all pairwise interactions with O(d·k) parameters (d=14 fields, k=8 factors) and computes them in O(d·k) time using the (∑v)²−∑v² trick. This adds expressive power over label-linear while keeping inference extremely fast (≈14·8·2 = 224 multiply-adds per item) and weights small (~260 KB). We treat all 14 input bytes as categorical indices (0–255) with learned embeddings, avoiding the ordinal/categorical split.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or Ir/item ≥ 1000 (label-linear is 719), the model does not meet the goal.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encoding: each of the 14 input bytes is a categorical index 0..255.
Model:  y_c = b_c + ∑_j W_c[j, x_j] + ½ ∑_f ( (∑_j V_c[j, x_j, f])² - ∑_j V_c[j, x_j, f]² )
Exported as: magic 0x464D5F31, d, k, C; then per-class [bias, W[d*256], V[d*256*k]] (float32 LE)."""
import json, math, struct, sys

import numpy as np

SEED = 1
K = 8                 # factor dimension
LR = 1e-2
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 50

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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Parameters per class: bias (1), W (d*256), V (d*256*K)
    bias = np.zeros(C, np.float32)
    W = (rng.normal(0, 0.01, (C, d * 256)).astype(np.float32))
    V = (rng.normal(0, 0.01, (C, d * 256, K)).astype(np.float32))

    # Adam state
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Linear term: (B, C)
            lin = bias + np.sum(W[:, xb.T].transpose(1, 2, 0), axis=1)   # W[:, xb.T] -> (C, d, B) -> (B, d, C) -> sum over d -> (B, C)

            # Factor term: for each class, sum V over d, then 0.5*(sum^2 - sum sq)
            # V shape (C, d*256, K). xb.T (d, B) -> indices.
            V_sel = V[:, xb.T]   # (C, d, B, K)
            V_sum = V_sel.sum(axis=1)                 # (C, B, K)
            V_sq_sum = (V_sel * V_sel).sum(axis=1)    # (C, B, K)
            inter = 0.5 * ((V_sum * V_sum - V_sq_sum).sum(axis=2)).T   # (B, C)

            logits = lin + inter                       # (B, C)
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad = probs / B                           # (B, C)

            # Gradients
            g_bias = grad.sum(axis=0)                  # (C,)
            # W grad: (C, d*256)
            g_W = np.zeros_like(W)
            for c in range(C):
                np.add.at(g_W[c], xb.T.ravel(), grad[:, c].repeat(d))
            # V grad: (C, d*256, K)
            g_V = np.zeros_like(V)
            # For each class, each factor f: ∂inter/∂V_{j,f} = (V_sum_f - V_{j,f}) * grad_c
            # We can compute by looping over d (14) which is small.
            for j in range(d):
                v_j = V_sel[:, j]          # (C, B, K)
                v_sum_j = V_sum            # (C, B, K)
                # grad_inter for this j: (v_sum_j - v_j) * grad[:, :, None]  (C, B, K)
                g_inter_j = (v_sum_j - v_j) * grad.T[:, :, None]   # (C, B, K)
                # accumulate into g_V[:, j*256 + xb[:, j], :]
                for c in range(C):
                    np.add.at(g_V[c], xb[:, j] + j * 256, g_inter_j[c])

            # L2 regularization
            g_W += L2 * W
            g_V += L2 * V

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            bias -= LR * (m_bias / (1 - beta1 ** t)) / (np.sqrt(v_bias / (1 - beta2 ** t)) + eps)
            # W
            m_W = beta1 * m_W + (1 - beta1) * g_W
            v_W = beta2 * v_W + (1 - beta2) * g_W * g_W
            W -= LR * (m_W / (1 - beta1 ** t)) / (np.sqrt(v_W / (1 - beta2 ** t)) + eps)
            # V
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    # Pack weights
    magic = 0x464D5F31  # "FM_1"
    header = struct.pack("<4i", magic, d, K, C)
    out = header
    for c in range(C):
        out += struct.pack("<f", bias[c])
        out += W[c].astype("<f4").tobytes()
        out += V[c].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D5F31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || K < 1 || K > 64 || C < 1 || C > 256) { fprintf(stderr, "bad header\n"); return 3; }

    /* weights layout per class: bias (1), W[d*256], V[d*256*K] */
    long per_class = 1 + (long)d * 256 + (long)d * 256 * K;
    long expected = 16 + (long)C * per_class * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *base = (const float *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *V_sum = malloc(sizeof(float) * K);
    float *V_sq_sum = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) {
            const float *class_base = base + c * per_class;
            float bias = class_base[0];
            const float *W = class_base + 1;
            const float *V = W + (long)d * 256;

            /* linear term */
            float lin = bias;
            for (int j = 0; j < d; j++) lin += W[j * 256 + xi[j]];

            /* factor term: 0.5 * sum_f ( (sum_j V_{j,f})^2 - sum_j V_{j,f}^2 ) */
            memset(V_sum, 0, K * sizeof(float));
            memset(V_sq_sum, 0, K * sizeof(float));
            for (int j = 0; j < d; j++) {
                const float *v = V + ((long)j * 256 + xi[j]) * K;
                for (int f = 0; f < K; f++) {
                    float val = v[f];
                    V_sum[f] += val;
                    V_sq_sum[f] += val * val;
                }
            }
            float inter = 0.0f;
            for (int f = 0; f < K; f++) {
                float s = V_sum[f];
                inter += 0.5f * (s * s - V_sq_sum[f]);
            }
            scores[c] = lin + inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(V_sum); free(V_sq_sum); free(scores);
    return 0;
}
```