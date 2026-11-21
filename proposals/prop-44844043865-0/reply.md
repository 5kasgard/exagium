```hypothesis
Family: label-fm (Factorization Machine with per-field embeddings).
Why: FMs capture pairwise feature interactions crucial for tabular data while staying extremely fast at inference (only 91·E dot products per class). With E=8 the model has ~65k parameters, trains in seconds, and should surpass the linear baseline (which has no interactions) while approaching GBDT accuracy at a fraction of the inference cost.
Failure signal: dev accuracy ≤ 0.8524 (label-linear) or inference cost >2× label-linear.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x464D4C31, d, E, C; float32 linear[C][d][256], emb[C][d][256][E], bias[C]."""
import json, math, struct, sys

import numpy as np

E = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100


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
    # Parameters per class
    linear = rng.normal(0, 0.01, (C, d, 256)).astype(np.float32)
    emb = rng.normal(0, 0.01, (C, d, 256, E)).astype(np.float32)
    bias = np.zeros(C, np.float32)

    params = [linear, emb, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward for both classes
            scores = np.zeros((B, C), np.float32)
            sum_v = np.zeros((B, C, E), np.float32)      # sum of embeddings per sample per class
            sum_v_sq = np.zeros((B, C, E), np.float32)   # sum of squares
            linear_part = np.zeros((B, C), np.float32)

            for c in range(C):
                # linear part
                for j in range(d):
                    linear_part[:, c] += linear[c, j, xb[:, j]]
                # embeddings
                embs = np.stack([emb[c, j, xb[:, j]] for j in range(d)], axis=1)  # (B, d, E)
                sum_v[:, c] = embs.sum(axis=1)
                sum_v_sq[:, c] = (embs ** 2).sum(axis=1)
                interaction = 0.5 * ((sum_v[:, c] ** 2 - sum_v_sq[:, c]).sum(axis=1))
                scores[:, c] = bias[c] + linear_part[:, c] + interaction

            # Softmax and cross-entropy gradient
            max_s = scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_scores = probs / B  # (B, C)

            # Gradients
            g_linear = np.zeros_like(linear)
            g_emb = np.zeros_like(emb)
            g_bias = np.zeros_like(bias)

            for c in range(C):
                g = grad_scores[:, c]  # (B,)
                g_bias[c] = g.sum()
                # linear gradients
                for j in range(d):
                    np.add.at(g_linear[c, j], xb[:, j], g)
                # embedding gradients
                # For each feature j: grad = g[:, None] * (sum_v[:, c] - emb_vector)
                for j in range(d):
                    emb_vec = emb[c, j, xb[:, j]]          # (B, E)
                    diff = sum_v[:, c] - emb_vec           # (B, E)
                    grad_emb = g[:, None] * diff           # (B, E)
                    np.add.at(g_emb[c, j], xb[:, j], grad_emb)

            # L2 regularization
            g_linear += L2 * linear
            g_emb += L2 * emb

            grads = [g_linear, g_emb, g_bias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    header = struct.pack("<4i", 0x464D4C31, d, E, C)
    body = linear.astype("<f4").tobytes() + emb.astype("<f4").tobytes() + bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class index (argmax, first max wins). */
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
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || E < 1) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L
        + 4L * C * d * 256
        + 4L * C * d * 256 * E
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *lin = (const float *)(hd + 4);
    const float *emb = lin + C * d * 256;
    const float *bias = emb + C * d * 256 * E;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(C * E * sizeof(float));
    float *sum_v_sq = malloc(C * E * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float best_score = -1e30f;
        unsigned char best_class = 0;

        for (int c = 0; c < C; c++) {
            float score = bias[c];
            const float *lin_c = lin + c * d * 256;
            const float *emb_c = emb + c * d * 256 * E;

            // linear part
            for (int j = 0; j < d; j++) score += lin_c[j * 256 + xi[j]];

            // interaction part: 0.5 * ( (sum_v)^2 - sum(v^2) )
            for (int e = 0; e < E; e++) { sum_v[c * E + e] = 0.0f; sum_v_sq[c * E + e] = 0.0f; }
            for (int j = 0; j < d; j++) {
                const float *v = emb_c + (j * 256 + xi[j]) * E;
                for (int e = 0; e < E; e++) {
                    float ve = v[e];
                    sum_v[c * E + e] += ve;
                    sum_v_sq[c * E + e] += ve * ve;
                }
            }
            float inter = 0.0f;
            for (int e = 0; e < E; e++) {
                float sv = sum_v[c * E + e];
                inter += sv * sv - sum_v_sq[c * E + e];
            }
            score += 0.5f * inter;

            if (score > best_score) { best_score = score; best_class = (unsigned char)c; }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```