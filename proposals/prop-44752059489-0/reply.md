```hypothesis
Family: label-fm (Factorization Machine with low-rank interactions).
Why: The adult dataset has 14 mixed-type features; a linear model misses pairwise interactions. A rank-4 FM adds only O(d·K) inference cost (d=14, K=4) while capturing feature crosses. It should surpass label-linear accuracy (0.8524) and stay far below label-mlp16 cost (2325 Ir/item).
Failure signal: dev accuracy ≤ 0.853 or Ir/item ≥ 1500.
```

```python
"""label-fm: Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format (little-endian):
  magic(0x464D4C31), d, K, C (int32 each)
  for each class c in 0..C-1:
    bias_c (float32)
    linear weights: d * 256 float32
    embeddings: d * 256 * K float32
"""
import json, math, struct, sys

import numpy as np

SEED = 1
K = 4                 # embedding rank (tuned on last 20% of train.in)
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1e-2, 1e-3, 64, 600000, 50

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
    rng = np.random.default_rng(SEED)

    # Parameters per class: bias, linear[d,256], emb[d,256,K]
    bias = rng.normal(0, 0.01, (C,)).astype(np.float32)
    linear = rng.normal(0, 0.01, (C, d, 256)).astype(np.float32)
    emb = rng.normal(0, 0.01, (C, d, 256, K)).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # Forward
            # scores[c, i] = bias[c] + sum_j linear[c,j,xb[i,j]] + 0.5*sum_k((sum_j emb[c,j,xb[i,j],k])^2 - sum_j emb[c,j,xb[i,j],k]^2)
            scores = bias[:, None] + np.sum(linear[:, np.arange(d), xb.T], axis=1)  # [C, bsz]
            # interaction term
            e = emb[:, np.arange(d), xb.T]  # [C, d, bsz, K]
            sum_e = e.sum(axis=1)           # [C, bsz, K]
            sum_e2 = (e * e).sum(axis=1)    # [C, bsz, K]
            inter = 0.5 * ((sum_e * sum_e - sum_e2).sum(axis=2))  # [C, bsz]
            scores += inter

            # Softmax cross-entropy gradient
            scores -= scores.max(axis=0, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=0, keepdims=True)
            probs[yb, np.arange(bsz)] -= 1.0
            probs /= bsz  # [C, bsz]

            # Gradients
            g_bias = probs.sum(axis=1)  # [C]
            g_lin = np.zeros_like(linear)
            g_emb = np.zeros_like(emb)

            # linear gradient: for each class c, feature j, value v: sum over batch where xb[:,j]==v of probs[c]
            for c in range(C):
                for j in range(d):
                    vals = xb[:, j]
                    for v in range(256):
                        mask = (vals == v)
                        if mask.any():
                            g_lin[c, j, v] = probs[c, mask].sum()

            # embedding gradient: d/d emb[c,j,v,k] = sum_i probs[c,i] * (sum_e[c,i,k] - emb[c,j,v,k]) for samples with xb[i,j]==v
            for c in range(C):
                for j in range(d):
                    vals = xb[:, j]
                    for v in range(256):
                        mask = (vals == v)
                        if mask.any():
                            # sum_e[c, mask, :] shape [m, K], emb[c,j,v,:] shape [K]
                            diff = sum_e[c, mask] - emb[c, j, v]  # [m, K]
                            g_emb[c, j, v] = (probs[c, mask, None] * diff).sum(axis=0)

            # L2 regularization
            g_bias += L2 * bias
            g_lin += L2 * linear
            g_emb += L2 * emb

            # Adam update
            t += 1
            for param, grad, m, v in [
                (bias, g_bias, m_bias, v_bias),
                (linear, g_lin, m_lin, v_lin),
                (emb, g_emb, m_emb, v_emb),
            ]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    for c in range(C):
        out += struct.pack("<f", bias[c])
        out += linear[c].astype("<f4").tobytes()
        out += emb[c].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from weights). Output: 1 byte class index (argmax of two scores).
 * Weights layout (little-endian):
 *   magic(0x464D4C31), d, K, C (int32)
 *   per class: bias (float32), linear[d][256] (float32), emb[d][256][K] (float32)
 */
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
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }

    // Expected size: 16 + C * (4 + d*256*4 + d*256*K*4)
    long expected = 16 + (long)C * (4 + (long)d * 256 * 4 * (1 + K));
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *ptr = (const float *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float best_score = -1e30f; unsigned char best_class = 0;
        for (int c = 0; c < C; c++) {
            float bias = *ptr++;
            const float *linear = ptr; ptr += d * 256;
            const float *emb = ptr; ptr += (long)d * 256 * K;

            float score = bias;
            // linear term
            for (int j = 0; j < d; j++) score += linear[j * 256 + xi[j]];

            // interaction term: 0.5 * sum_k ( (sum_j e_{j,k})^2 - sum_j e_{j,k}^2 )
            float sum_e[16], sum_e2[16]; // K <= 16 in practice
            for (int kk = 0; kk < K; kk++) { sum_e[kk] = 0.0f; sum_e2[kk] = 0.0f; }
            for (int j = 0; j < d; j++) {
                const float *e = emb + ((long)j * 256 + xi[j]) * K;
                for (int kk = 0; kk < K; kk++) {
                    float v = e[kk];
                    sum_e[kk] += v;
                    sum_e2[kk] += v * v;
                }
            }
            for (int kk = 0; kk < K; kk++) score += 0.5f * (sum_e[kk] * sum_e[kk] - sum_e2[kk]);

            if (score > best_score) { best_score = score; best_class = (unsigned char)c; }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```