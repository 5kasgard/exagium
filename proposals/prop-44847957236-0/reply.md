```hypothesis
Family: label-fm (factorization machine with shared embeddings).
Why: Adult has strong pairwise interactions (education × occupation, capital-gain × hours/week). A factorization machine captures all pairwise interactions in O(d·k) time/space (d=14, k=4) and adds only ~70 KB to the linear model. It should beat label-linear (0.8524) while staying far cheaper than label-mlp16 (2325 Ir/item).
Failure signal: dev accuracy ≤ 0.8524 or Ir/item > 1500.
```

```python
"""label-fm: factorization machine for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Weights format (little-endian):
  int32 magic=0x464D4C31, d, k, C
  float32 linear_weights[d*256*C]   # per feature, value, class
  float32 embeddings[d*256*k]       # per feature, value, factor
  float32 W[C*k]                    # projection from factors to classes
  float32 b[C]                      # class biases
Training: Adam, same schedule as label-linear, deterministic seed 1."""
import json, math, struct, sys

import numpy as np

K = 4
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Parameters
    linear_w = np.zeros((d, 256, C), dtype=np.float32)          # (d, 256, C)
    embed = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32) # (d, 256, K)
    W = rng.normal(0, math.sqrt(2.0 / K), (C, K)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    params = [linear_w, embed, W, b]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]                 # (B, d)
            yb = y[idx]                 # (B,)
            B = len(yb)

            # ---- Forward ----
            # Linear term: (B, C)
            lin_term = np.zeros((B, C), dtype=np.float32)
            for j in range(d):
                lin_term += linear_w[j, xb[:, j], :]

            # Embeddings: (B, d, K) -> sum over d -> (B, K)
            emb_vals = embed[:, xb.T, :].transpose(1, 0, 2)   # (B, d, K)
            sum_emb = emb_vals.sum(axis=1)                    # (B, K)
            sum_emb_sq = (emb_vals ** 2).sum(axis=1)          # (B, K)
            inter = 0.5 * (sum_emb ** 2 - sum_emb_sq)         # (B, K)

            logits = inter @ W.T + b + lin_term               # (B, C)

            # Softmax cross-entropy
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B                           # (B, C)

            # ---- Backward ----
            # Gradients for b, W
            grad_b = grad_logits.sum(axis=0)
            grad_W = grad_logits.T @ inter

            # Gradient for interaction
            grad_inter = grad_logits @ W                      # (B, K)

            # Gradient for sum_emb and sum_emb_sq
            grad_sum_emb = grad_inter * sum_emb
            grad_sum_emb_sq = -0.5 * grad_inter

            # Gradient for embeddings per feature
            grad_emb = np.zeros_like(embed)
            grad_lin = np.zeros_like(linear_w)

            for j in range(d):
                # Indices for this feature across batch
                idx_j = xb[:, j]                              # (B,)
                # Gradient for linear weights: (B, C) -> add to (256, C)
                np.add.at(grad_lin[j], idx_j, grad_logits)

                # Gradient for embeddings:
                # dL/dE = grad_sum_emb + 2 * grad_sum_emb_sq * E
                e_j = embed[j, idx_j, :]                      # (B, K)
                g_emb = grad_sum_emb + 2.0 * grad_sum_emb_sq * e_j
                np.add.at(grad_emb[j], idx_j, g_emb)

            # L2 regularization
            grad_lin += L2 * linear_w
            grad_emb += L2 * embed
            grad_W += L2 * W

            grads = [grad_lin, grad_emb, grad_W, grad_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # ---- Export ----
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += linear_w.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index. */
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
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }

    size_t expected = 16
        + (size_t)d * 256 * C * sizeof(float)   /* linear_w */
        + (size_t)d * 256 * K * sizeof(float)   /* embed */
        + (size_t)C * K * sizeof(float)         /* W */
        + (size_t)C * sizeof(float);            /* b */
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *linear_w = p; p += (size_t)d * 256 * C;
    const float *embed = p;    p += (size_t)d * 256 * K;
    const float *W = p;        p += (size_t)C * K;
    const float *b = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = calloc(K, sizeof(float));
    float *sum_emb_sq = calloc(K, sizeof(float));
    float *logits = malloc(C * sizeof(float));
    float *inter = malloc(K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* Reset accumulators */
        for (int c = 0; c < C; c++) logits[c] = b[c];
        for (int u = 0; u < K; u++) { sum_emb[u] = 0.0f; sum_emb_sq[u] = 0.0f; }

        /* Linear term + embedding sums */
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lw = linear_w + ((size_t)j * 256 + v) * C;
            for (int c = 0; c < C; c++) logits[c] += lw[c];

            const float *e = embed + ((size_t)j * 256 + v) * K;
            for (int u = 0; u < K; u++) {
                float val = e[u];
                sum_emb[u] += val;
                sum_emb_sq[u] += val * val;
            }
        }

        /* Interaction term */
        for (int u = 0; u < K; u++)
            inter[u] = 0.5f * (sum_emb[u] * sum_emb[u] - sum_emb_sq[u]);

        /* Project to classes */
        for (int c = 0; c < C; c++) {
            float acc = 0.0f;
            const float *w = W + (size_t)c * K;
            for (int u = 0; u < K; u++) acc += inter[u] * w[u];
            logits[c] += acc;
        }

        /* Argmax (first max wins) */
        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (logits[c] > logits[best]) best = (unsigned char)c;

        put_rec(stdout, &best, 1);
    }

    free(sum_emb); free(sum_emb_sq); free(logits); free(inter);
    return 0;
}
```