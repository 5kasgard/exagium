```hypothesis
Family: label-fm (Factorization Machine, degree-2 polynomial network).
Why: Character prediction relies on pairwise context interactions (e.g., "t"+"h"→"e", "q"+"u"→"e").
FM explicitly models all 16×15/2 position pairs via low-rank embeddings (k=16), adding only O(d·k+k·C)
params over linear. Inference stays table-lookup fast: 16 embedding fetches + O(k) arithmetic.
Failure signal: dev accuracy ≤ 0.3375 (label-linear) or inference cost ≥ 2× label-linear.
```
```python
"""Factorization Machine (degree-2) for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Weights format (little-endian):
  int32 magic=0x464D3132, d, k, C
  float32 Linear[d][256][C]
  float32 Emb[d][256][k]
  float32 V2[k][C]
  float32 Bias[C]"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
K = 16            # embedding dimension for pairwise interactions
SEED = 1
LR = 0.01
L2 = 1e-5
BATCH = 256
EPOCHS = 30
SAMPLES = 600000  # max samples per epoch for epoch count

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    """Return gradient of softmax cross-entropy w.r.t. logits: (P - one_hot) / B."""
    B = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # Tuning split: last 20% in file order
    split = int(n * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]
    n_tr = X_tr.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Linear: (d, 256, C)
    Linear = (rng.normal(0, 0.01, (d, 256, C)) / math.sqrt(C)).astype(np.float32)
    # Emb: (d, 256, K)
    Emb = (rng.normal(0, 0.01, (d, 256, K)) / math.sqrt(K)).astype(np.float32)
    # V2: (K, C)
    V2 = (rng.normal(0, 0.01, (K, C)) / math.sqrt(C)).astype(np.float32)
    # Bias: (C,)
    Bias = np.zeros(C, np.float32)

    params = [Linear, Emb, V2, Bias]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    max_epochs = min(EPOCHS, math.ceil(SAMPLES / n_tr))
    for epoch in range(max_epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            xb = X_tr[idx]          # (B, d)
            yb = y_tr[idx]          # (B,)
            B = xb.shape[0]

            # --- Forward ---
            # Linear scores: Bias + sum_j Linear[j, xb[:,j], :]
            lin_scores = Bias.copy()
            for j in range(d):
                lin_scores += Linear[j, xb[:, j], :]

            # Interaction scores
            sum_emb = np.zeros((B, K), np.float32)
            sum_emb_sq = np.zeros((B, K), np.float32)
            for j in range(d):
                e = Emb[j, xb[:, j], :]      # (B, K)
                sum_emb += e
                sum_emb_sq += e * e
            interaction = 0.5 * (sum_emb * sum_emb - sum_emb_sq)   # (B, K)
            inter_scores = interaction @ V2                          # (B, C)

            scores = lin_scores + inter_scores                       # (B, C)

            # --- Loss gradient ---
            dscores = softmax_cross_entropy_grad(scores, yb)         # (B, C)

            # --- Backward ---
            # Bias
            dBias = dscores.sum(axis=0)

            # Linear
            dLinear = np.zeros_like(Linear)
            for j in range(d):
                np.add.at(dLinear[j], xb[:, j], dscores)

            # V2
            dV2 = interaction.T @ dscores                            # (K, C)

            # Interaction -> sum_emb, sum_emb_sq
            d_interaction = dscores @ V2.T                           # (B, K)
            d_sum_emb = d_interaction * sum_emb                      # (B, K)
            d_sum_emb_sq = -0.5 * d_interaction                      # (B, K)

            # Emb
            dEmb = np.zeros_like(Emb)
            for j in range(d):
                e = Emb[j, xb[:, j], :]                              # (B, K)
                grad_e = d_sum_emb + 2.0 * e * d_sum_emb_sq          # (B, K)
                np.add.at(dEmb[j], xb[:, j], grad_e)

            # L2 regularization
            dLinear += L2 * Linear
            dEmb += L2 * Emb
            dV2 += L2 * V2
            # No L2 on Bias typically

            grads = [dLinear, dEmb, dV2, dBias]

            # --- Adam update ---
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] *= beta1; m[i] += (1 - beta1) * gr
                v[i] *= beta2; v[i] += (1 - beta2) * gr * gr
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

        # Optional: validation accuracy monitoring (silent)

    # --- Export weights ---
    # Validate sizes
    expected = 16 + 4 * (d * 256 * C + d * 256 * K + K * C + C)
    out = struct.pack("<4i", 0x464D3132, d, K, C)
    out += Linear.astype("<f4").tobytes()
    out += Emb.astype("<f4").tobytes()
    out += V2.astype("<f4").tobytes()
    out += Bias.astype("<f4").tobytes()
    assert len(out) == expected, f"size mismatch {len(out)} != {expected}"
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (degree-2) inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: 1 byte class index (argmax, first max wins).
 * Weights format (little-endian):
 *   int32 magic=0x464D3132, d, k, C
 *   float32 Linear[d][256][C]
 *   float32 Emb[d][256][k]
 *   float32 V2[k][C]
 *   float32 Bias[C] */
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad dims\n"); return 3; }

    long expected = 16L + 4L * ((long)d * 256 * C + (long)d * 256 * K + (long)K * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *Linear = (const float *)(hd + 4);
    const float *Emb = Linear + (long)d * 256 * C;
    const float *V2 = Emb + (long)d * 256 * K;
    const float *Bias = V2 + (long)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_emb = malloc(sizeof(float) * K);
    float *sum_emb_sq = malloc(sizeof(float) * K);
    float *interaction = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* Linear part: Bias + sum_j Linear[j][xi[j]] */
        memcpy(scores, Bias, sizeof(float) * C);
        for (int j = 0; j < d; j++) {
            const float *row = Linear + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += row[c];
        }

        /* Interaction part: 0.5 * ((sum e)^2 - sum e^2) @ V2 */
        memset(sum_emb, 0, sizeof(float) * K);
        memset(sum_emb_sq, 0, sizeof(float) * K);
        for (int j = 0; j < d; j++) {
            const float *e = Emb + ((long)j * 256 + xi[j]) * K;
            for (int h = 0; h < K; h++) {
                float val = e[h];
                sum_emb[h] += val;
                sum_emb_sq[h] += val * val;
            }
        }
        for (int h = 0; h < K; h++)
            interaction[h] = 0.5f * (sum_emb[h] * sum_emb[h] - sum_emb_sq[h]);

        for (int c = 0; c < C; c++) {
            float acc = 0.0f;
            for (int h = 0; h < K; h++) acc += interaction[h] * V2[(long)h * C + c];
            scores[c] += acc;
        }

        /* Argmax (first max wins) */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(scores); free(sum_emb); free(sum_emb_sq); free(interaction);
    return 0;
}
```