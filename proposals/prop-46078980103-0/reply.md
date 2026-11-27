```hypothesis
Family: label-fm (Factorization Machine on original 14 byte-columns with per-value embeddings).
Why: Captures pairwise feature interactions via low-rank factorization (rank k=8), crucial for tabular data where categorical combinations (e.g., education×occupation) matter. Adds only O(d·k) params vs O(d²) full interactions. Inference is O(d·(C+k)) ≈ 14·10 ops, faster than MLP/GBDT.
Failure signal: dev accuracy ≤ 0.8524 (label-linear) or Ir/item ≥ 1500 (2× label-linear).
```
```python
"""Factorization Machine for tabular label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, numpy only. Encodes ordinal z-scored, categorical one-hot per training values.
FM: w0[c] + sum_j w_j[x_j][c] + 0.5*sum_f((sum_j V_j[x_j][f])^2 - sum_j V_j[x_j][f]^2) for each class c.
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)). Rank k=8.
Exports: magic 0x464D3130, d, k, C; float32 w0[C], W[d][256][C], V[d][256][k] (row-major)."""
import json, math, struct, sys

import numpy as np

K = 8
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
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    rng = np.random.default_rng(SEED)

    # Build per-column lookup tables for linear weights and embeddings
    # W[j][v][c], V[j][v][f]
    W = np.zeros((d, 256, C), dtype=np.float32)
    V = rng.uniform(-0.01, 0.01, (d, 256, K)).astype(np.float32)
    w0 = np.zeros(C, dtype=np.float32)

    # For ordinal: we'll use the z-scored value as a scalar multiplier on a single weight/embedding per column
    # But the card says ordinal are 255 quantile bins (0-254), 255=unseen. So treat as categorical with 256 values.
    # Actually the description: "numeric columns as 255 training-quantile bins, categorical columns as category index (255 = unseen)"
    # So all 14 columns are already binned into 0-255. We can treat all as categorical with 256 possible values.
    # This matches the label-linear approach which uses 256 one-hot per column.
    # So we learn W[j][v][c] and V[j][v][f] for each column j, value v.

    n = X.shape[0]
    # Precompute nothing; we'll index directly in the loop

    params = [w0, W, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward_batch(x_batch):
        """x_batch: (B, d) uint8. Returns scores (B, C)"""
        B = x_batch.shape[0]
        scores = np.tile(w0, (B, 1))  # (B, C)
        # Linear part
        for j in range(d):
            v = x_batch[:, j]  # (B,)
            scores += W[j, v, :]  # (B, C) via broadcasting
        # FM part: 0.5 * sum_f [(sum_j V_j,v,f)^2 - sum_j V_j,v,f^2]
        sum_V = np.zeros((B, K), dtype=np.float32)
        sum_V_sq = np.zeros((B, K), dtype=np.float32)
        for j in range(d):
            v = x_batch[:, j]
            V_jv = V[j, v, :]  # (B, K)
            sum_V += V_jv
            sum_V_sq += V_jv * V_jv
        fm_term = 0.5 * np.sum(sum_V * sum_V - sum_V_sq, axis=1, keepdims=True)  # (B, 1)
        scores += fm_term  # broadcast to (B, C)
        return scores

    def softmax_cross_entropy_grad(scores, y_batch):
        """scores (B, C), y_batch (B,). Returns grad_scores (B, C)"""
        B = scores.shape[0]
        max_s = scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores - max_s)
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_batch] -= 1.0
        return probs / B

    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            scores = forward_batch(xb)
            g_scores = softmax_cross_entropy_grad(scores, yb)  # (B, C)

            # Gradients
            gw0 = g_scores.sum(axis=0)  # (C,)
            gW = np.zeros_like(W)
            gV = np.zeros_like(V)

            # Linear gradients
            for j in range(d):
                v = xb[:, j]  # (B,)
                # gW[j, v, c] += sum_b g_scores[b, c] for each v
                # Use np.add.at for unsorted segment sum
                for c in range(C):
                    np.add.at(gW[j, :, c], v, g_scores[:, c])

            # FM gradients
            # fm_term = 0.5 * sum_f (sum_V_f^2 - sum_V_sq_f)
            # d(fm_term)/d(V_j,v,f) = sum_V_f - V_j,v,f
            sum_V = np.zeros((xb.shape[0], K), dtype=np.float32)
            for j in range(d):
                v = xb[:, j]
                sum_V += V[j, v, :]
            # For each j, v: grad_V = (sum_V - V_j,v) * g_scores_sum_over_c
            # g_scores is (B, C), fm_term added to all C equally, so grad_fm = g_scores.sum(axis=1) (B,)
            grad_fm = g_scores.sum(axis=1)  # (B,)
            for j in range(d):
                v = xb[:, j]
                V_jv = V[j, v, :]  # (B, K)
                # gV[j, v, f] += sum_b grad_fm[b] * (sum_V[b, f] - V_jv[b, f])
                for f in range(K):
                    np.add.at(gV[j, :, f], v, grad_fm * (sum_V[:, f] - V_jv[:, f]))

            # L2 regularization
            gw0 += L2 * w0
            gW += L2 * W
            gV += L2 * V

            grads = [gw0, gW, gV]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<4i", 0x464D3130, d, K, C)
    out += w0.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (original columns). Output: 1 byte class index.
 * Weights: magic 0x464D3130, int32 d, K, C; float32 w0[C], W[d][256][C], V[d][256][K]. */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + (long)C + (long)d * 256 * C + (long)d * 256 * K;
    expected *= 4; // float32 bytes
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *w0 = p; p += C;
    const float *W = p; p += (long)d * 256 * C;
    const float *V = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_V = malloc(sizeof(float) * K);
    float *sum_V_sq = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear part
        for (int c = 0; c < C; c++) scores[c] = w0[c];
        for (int j = 0; j < d; j++) {
            const float *Wj = W + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += Wj[c];
        }

        // FM part
        for (int f = 0; f < K; f++) { sum_V[f] = 0.0f; sum_V_sq[f] = 0.0f; }
        for (int j = 0; j < d; j++) {
            const float *Vj = V + ((long)j * 256 + xi[j]) * K;
            for (int f = 0; f < K; f++) {
                float v = Vj[f];
                sum_V[f] += v;
                sum_V_sq[f] += v * v;
            }
        }
        float fm = 0.0f;
        for (int f = 0; f < K; f++) fm += 0.5f * (sum_V[f] * sum_V[f] - sum_V_sq[f]);
        for (int c = 0; c < C; c++) scores[c] += fm;

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```