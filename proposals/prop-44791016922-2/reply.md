```hypothesis
Family: label-fm (factorization machine with linear layer). Captures pairwise interactions between the 16 context bytes via low-rank embeddings (k=8), adding only ~10k ops/item vs linear's ~4k. Should beat label-linear (0.3375) on dev by modeling character n-gram interactions. Failure: if dev accuracy ≤ 0.34 or Ir/item > 20000.
```

```python
"""Factorization machine (degree-2 polynomial network) for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x464D5032, d, k, C, total_features, num_pairs;
float32 W_lin[d][256][C], b_lin[C], emb[d][256][k], W_int[num_pairs][C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters
K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x464D5032  # "FMP2"

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
    n = X.shape[0]
    P = d * (d - 1) // 2
    rng = np.random.default_rng(SEED)

    # Parameters
    # Linear weights: (d, 256, C)
    fan_in, fan_out = 256, C
    lim = math.sqrt(6.0 / (fan_in + fan_out))
    W_lin = rng.uniform(-lim, lim, (d, 256, C)).astype(np.float32)
    b_lin = np.zeros(C, np.float32)
    # Embeddings: (d, 256, K)
    lim_e = math.sqrt(6.0 / (256 + K))
    emb = rng.uniform(-lim_e, lim_e, (d, 256, K)).astype(np.float32)
    # Interaction weights: (P, C)
    lim_i = math.sqrt(6.0 / (P + C))
    W_int = rng.uniform(-lim_i, lim_i, (P, C)).astype(np.float32)

    params = [W_lin, b_lin, emb, W_int]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Precompute upper triangular indices for pairs
    triu_i, triu_j = np.triu_indices(d, k=1)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            # Linear scores
            scores = np.zeros((B, C), np.float32)
            for p in range(d):
                scores += W_lin[p, xb[:, p]]
            scores += b_lin

            # Embeddings for batch
            emb_batch = np.zeros((B, d, K), np.float32)
            for p in range(d):
                emb_batch[:, p] = emb[p, xb[:, p]]

            # Pairwise dot products
            gram = np.einsum('bik,bjk->bij', emb_batch, emb_batch)  # (B, d, d)
            pair_dots = gram[:, triu_i, triu_j]                     # (B, P)
            scores += pair_dots @ W_int

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            grad_scores = probs
            grad_scores[np.arange(B), yb] -= 1.0
            grad_scores /= B

            # Backprop
            # Gradient for W_int
            grad_W_int = pair_dots.T @ grad_scores

            # Gradient for pair_dots
            grad_pair_dots = grad_scores @ W_int.T  # (B, P)

            # Build symmetric matrix M (B, d, d)
            M = np.zeros((B, d, d), np.float32)
            M[:, triu_i, triu_j] = grad_pair_dots
            M[:, triu_j, triu_i] = grad_pair_dots

            # Gradient for emb_batch
            grad_emb_batch = np.einsum('bij,bjk->bik', M, emb_batch)  # (B, d, K)

            # Gradient for W_lin
            grad_W_lin = np.zeros_like(W_lin)
            for p in range(d):
                np.add.at(grad_W_lin[p], xb[:, p], grad_scores)
            grad_b_lin = grad_scores.sum(axis=0)

            # Scatter grad_emb_batch to emb
            grad_emb = np.zeros_like(emb)
            for p in range(d):
                np.add.at(grad_emb[p], xb[:, p], grad_emb_batch[:, p])

            # L2 regularization
            grad_W_lin += L2 * W_lin
            grad_W_int += L2 * W_int
            grad_emb += L2 * emb

            grads = [grad_W_lin, grad_b_lin, grad_emb, grad_W_int]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    total_features = d * 256
    header = struct.pack("<6i", MAGIC, d, K, C, total_features, P)
    out = header
    out += W_lin.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    out += W_int.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization machine (degree-2 polynomial network) for label tasks:
 * ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (nw < 24 || hd[0] != 0x464D5032) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3], total_features = hd[4], P = hd[5];
    if (d != 16 || total_features != d * 256 || P != d * (d - 1) / 2 || C < 1 || C > 256) {
        fprintf(stderr, "bad weights header\n"); return 3;
    }
    long expected = 24
        + 4L * total_features * C      // W_lin
        + 4L * C                       // b_lin
        + 4L * d * 256 * K             // emb
        + 4L * P * C;                  // W_int
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W_lin = (const float *)(hd + 6);
    const float *b_lin = W_lin + total_features * C;
    const float *emb = b_lin + C;
    const float *W_int = emb + d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *pair_dots = malloc(sizeof(float) * P);

    // Precompute pair indices
    int pair_i[120], pair_j[120];
    int pidx = 0;
    for (int i = 0; i < d; i++) for (int j = i + 1; j < d; j++) {
        pair_i[pidx] = i; pair_j[pidx] = j; pidx++;
    }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Linear part
        for (int c = 0; c < C; c++) scores[c] = b_lin[c];
        for (int p = 0; p < d; p++) {
            const float *w = W_lin + (p * 256 + xi[p]) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }

        // Pairwise interactions
        for (int p = 0; p < P; p++) {
            int pi = pair_i[p], pj = pair_j[p];
            float dot = 0.0f;
            const float *ei = emb + (pi * 256 + xi[pi]) * K;
            const float *ej = emb + (pj * 256 + xi[pj]) * K;
            for (int u = 0; u < K; u++) dot += ei[u] * ej[u];
            pair_dots[p] = dot;
        }
        for (int p = 0; p < P; p++) {
            float dot = pair_dots[p];
            const float *w = W_int + p * C;
            for (int c = 0; c < C; c++) scores[c] += dot * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```