```hypothesis
We propose label-fm, a factorization machine with class-specific embeddings (K=8). FMs capture pairwise feature interactions crucial for tabular data like Adult, and should outperform the linear baseline (0.8524) while keeping inference cost low (~250 Ir/item, similar to label-linear's 719). If dev accuracy does not exceed label-linear, or if inference cost exceeds label-mlp16 (2325 Ir/item), the hypothesis fails.
```
```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Exports per-byte lookup tables: magic 0x464D4C31, d, K, C; float32 linear[d][256][C], embed[d][256][K][C]."""
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
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # Build encoding matrices per feature
    enc = []
    dims = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)  # (256, 1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
        dims.append(E.shape[1])
    D = sum(dims)

    # Parameters: linear (D, C), embed (D, K, C)
    W_lin = np.zeros((D, C), np.float32)
    V = (rng.uniform(-1, 1, (D, K, C)) * math.sqrt(6.0 / D)).astype(np.float32)

    # Adam state
    m_lin, v_lin = np.zeros_like(W_lin), np.zeros_like(W_lin)
    m_emb, v_emb = np.zeros_like(V), np.zeros_like(V)
    t = 0

    # Precompute feature slices
    slices = []
    off = 0
    for dim in dims:
        slices.append(slice(off, off + dim))
        off += dim

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z_list = [enc[j][X[idx, j]] for j in range(d)]
            Z = np.concatenate(z_list, axis=1)  # (B, D)
            yy = y[idx]

            # Forward
            linear_scores = Z @ W_lin  # (B, C)
            interaction = np.zeros((len(yy), C), np.float32)
            S_list = []
            for c in range(C):
                Vc = V[:, :, c]  # (D, K)
                S = Z @ Vc  # (B, K)
                S_sq = (Z ** 2) @ (Vc ** 2)
                interaction[:, c] = 0.5 * np.sum(S * S - S_sq, axis=1)
                S_list.append(S)
            scores = linear_scores + interaction

            # Softmax cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_scores = (probs / len(yy)).astype(np.float32)  # (B, C)

            # Gradients
            grad_W_lin = Z.T @ grad_scores + L2 * W_lin
            grad_V = np.zeros_like(V)
            for c in range(C):
                g = grad_scores[:, c:c+1]  # (B, 1)
                S = S_list[c]
                Vc = V[:, :, c]
                # dL/dS = g * S
                dL_dS = g * S
                # dL/dS_sq = -0.5 * g
                dL_dS_sq = -0.5 * g
                # dL/dV = Z.T @ dL_dS + 2 * Vc * ((Z^2).T @ dL_dS_sq)
                grad_V[:, :, c] = Z.T @ dL_dS + 2.0 * Vc * ((Z ** 2).T @ dL_dS_sq) + L2 * Vc

            # Adam update
            t += 1
            for param, grad, m, v in [(W_lin, grad_W_lin, m_lin, v_lin), (V, grad_V, m_emb, v_emb)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Compress to per-byte lookup tables
    linear_tables = np.zeros((d, 256, C), np.float32)
    embed_tables = np.zeros((d, 256, K, C), np.float32)
    for j in range(d):
        sl = slices[j]
        linear_tables[j] = enc[j] @ W_lin[sl]
        for c in range(C):
            embed_tables[j, :, :, c] = enc[j] @ V[sl, :, c]

    # Write weights.bin
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += linear_tables.astype("<f4").tobytes()
    out += embed_tables.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class. */
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
    if (C < 1 || C > 256 || K < 1 || K > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * d * 256 * C + 4L * d * 256 * K * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float *embed = linear + (long)d * 256 * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *scores = malloc(sizeof(float) * C);
    float *sum_k = malloc(sizeof(float) * K);
    float *sum_sq_k = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = 0.0f;

        // Linear part
        for (int j = 0; j < d; j++) {
            const float *lin = linear + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += lin[c];
        }

        // Interaction part
        for (int c = 0; c < C; c++) {
            for (int kk = 0; kk < K; kk++) {
                sum_k[kk] = 0.0f;
                sum_sq_k[kk] = 0.0f;
            }
            for (int j = 0; j < d; j++) {
                const float *emb = embed + (((long)j * 256 + xi[j]) * K + 0) * C + c;
                for (int kk = 0; kk < K; kk++) {
                    float e = emb[kk * C];
                    sum_k[kk] += e;
                    sum_sq_k[kk] += e * e;
                }
            }
            for (int kk = 0; kk < K; kk++) {
                scores[c] += 0.5f * (sum_k[kk] * sum_k[kk] - sum_sq_k[kk]);
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(scores); free(sum_k); free(sum_sq_k);
    return 0;
}
```