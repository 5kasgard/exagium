```hypothesis
We propose a factorization machine family "label-fm" that adds pairwise feature interactions via low-rank embeddings to a linear model. The adult dataset has many categorical features where interactions are known to be important (e.g., education × occupation). FM captures these with O(d·K) parameters and inference cost, beating linear (0.8524) while staying far cheaper than MLP16 (2325 Ir/item). If dev accuracy does not exceed 0.855, the interactions are not helping enough.
```
```python
"""Factorization machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Model: linear logits + per-class FM interaction (embedding dim K).
Exported as per-byte lookup tables: magic 0x464D0001, d, K, C; float32 T_lin[d][256][C], T_emb[d][256][C*K], bias[C]."""
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
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    # Build encoding matrices E_j (256 x m_j) and expanded feature matrix Z (n x M)
    enc = []
    offsets = [0]
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None]  # 256 x 1
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
        offsets.append(offsets[-1] + E.shape[1])
    M = offsets[-1]
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n = Z.shape[0]

    # Parameters: W_lin (M x C), W_emb (M x C*K), bias (C)
    W_lin = (rng.uniform(-1, 1, (M, C)) * math.sqrt(6.0 / M)).astype(np.float32)
    W_emb = (rng.uniform(-1, 1, (M, C * K)) * math.sqrt(6.0 / (M * K))).astype(np.float32)
    bias = np.zeros(C, np.float32)
    params = [W_lin, W_emb, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            bsz = len(yy)

            # Forward
            lin_logits = z @ W_lin + bias  # (bsz, C)
            logits = lin_logits.copy()
            emb_sums = []  # list of (bsz, K) per class
            sqnorms = []   # list of (M,) per class
            for c in range(C):
                W_emb_c = W_emb[:, c*K:(c+1)*K]  # (M, K)
                emb_sum = z @ W_emb_c  # (bsz, K)
                emb_sums.append(emb_sum)
                sqnorm = np.sum(W_emb_c * W_emb_c, axis=1)  # (M,)
                sqnorms.append(sqnorm)
                emb_sq_sum = (z * z) @ sqnorm  # (bsz,)
                interaction = 0.5 * (np.sum(emb_sum * emb_sum, axis=1) - emb_sq_sum)
                logits[:, c] += interaction

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yy] -= 1.0
            dlogits = (probs / bsz).astype(np.float32)  # (bsz, C)

            # Gradients
            grad_W_lin = z.T @ dlogits + L2 * W_lin
            grad_bias = dlogits.sum(axis=0)
            grad_W_emb = np.zeros_like(W_emb)
            for c in range(C):
                dL_dinter = dlogits[:, c]  # (bsz,)
                emb_sum = emb_sums[c]       # (bsz, K)
                sqnorm = sqnorms[c]         # (M,)
                # dL/dS = dL/dinter[:,None] * emb_sum
                dL_dS = dL_dinter[:, None] * emb_sum  # (bsz, K)
                # dL/dQ = -0.5 * dL/dinter
                dL_dQ = -0.5 * dL_dinter  # (bsz,)
                # Gradient from S: Z.T @ dL_dS
                W_emb_c = W_emb[:, c*K:(c+1)*K]
                grad_W_emb[:, c*K:(c+1)*K] += z.T @ dL_dS
                # Gradient from Q: dL/dsqnorm = (Z^2).T @ dL_dQ
                dL_dsqnorm = (z * z).T @ dL_dQ  # (M,)
                grad_W_emb[:, c*K:(c+1)*K] += 2.0 * W_emb_c * dL_dsqnorm[:, None]
            grad_W_emb += L2 * W_emb

            grads = [grad_W_lin, grad_W_emb, grad_bias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build lookup tables for inference
    T_lin = []
    T_emb = []
    for j, E in enumerate(enc):
        off = offsets[j]
        T_lin.append((E.astype(np.float32) @ W_lin[off:offsets[j+1]]).astype(np.float32))
        T_emb.append((E.astype(np.float32) @ W_emb[off:offsets[j+1]]).astype(np.float32))
    T_lin = np.stack(T_lin)  # (d, 256, C)
    T_emb = np.stack(T_emb)  # (d, 256, C*K)

    # Write weights.bin
    magic = 0x464D0001
    header = struct.pack("<4i", magic, d, K, C)
    out = header + T_lin.astype("<f4").tobytes() + T_emb.astype("<f4").tobytes() + bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class index (argmax of two logits). */
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
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + (long)d * 256 * C * K + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T_lin = (const float *)(hd + 4);
    const float *T_emb = T_lin + (long)d * 256 * C;
    const float *bias = T_emb + (long)d * 256 * C * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float logits[2];
    float emb_sum[2][16];  // K <= 16 in practice

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        logits[0] = bias[0];
        logits[1] = bias[1];
        for (int c = 0; c < C; c++) for (int kk = 0; kk < K; kk++) emb_sum[c][kk] = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_ptr = T_lin + ((long)j * 256 + v) * C;
            const float *emb_ptr = T_emb + ((long)j * 256 + v) * C * K;
            for (int c = 0; c < C; c++) {
                logits[c] += lin_ptr[c];
                for (int kk = 0; kk < K; kk++) {
                    emb_sum[c][kk] += emb_ptr[c * K + kk];
                }
            }
        }

        for (int c = 0; c < C; c++) {
            float sum = 0.0f, sum_sq = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                float e = emb_sum[c][kk];
                sum += e;
                sum_sq += e * e;
            }
            logits[c] += 0.5f * (sum * sum - sum_sq);
        }

        unsigned char best = (logits[1] > logits[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```