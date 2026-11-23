```hypothesis
Factorization Machine (FM) with shared embeddings (k=8) and per-class linear terms captures pairwise feature interactions that the linear model misses. The Adult dataset has known interactions (e.g., education × occupation) that FM models well. Inference cost is O(d·k) ≈ 14·8 = 112 FMA per item, far below label-gbdt. If dev accuracy ≤ 0.8524 (label-linear) the hypothesis fails.
```
```python
"""Factorization Machine for tabular classification: shared embeddings + per-class linear.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: int32 magic 0x464D3132, d, k, C; float32 linear[C][d][256], emb[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 256, 30
MAGIC = 0x464D3132  # "FM12"

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

    # parameters
    linear = rng.normal(0, 0.01, (C, d, 256)).astype(np.float32)
    emb    = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)
    # Adam state
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb),    np.zeros_like(emb)
    t = 0

    idx = np.arange(n)
    for _ in range(EPOCHS):
        rng.shuffle(idx)
        for s in range(0, n, BATCH):
            batch_idx = idx[s:s+BATCH]
            xb = X[batch_idx]          # (B, d)
            yb = y[batch_idx]          # (B,)
            B = xb.shape[0]

            # gather embeddings: (B, d, K)
            e = emb[np.arange(d), xb.T].transpose(1, 0, 2)   # (B, d, K)
            S = e.sum(axis=1)                              # (B, K)
            S2 = (e*e).sum(axis=1)                         # (B, K)
            interaction = 0.5 * ((S*S - S2).sum(axis=1))   # (B,)

            # linear scores per class: (B, C)
            lin_scores = np.empty((B, C), np.float32)
            for c in range(C):
                lin_scores[:, c] = linear[c, np.arange(d), xb.T].sum(axis=0)

            scores = lin_scores + interaction[:, None]     # (B, C)
            # softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B                                     # dL/dscores (B, C)

            # gradient for linear: (C, d, 256)
            g_lin = np.zeros_like(linear)
            for c in range(C):
                # accumulate probs[:, c] into g_lin[c, i, val]
                for i in range(d):
                    vals = xb[:, i]
                    np.add.at(g_lin[c, i], vals, probs[:, c])

            # gradient for emb: (d, 256, K)
            g_emb = np.zeros_like(emb)
            # dL/demb[i,val] = (sum_c probs[:,c]) * (S - e_i) for each example
            coeff = probs.sum(axis=1)                      # (B,)
            for i in range(d):
                vals = xb[:, i]                            # (B,)
                e_i = e[:, i, :]                           # (B, K)
                # (S - e_i) * coeff[:, None] -> (B, K)
                grad_i = (S - e_i) * coeff[:, None]
                for k in range(K):
                    np.add.at(g_emb[i, :, k], vals, grad_i[:, k])

            # L2 regularization
            g_lin += L2 * linear
            g_emb += L2 * emb

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            linear -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # write weights
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index (argmax of 2 scores).
 * Weights: magic, d, k, C; float32 linear[C][d][256], emb[d][256][k]. */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K < 1 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    size_t lin_sz = (size_t)C * d * 256;
    size_t emb_sz = (size_t)d * 256 * K;
    size_t expected = 16 + 4*(lin_sz + emb_sz);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float *emb = linear + lin_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *S = malloc(sizeof(float) * K);
    float *S2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float scores[2] = {0.0f, 0.0f};
        for (int j = 0; j < K; j++) { S[j] = 0.0f; S2[j] = 0.0f; }
        for (int f = 0; f < d; f++) {
            unsigned char val = xi[f];
            scores[0] += linear[0*d*256 + f*256 + val];
            scores[1] += linear[1*d*256 + f*256 + val];
            const float *e = emb + ((size_t)f * 256 + val) * K;
            for (int j = 0; j < K; j++) {
                float v = e[j];
                S[j] += v;
                S2[j] += v * v;
            }
        }
        float interaction = 0.0f;
        for (int j = 0; j < K; j++) interaction += S[j]*S[j] - S2[j];
        interaction *= 0.5f;
        scores[0] += interaction;
        scores[1] += interaction;
        unsigned char best = (scores[1] > scores[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```