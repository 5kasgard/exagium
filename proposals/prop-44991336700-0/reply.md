```hypothesis
Family: label-fm (Factorization Machine, degree-2). Adult income has strong pairwise interactions (education×occupation, marital-status×relationship, capital-gain×hours-per-week) that linear models miss. FM captures all pairwise interactions via low-rank latent vectors (k=4) shared across classes, adding only ~21K parameters over linear. Deterministic Adam training on the full 26k rows should beat label-linear's 0.8524 dev accuracy; if dev accuracy ≤ 0.853 the hypothesis fails.
```

```python
"""Factorization Machine (degree-2) for tabular label tasks with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy. Exports shared embeddings + per-class linear weights.
Format: magic 0x464D0001, d, k, C; float32 emb[d][256][k], lin[d][256][C], bias[C]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-5, 256, 600000, 100


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
    # embeddings: (d, 256, K), linear: (d, 256, C), bias: (C,)
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)
    lin = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    bias = np.zeros(C, np.float32)

    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_lin, v_lin = np.zeros_like(lin), np.zeros_like(lin)
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # forward: linear part
            lin_sum = np.zeros((bsz, C), np.float32)
            for j in range(d):
                lin_sum += lin[j, xb[:, j]]  # (bsz, C)

            # forward: FM interaction part (shared across classes)
            # sum_v = Σ_j emb[j, x_j]  -> (bsz, K)
            # sum_v2 = Σ_j emb[j, x_j]^2 -> (bsz, K)
            # interaction = 0.5 * Σ_k (sum_v[:,k]^2 - sum_v2[:,k]) -> (bsz,)
            sum_v = np.zeros((bsz, K), np.float32)
            sum_v2 = np.zeros((bsz, K), np.float32)
            for j in range(d):
                e = emb[j, xb[:, j]]  # (bsz, K)
                sum_v += e
                sum_v2 += e * e
            inter = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)  # (bsz,)

            # logits: bias + linear + interaction (broadcast to C)
            logits = bias + lin_sum + inter[:, None]  # (bsz, C)

            # softmax + cross-entropy gradient
            max_logits = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            grad_logits = probs / bsz  # (bsz, C)

            # gradients
            # bias
            g_bias = grad_logits.sum(axis=0) + L2 * bias
            # linear
            g_lin = np.zeros_like(lin)
            for j in range(d):
                # grad_logits.T (C, bsz) @ one-hot? accumulate per value
                # For each sample, add grad_logits[i] to g_lin[j, xb[i,j]]
                np.add.at(g_lin[j], xb[:, j], grad_logits)
            g_lin += L2 * lin
            # embeddings
            g_emb = np.zeros_like(emb)
            # d(inter)/d(emb[j, v, k]) = sum_v[i,k] - emb[j, v, k] for samples with x_j=v
            # So g_emb[j, v, k] = Σ_{i: xb[i,j]=v} grad_logits[i].sum() * (sum_v[i,k] - emb[j, v, k])
            # Vectorized per feature:
            for j in range(d):
                vals = xb[:, j]  # (bsz,)
                # sum_v_per_sample = sum_v  (bsz, K)
                # emb_val = emb[j, vals]  (bsz, K)
                # coeff = grad_logits.sum(axis=1)  (bsz,)  -- wait, interaction is shared, so gradient wrt inter is sum_c grad_logits[i,c]
                coeff = grad_logits.sum(axis=1)  # (bsz,)
                # g_emb[j, vals, k] += coeff[i] * (sum_v[i,k] - emb[j, vals[i], k])
                diff = sum_v - emb[j, vals]  # (bsz, K)
                weighted_diff = diff * coeff[:, None]  # (bsz, K)
                np.add.at(g_emb[j], vals, weighted_diff)
            g_emb += L2 * emb

            # Adam update
            t += 1
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            lin -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # write weights
    # magic, d, k, C
    out = struct.pack("<4i", 0x464D0001, d, K, C)
    out += emb.astype("<f4").tobytes()
    out += lin.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine (degree-2) inference for label tasks.
 * ./infer weights.bin in.rec N
 * Weights: magic 0x464D0001, d, k, C; float32 emb[d][256][k], lin[d][256][C], bias[C].
 * Input record: d bytes. Output: 1 byte class index (argmax). */
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
    if (d < 1 || d > 256 || K < 1 || K > 64 || C < 1 || C > 256) {
        fprintf(stderr, "bad weights dims\n"); return 3;
    }
    long expected = 16L + 4L * ((long)d * 256 * K + (long)d * 256 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *lin = emb + (long)d * 256 * K;
    const float *bias = lin + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v2 = malloc(sizeof(float) * K);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // linear part: bias + Σ_j lin[j, xi[j]]
        for (int c = 0; c < C; c++) logits[c] = bias[c];
        for (int j = 0; j < d; j++) {
            const float *lj = lin + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) logits[c] += lj[c];
        }

        // FM interaction: 0.5 * Σ_k ( (Σ_j emb[j,xi[j],k])^2 - Σ_j emb[j,xi[j],k]^2 )
        memset(sum_v, 0, sizeof(float) * K);
        memset(sum_v2, 0, sizeof(float) * K);
        for (int j = 0; j < d; j++) {
            const float *ej = emb + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) {
                float v = ej[kk];
                sum_v[kk] += v;
                sum_v2[kk] += v * v;
            }
        }
        float inter = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            float sv = sum_v[kk];
            inter += 0.5f * (sv * sv - sum_v2[kk]);
        }
        for (int c = 0; c < C; c++) logits[c] += inter;

        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```