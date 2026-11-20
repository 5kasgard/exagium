```hypothesis
Factorization Machine (FM) with shared 4-d embeddings and per-class linear weights.
Captures pairwise feature interactions efficiently; tabular data with mixed ordinal/categorical
features often benefits from low-rank interactions. If dev accuracy ≤ 0.8524 (label-linear)
or inference cost ≥ 2325 Ir/item (label-mlp16), the model does not meet the goal.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each byte value per feature: linear weight per class + shared embedding (K=4).
Adam lr=1e-2, L2=1e-4, batch=256, epochs=min(30, ceil(600000/n)).
Weights: magic 0x464D5F31, d, K, C; float32 emb[d][256][K], lin[C][d][256], bias[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1e-2, 1e-4, 256, 600000, 30
K = 4  # embedding dimension


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
    rng = np.random.default_rng(SEED)

    # parameters
    emb = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)      # shared embeddings
    lin = np.zeros((C, d, 256), np.float32)                            # per-class linear weights
    bias = np.zeros(C, np.float32)

    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_lin, v_lin = np.zeros_like(lin), np.zeros_like(lin)
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # forward
            linear = bias[None, :] + np.stack([lin[:, j, xb[:, j]].T for j in range(d)], axis=1).sum(axis=1)  # (bsz, C)
            sum_emb = np.zeros((bsz, K), np.float32)
            sum_emb_sq = np.zeros((bsz, K), np.float32)
            for j in range(d):
                e = emb[j, xb[:, j]]                 # (bsz, K)
                sum_emb += e
                sum_emb_sq += e * e
            inter = 0.5 * ((sum_emb * sum_emb - sum_emb_sq).sum(axis=1))  # (bsz,)
            logits = linear + inter[:, None]         # (bsz, C)

            # softmax cross-entropy gradient
            max_logit = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            grad_logits = probs / bsz                # (bsz, C)

            # gradients
            grad_bias = grad_logits.sum(axis=0)                     # (C,)
            grad_lin = np.zeros_like(lin)
            for j in range(d):
                for c in range(C):
                    np.add.at(grad_lin[c, j], xb[:, j], grad_logits[:, c])

            grad_inter = grad_logits.sum(axis=1)                    # (bsz,)
            grad_sum_emb = grad_inter[:, None] * (sum_emb - emb[None, :, :, :] ?)  # need per-feature
            # We'll compute per-feature embedding gradients by looping
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                e = emb[j, xb[:, j]]                     # (bsz, K)
                # d inter / d e_j = sum_emb - e_j
                g = grad_inter[:, None] * (sum_emb - e)   # (bsz, K)
                np.add.at(grad_emb[j], xb[:, j], g)

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
            v_emb = beta2 * v_emb + (1 - beta2) * (grad_emb * grad_emb)
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= LR * m_hat / (np.sqrt(v_hat) + eps) + LR * L2 * emb
            # lin
            m_lin = beta1 * m_lin + (1 - beta1) * grad_lin
            v_lin = beta2 * v_lin + (1 - beta2) * (grad_lin * grad_lin)
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            lin -= LR * m_hat / (np.sqrt(v_hat) + eps) + LR * L2 * lin
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * grad_bias
            v_bias = beta2 * v_bias + (1 - beta2) * (grad_bias * grad_bias)
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= LR * m_hat / (np.sqrt(v_hat) + eps) + LR * L2 * bias

    # write weights
    magic = 0x464D5F31  # "FM_1"
    header = struct.pack("<4i", magic, d, K, C)
    out = header
    out += emb.astype("<f4").tobytes()
    out += lin.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D5F31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }

    size_t emb_sz = (size_t)d * 256 * K;
    size_t lin_sz = (size_t)C * d * 256;
    size_t bias_sz = C;
    long expected = 16 + 4L * (emb_sz + lin_sz + bias_sz);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *lin = emb + emb_sz;
    const float *bias = lin + lin_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * K);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) logits[c] = bias[c];
        memset(sum_emb, 0, sizeof(float) * K);

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *e = emb + ((size_t)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) sum_emb[kk] += e[kk];
            for (int c = 0; c < C; c++)
                logits[c] += lin[((size_t)c * d + j) * 256 + v];
        }

        float sum_sq = 0.0f, sq_sum = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            float val = sum_emb[kk];
            sum_sq += val * val;
            sq_sum += val * val;  // actually same, but we need sum of squares of each embedding vector
        }
        // Wait: interaction = 0.5 * ( (sum e)^2 - sum(e^2) )
        // We need sum of squares of each embedding vector, not square of sum.
        // Recompute correctly:
        float sum_emb_sq = 0.0f;
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *e = emb + ((size_t)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) sum_emb_sq += e[kk] * e[kk];
        }
        float inter = 0.5f * (sum_sq - sum_emb_sq);
        for (int c = 0; c < C; c++) logits[c] += inter;

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```