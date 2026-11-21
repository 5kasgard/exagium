```hypothesis
Family: label-fm (Factorization Machine).
Why: Adult has strong pairwise feature interactions (e.g., education×occupation, marital-status×sex) that linear models miss.
FM captures all pairwise interactions with O(k·d) parameters (k=8 latent dims, d=14 features) and O(k·d) inference cost,
adding minimal overhead vs label-linear (719 Ir/item).
Failure: If dev accuracy ≤ 0.8524 (label-linear) or inference cost > 2000 Ir/item.
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Learns per-byte-value linear weights and k-dim embeddings.
Exports: int32 magic 0x464D0001, d, k, C; float32 bias, w[d*256], V[d*256*k]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split conceptually)
K = 8
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
TOTAL_SAMPLES = 600000
MAX_EPOCHS = 50


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
    d = card["input"]["bytes"]  # 14
    C = card["classes"]  # 2
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    n_features = d * 256
    bias = np.float32(0.0)
    w = np.zeros(n_features, dtype=np.float32)
    V = rng.normal(0, 0.01, (n_features, K)).astype(np.float32)

    # Adam state
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    m_b, v_b = np.float32(0.0), np.float32(0.0)

    # Precompute feature indices for each sample: (n, d)
    feat_idx = X.astype(np.int32) + np.arange(d, dtype=np.int32) * 256

    steps_per_epoch = (n + BATCH - 1) // BATCH
    max_steps = min(MAX_EPOCHS * steps_per_epoch, TOTAL_SAMPLES // BATCH)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(max_steps):
        batch_idx = rng.integers(0, n, size=BATCH)
        idx_batch = feat_idx[batch_idx]  # (B, d)
        y_batch = y[batch_idx].astype(np.float32)

        # Forward
        w_batch = w[idx_batch]  # (B, d)
        linear = bias + w_batch.sum(axis=1)
        V_batch = V[idx_batch]  # (B, d, K)
        embed_sum = V_batch.sum(axis=1)  # (B, K)
        embed_sq_sum = (V_batch * V_batch).sum(axis=1)
        interaction = 0.5 * ((embed_sum * embed_sum - embed_sq_sum).sum(axis=1))
        logit = linear + interaction

        # Logistic loss gradient
        p = 1.0 / (1.0 + np.exp(-logit))
        grad = p - y_batch  # (B,)

        # Gradients
        dbias = grad.mean()

        grad_w = np.zeros_like(w)
        np.add.at(grad_w, idx_batch.ravel(), np.repeat(grad, d))
        grad_w /= BATCH

        grad_V_batch = grad[:, None, None] * (embed_sum[:, None, :] - V_batch)  # (B, d, K)
        grad_V = np.zeros_like(V)
        flat_idx = idx_batch.ravel()
        for f in range(K):
            np.add.at(grad_V[:, f], flat_idx, grad_V_batch[:, :, f].ravel())
        grad_V /= BATCH

        # L2 regularization
        grad_w += L2 * w
        grad_V += L2 * V

        # Adam update
        t += 1
        m_b = beta1 * m_b + (1 - beta1) * dbias
        v_b = beta2 * v_b + (1 - beta2) * (dbias * dbias)
        m_b_hat = m_b / (1 - beta1 ** t)
        v_b_hat = v_b / (1 - beta2 ** t)
        bias -= LR * m_b_hat / (np.sqrt(v_b_hat) + eps)

        m_w = beta1 * m_w + (1 - beta1) * grad_w
        v_w = beta2 * v_w + (1 - beta2) * (grad_w * grad_w)
        m_w_hat = m_w / (1 - beta1 ** t)
        v_w_hat = v_w / (1 - beta2 ** t)
        w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)

        m_V = beta1 * m_V + (1 - beta1) * grad_V
        v_V = beta2 * v_V + (1 - beta2) * (grad_V * grad_V)
        m_V_hat = m_V / (1 - beta1 ** t)
        v_V_hat = v_V / (1 - beta2 ** t)
        V -= LR * m_V_hat / (np.sqrt(v_V_hat) + eps)

    # Write weights
    magic = 0x464D0001
    out = struct.pack("<4i", magic, d, K, C)
    out += np.array([bias], dtype=np.float32).tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class (0 or 1).
 * Weights format: int32 magic(0x464D0001), d, k, C; float32 bias, w[d*256], V[d*256*k]. */
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
    if (C != 2 || d != 14 || K <= 0 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 * (1 + K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *w = bias + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *embed_sum = malloc(sizeof(float) * K);
    float *embed_sq_sum = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = bias[0];
        for (int f = 0; f < K; f++) { embed_sum[f] = 0.0f; embed_sq_sum[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            int v = xi[j];
            long idx = (long)j * 256 + v;
            linear += w[idx];
            const float *Vj = V + (long)idx * K;
            for (int f = 0; f < K; f++) {
                float val = Vj[f];
                embed_sum[f] += val;
                embed_sq_sum[f] += val * val;
            }
        }

        float interaction = 0.0f;
        for (int f = 0; f < K; f++) {
            interaction += embed_sum[f] * embed_sum[f] - embed_sq_sum[f];
        }
        interaction *= 0.5f;
        float logit = linear + interaction;

        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    free(embed_sum);
    free(embed_sq_sum);
    return 0;
}
```