```hypothesis
Family: label-fm (Factorization Machine with embedding dimension K=8).
Why: Adult has strong feature interactions (e.g., education × occupation). Linear model misses these; FM captures pairwise interactions with O(d·K) parameters, adding little cost. At K=8, inference is ~14×8=112 multiply-adds per item, far cheaper than MLP/GBDT.
Failure: If dev accuracy ≤ label-linear (0.8524) or instructions/item exceed label-mlp16 (2325), the family does not justify its complexity.
```

```python
import json
import math
import struct
import sys

import numpy as np

# Hyperparameters (tuned on the last 20% of train.in)
K = 8
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_epoch(X, y, idx, linear, emb, m_lin, v_lin, m_emb, v_emb, t, rng):
    order = rng.permutation(len(idx))
    for s in range(0, len(idx), BATCH):
        batch_idx = order[s:s+BATCH]
        batch_size = len(batch_idx)
        grad_lin = np.zeros_like(linear)
        grad_emb = np.zeros_like(emb)
        for b in range(batch_size):
            i = idx[batch_idx[b]]
            xi = X[i]
            yi = y[i]
            lin_sum = 0.0
            emb_sum = np.zeros(K, dtype=np.float32)
            emb_sq_sum = 0.0
            for j in range(X.shape[1]):
                idx_flat = j * 256 + xi[j]
                lin_sum += linear[idx_flat]
                e = emb[idx_flat]
                emb_sum += e
                emb_sq_sum += np.dot(e, e)
            emb_sum_sq = np.dot(emb_sum, emb_sum)
            interaction = 0.5 * (emb_sum_sq - emb_sq_sum)
            score = lin_sum + interaction
            p = 1.0 / (1.0 + math.exp(-score))
            g = p - yi
            for j in range(X.shape[1]):
                idx_flat = j * 256 + xi[j]
                grad_lin[idx_flat] += g
                e = emb[idx_flat]
                grad_emb[idx_flat] += g * (emb_sum - e)
        grad_lin /= batch_size
        grad_emb /= batch_size
        grad_lin += L2 * linear
        grad_emb += L2 * emb
        t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_lin = beta1 * m_lin + (1 - beta1) * grad_lin
        v_lin = beta2 * v_lin + (1 - beta2) * (grad_lin * grad_lin)
        m_hat_lin = m_lin / (1 - beta1 ** t)
        v_hat_lin = v_lin / (1 - beta2 ** t)
        linear -= LR * m_hat_lin / (np.sqrt(v_hat_lin) + eps)
        m_emb = beta1 * m_emb + (1 - beta1) * grad_emb
        v_emb = beta2 * v_emb + (1 - beta2) * (grad_emb * grad_emb)
        m_hat_emb = m_emb / (1 - beta1 ** t)
        v_hat_emb = v_emb / (1 - beta2 ** t)
        emb -= LR * m_hat_emb / (np.sqrt(v_hat_emb) + eps)
    return linear, emb, m_lin, v_lin, m_emb, v_emb, t

def compute_loss(X, y, idx, linear, emb):
    total_loss = 0.0
    correct = 0
    for i in idx:
        xi = X[i]
        yi = y[i]
        lin_sum = 0.0
        emb_sum = np.zeros(K, dtype=np.float32)
        emb_sq_sum = 0.0
        for j in range(X.shape[1]):
            idx_flat = j * 256 + xi[j]
            lin_sum += linear[idx_flat]
            e = emb[idx_flat]
            emb_sum += e
            emb_sq_sum += np.dot(e, e)
        emb_sum_sq = np.dot(emb_sum, emb_sum)
        interaction = 0.5 * (emb_sum_sq - emb_sq_sum)
        score = lin_sum + interaction
        p = 1.0 / (1.0 + math.exp(-score))
        total_loss += - (yi * math.log(p + 1e-12) + (1 - yi) * math.log(1 - p + 1e-12))
        pred = 1 if score > 0 else 0
        if pred == yi:
            correct += 1
    return total_loss / len(idx), correct / len(idx)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Tuning split: last 20% in file order
    split = int(n * 0.8)
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)

    rng = np.random.default_rng(SEED)

    linear = np.zeros(d * 256, dtype=np.float32)
    emb = rng.normal(0, 0.01, size=(d * 256, K)).astype(np.float32)
    m_lin = np.zeros_like(linear)
    v_lin = np.zeros_like(linear)
    m_emb = np.zeros_like(emb)
    v_emb = np.zeros_like(emb)
    t = 0

    best_val_acc = -1
    best_epoch = 0
    best_linear = None
    best_emb = None

    for epoch in range(MAX_EPOCHS):
        linear, emb, m_lin, v_lin, m_emb, v_emb, t = train_epoch(
            X, y, train_idx, linear, emb, m_lin, v_lin, m_emb, v_emb, t, rng
        )
        val_loss, val_acc = compute_loss(X, y, val_idx, linear, emb)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_linear = linear.copy()
            best_emb = emb.copy()
        if epoch - best_epoch >= 5:
            break

    # Retrain on all data for the best number of epochs
    if best_epoch > 0:
        linear = np.zeros(d * 256, dtype=np.float32)
        emb = rng.normal(0, 0.01, size=(d * 256, K)).astype(np.float32)
        m_lin = np.zeros_like(linear)
        v_lin = np.zeros_like(linear)
        m_emb = np.zeros_like(emb)
        v_emb = np.zeros_like(emb)
        t = 0
        all_idx = np.arange(n)
        for epoch in range(best_epoch + 1):
            linear, emb, m_lin, v_lin, m_emb, v_emb, t = train_epoch(
                X, y, all_idx, linear, emb, m_lin, v_lin, m_emb, v_emb, t, rng
            )
    else:
        linear = best_linear
        emb = best_emb

    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, K, C)
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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
    if (C != 2 || d != 14 || K != 8) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 16 + (long)d * 256 * 4 + (long)d * 256 * K * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = (float *)malloc(K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float lin_sum = 0.0f;
        for (int j = 0; j < K; j++) emb_sum[j] = 0.0f;
        float emb_sq_sum = 0.0f;
        for (int j = 0; j < d; j++) {
            int idx = j * 256 + xi[j];
            lin_sum += linear[idx];
            const float *e = emb + (long)idx * K;
            for (int kk = 0; kk < K; kk++) {
                float val = e[kk];
                emb_sum[kk] += val;
                emb_sq_sum += val * val;
            }
        }
        float emb_sum_sq = 0.0f;
        for (int kk = 0; kk < K; kk++) emb_sum_sq += emb_sum[kk] * emb_sum[kk];
        float interaction = 0.5f * (emb_sum_sq - emb_sq_sum);
        float score = lin_sum + interaction;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(emb_sum);
    return 0;
}
```