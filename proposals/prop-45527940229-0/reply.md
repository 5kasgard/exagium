```hypothesis
Family: label-fm (factorization machine with linear term and pairwise interactions via embeddings).
Why: Adult has mixed ordinal/categorical features; FMs capture second-order interactions efficiently with O(d*K) parameters and inference cost, often outperforming linear models on tabular data. With K=8 embeddings, the model adds minimal overhead vs label-linear (719 Ir/item) while modeling feature crosses.
Failure: If dev accuracy ≤ 0.8524 (label-linear) or instructions/item ≥ 2000 (exceeding label-mlp16 cost), the model does not meet the goal.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes ordinal as normalized scalars, categorical as one-hot (with unseen).
Learns linear weights w_i and embeddings V_i (dim K) per feature.
Exports per-byte lookup tables: int32 magic 0x464D3130, d, K, C; float32 T[d][256][1+K], bias."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100


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
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Build feature mapping: for each column, assign feature indices and compute values
    feat_idx = np.zeros((n, d), dtype=np.int32)
    feat_val = np.zeros((n, d), dtype=np.float32)
    col_feat_start = []
    col_feat_count = []
    k = 0
    for j, t in enumerate(card["input"]["types"]):
        col_feat_start.append(k)
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            norm = (np.arange(256, dtype=np.float64) - mu) / sd
            feat_idx[:, j] = k
            feat_val[:, j] = norm[X[:, j]].astype(np.float32)
            k += 1
            col_feat_count.append(1)
        else:
            vals = np.unique(X[:, j])
            val_to_idx = {val: idx for idx, val in enumerate(vals)}
            unseen = len(vals)
            cnt = len(vals) + 1
            for i in range(n):
                v = X[i, j]
                feat_idx[i, j] = k + (val_to_idx[v] if v in val_to_idx else unseen)
                feat_val[i, j] = 1.0
            k += cnt
            col_feat_count.append(cnt)

    # Initialize parameters
    w = rng.normal(0, 0.01, k).astype(np.float32)
    V = rng.normal(0, 0.01, (k, K)).astype(np.float32)
    bias = np.float32(0.0)

    # Adam state
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    m_b, v_b = 0.0, 0.0
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            batch_idx = feat_idx[idx]          # (B, d)
            batch_val = feat_val[idx]          # (B, d)
            batch_y = y[idx]                   # (B,)

            # Linear term: sum_j w[feat] * val
            lin = np.sum(w[batch_idx] * batch_val, axis=1)
            # Embedding sum and squared sum
            V_sel = V[batch_idx]               # (B, d, K)
            V_scaled = V_sel * batch_val[:, :, None]  # (B, d, K)
            emb_sum = np.sum(V_scaled, axis=1)        # (B, K)
            emb_sq_sum = np.sum(V_scaled ** 2, axis=1) # (B, K)
            inter = 0.5 * (np.sum(emb_sum ** 2, axis=1) - np.sum(emb_sq_sum, axis=1))
            logit = bias + lin + inter

            # Logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-logit))
            grad = p - batch_y                 # (B,)

            # Gradients
            grad_lin = grad
            grad_inter = grad
            # w gradient
            gw = np.zeros_like(w)
            np.add.at(gw, batch_idx.ravel(), (grad_lin[:, None] * batch_val.ravel()).astype(np.float32))
            gw += L2 * w
            # V gradient
            gV = np.zeros_like(V)
            # d(inter)/dV = (emb_sum - V_scaled) * val  (per feature)
            # For each example, for each feature: (emb_sum[b] - V_scaled[b,j]) * val[b,j]
            # We can compute by looping over d (d=14 small) for clarity and speed
            for j in range(d):
                idx_j = batch_idx[:, j]
                val_j = batch_val[:, j]
                V_scaled_j = V_scaled[:, j]
                emb_sum_j = emb_sum - V_scaled_j  # (B, K)
                factor = (grad_inter[:, None] * emb_sum_j) * val_j[:, None]  # (B, K)
                np.add.at(gV, idx_j, factor)
            gV += L2 * V
            # bias gradient
            gb = np.sum(grad)

            t += 1
            # Adam update
            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * (gw * gw)
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)

            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * (gV * gV)
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            V -= LR * m_V_hat / (np.sqrt(v_V_hat) + eps)

            m_b = beta1 * m_b + (1 - beta1) * gb
            v_b = beta2 * v_b + (1 - beta2) * (gb * gb)
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)
            bias -= LR * m_b_hat / (math.sqrt(v_b_hat) + eps)

    # Build lookup tables T[d][256][1+K]
    T = np.zeros((d, 256, 1 + K), dtype=np.float32)
    for j in range(d):
        start = col_feat_start[j]
        cnt = col_feat_count[j]
        if card["input"]["types"][j] == "ordinal":
            v = np.arange(256, dtype=np.float64)
            mu, sd = X[:, j].astype(np.float64).mean(), X[:, j].astype(np.float64).std() or 1.0
            norm = (v - mu) / sd
            fid = start
            T[j, :, 0] = (w[fid] * norm).astype(np.float32)
            T[j, :, 1:] = (V[fid] * norm[:, None]).astype(np.float32)
        else:
            vals = np.unique(X[:, j])
            val_to_idx = {val: idx for idx, val in enumerate(vals)}
            unseen = len(vals)
            for byte_val in range(256):
                if byte_val in val_to_idx:
                    fid = start + val_to_idx[byte_val]
                else:
                    fid = start + unseen
                T[j, byte_val, 0] = w[fid]
                T[j, byte_val, 1:] = V[fid]

    # Write weights
    magic = 0x464D3130  # "FM10"
    C = card["classes"]
    header = struct.pack("<4i", magic, d, K, C)
    body = T.astype("<f4").tobytes() + np.array([bias], dtype="<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class index (0/1). */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D3130) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K <= 0 || d <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16 + 4L * d * 256 * (1 + K) + 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(wb + 16);
    const float bias = *(const float *)(wb + 16 + 4L * d * 256 * (1 + K));

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float linear_sum = 0.0f;
        float emb_sum[128];  // K <= 128 reasonable
        float emb_sq_sum[128];
        for (int u = 0; u < K; u++) { emb_sum[u] = 0.0f; emb_sq_sum[u] = 0.0f; }
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * (1 + K);
            linear_sum += t[0];
            for (int u = 0; u < K; u++) {
                float v = t[1 + u];
                emb_sum[u] += v;
                emb_sq_sum[u] += v * v;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) inter += emb_sum[u] * emb_sum[u] - emb_sq_sum[u];
        inter *= 0.5f;
        float score = bias + linear_sum + inter;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```