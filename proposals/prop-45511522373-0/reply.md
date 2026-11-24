```hypothesis
We propose a factorization machine (FM) with embedding size 8 (family "label-fm"). FMs capture pairwise feature interactions effectively for tabular data and are known to perform well on the Adult dataset. The model uses the same input encoding as label-linear (ordinal z-scored, categorical one-hot) but adds a low-rank interaction term. Trained with Adam and binary cross-entropy, it should exceed label-linear's 0.8524 dev accuracy while remaining fast due to small embeddings and precomputed lookup tables. If dev accuracy ≤ 0.8524, the hypothesis fails.
```

```python
"""Factorization Machine for tabular label tasks (family label-fm).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes ordinal as z-scored continuous, categorical as one-hot over training values (255=unseen).
FM: bias + linear weights + 8-dim embeddings for pairwise interactions.
Adam lr=0.01, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights format: magic 0x464D4C31, d, k, C, bias; then per column:
  uint8 is_ordinal; if ordinal: mean, std, lin_w, emb[k];
  else: 256*(lin_w + emb[k]) floats.
"""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_raw = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)  # 0 or 1
    C = card["classes"]
    types = card["input"]["types"]
    d = card["input"]["bytes"]
    n = X_raw.shape[0]
    rng = np.random.default_rng(SEED)

    # Build feature mapping
    col_info = []
    feat_idx = 0
    for j, t in enumerate(types):
        if t == "ordinal":
            vals = X_raw[:, j].astype(np.float64)
            mu, sd = vals.mean(), vals.std()
            if sd == 0:
                sd = 1.0
            col_info.append({"type": "ordinal", "mean": mu, "std": sd, "feat_idx": feat_idx})
            feat_idx += 1
        else:
            uniq = np.unique(X_raw[:, j])
            if 255 not in uniq:
                uniq = np.append(uniq, 255)
            map_arr = np.full(256, -1, dtype=np.int32)
            for idx, v in enumerate(uniq):
                map_arr[v] = feat_idx + idx
            col_info.append({"type": "categorical", "map": map_arr, "num_feat": len(uniq), "feat_idx": feat_idx})
            feat_idx += len(uniq)
    D = feat_idx

    # Build dense feature matrix Z (n x D)
    Z = np.zeros((n, D), dtype=np.float32)
    for j, info in enumerate(col_info):
        if info["type"] == "ordinal":
            idx = info["feat_idx"]
            Z[:, idx] = (X_raw[:, j].astype(np.float32) - info["mean"]) / info["std"]
        else:
            map_arr = info["map"]
            base = info["feat_idx"]
            for i in range(n):
                v = X_raw[i, j]
                f = map_arr[v]
                if f >= 0:
                    Z[i, f] = 1.0

    # Initialize parameters
    bias = np.float32(0.0)
    w = np.zeros(D, dtype=np.float32)
    V = (rng.normal(0, 0.01, size=(D, K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    num_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(num_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]          # (B, D)
            yy = y[idx]         # (B,)
            B = len(yy)

            # Forward
            linear = bias + z @ w                     # (B,)
            S = z @ V                                 # (B, K)
            S_sq = (z ** 2) @ (V ** 2)                # (B, K)
            inter = 0.5 * np.sum(S ** 2 - S_sq, axis=1)
            score = linear + inter
            p = 1.0 / (1.0 + np.exp(-score))
            grad = (p - yy).astype(np.float32)        # (B,)

            # Gradients (mean over batch)
            g_bias = grad.mean()
            g_w = (grad[:, None] * z).mean(axis=0) + L2 * w
            # g_V = mean_b( grad_b * (z_b[:,:,None] * S_b[:,None,:] - z_b[:,:,None]**2 * V[None,:,:]) ) + L2 * V
            g_V = (grad[:, None, None] * (z[:, :, None] * S[:, None, :] - (z ** 2)[:, :, None] * V[None, :, :])).mean(axis=0) + L2 * V

            t += 1
            # Adam updates
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V

            m_bias_hat = m_bias / (1 - beta1 ** t)
            v_bias_hat = v_bias / (1 - beta2 ** t)
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)

            bias -= LR * m_bias_hat / (np.sqrt(v_bias_hat) + eps)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)
            V -= LR * m_V_hat / (np.sqrt(v_V_hat) + eps)

    # Write weights.bin
    with open(sys.argv[4], "wb") as f:
        # Header
        f.write(struct.pack("<IIIIf", 0x464D4C31, d, K, C, bias))
        # Per column
        for info in col_info:
            if info["type"] == "ordinal":
                idx = info["feat_idx"]
                f.write(struct.pack("<B", 1))  # is_ordinal
                f.write(struct.pack("<fff", info["mean"], info["std"], w[idx]))
                f.write(V[idx].astype("<f4").tobytes())
            else:
                f.write(struct.pack("<B", 0))  # categorical
                base = info["feat_idx"]
                num_feat = info["num_feat"]
                map_arr = info["map"]
                # Prepare 256 entries
                lin_table = np.zeros(256, dtype=np.float32)
                emb_table = np.zeros((256, K), dtype=np.float32)
                for v in range(256):
                    f_idx = map_arr[v]
                    if f_idx >= 0:
                        lin_table[v] = w[f_idx]
                        emb_table[v] = V[f_idx]
                f.write(lin_table.astype("<f4").tobytes())
                f.write(emb_table.astype("<f4").tobytes())


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label-fm family.
 * ./infer weights.bin in.rec N
 * Input record: 14 bytes. Output: one byte class (0 or 1).
 * Weights format: see train.py. */
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

typedef struct {
    uint8_t is_ordinal;
    float mean, std, linear_weight;
    float *emb;           // size K
    float *cat_table;     // size 256 * (1 + K) for categorical
} Column;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    unsigned char *ptr = wb;

    // Parse header
    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = *(uint32_t*)ptr; ptr += 4;
    if (magic != 0x464D4C31) { fprintf(stderr, "bad magic\n"); return 3; }
    int32_t d = *(int32_t*)ptr; ptr += 4;
    int32_t K = *(int32_t*)ptr; ptr += 4;
    int32_t C = *(int32_t*)ptr; ptr += 4;
    float bias = *(float*)ptr; ptr += 4;
    if (d != 14 || C != 2 || K <= 0 || K > 64) { fprintf(stderr, "bad header values\n"); return 3; }

    Column cols[14];
    for (int j = 0; j < d; j++) {
        if (ptr + 1 > wb + nw) { fprintf(stderr, "truncated weights (col type)\n"); return 3; }
        cols[j].is_ordinal = *ptr++;
        if (cols[j].is_ordinal) {
            if (ptr + 4*3 > wb + nw) { fprintf(stderr, "truncated weights (ordinal params)\n"); return 3; }
            cols[j].mean = *(float*)ptr; ptr += 4;
            cols[j].std = *(float*)ptr; ptr += 4;
            cols[j].linear_weight = *(float*)ptr; ptr += 4;
            cols[j].emb = (float*)malloc(K * sizeof(float));
            if (ptr + K*4 > wb + nw) { fprintf(stderr, "truncated weights (ordinal emb)\n"); return 3; }
            memcpy(cols[j].emb, ptr, K * 4); ptr += K * 4;
            cols[j].cat_table = NULL;
        } else {
            cols[j].emb = NULL;
            size_t cat_sz = 256 * (1 + K) * 4;
            if (ptr + cat_sz > wb + nw) { fprintf(stderr, "truncated weights (categorical table)\n"); return 3; }
            cols[j].cat_table = (float*)malloc(cat_sz);
            memcpy(cols[j].cat_table, ptr, cat_sz); ptr += cat_sz;
        }
    }
    if (ptr != wb + nw) { fprintf(stderr, "extra bytes in weights file\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = bias;
        float sum_emb[64] = {0};
        float sum_emb_sq[64] = {0};

        for (int j = 0; j < d; j++) {
            unsigned char byte = xi[j];
            if (cols[j].is_ordinal) {
                float x_val = (byte - cols[j].mean) / cols[j].std;
                linear += cols[j].linear_weight * x_val;
                for (int f = 0; f < K; f++) {
                    float v = cols[j].emb[f];
                    sum_emb[f] += x_val * v;
                    sum_emb_sq[f] += x_val * x_val * v * v;
                }
            } else {
                // categorical table: 256 entries, each (1 + K) floats
                const float *entry = cols[j].cat_table + byte * (1 + K);
                linear += entry[0];
                for (int f = 0; f < K; f++) {
                    float v = entry[1 + f];
                    sum_emb[f] += v;
                    sum_emb_sq[f] += v * v;
                }
            }
        }

        float inter = 0.0f;
        for (int f = 0; f < K; f++) {
            inter += sum_emb[f] * sum_emb[f] - sum_emb_sq[f];
        }
        inter *= 0.5f;
        float score = linear + inter;
        out_byte = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }

    // Free allocated memory (not strictly necessary)
    for (int j = 0; j < d; j++) {
        free(cols[j].emb);
        free(cols[j].cat_table);
    }
    free(wb);
    free(x);
    return 0;
}
```