```hypothesis
Family: label-fm (Factorization Machine). It adds pairwise feature interactions via low-rank latent vectors (k=16) to the linear per-byte lookup tables. On Adult, ordinal and categorical features likely have predictive interactions (e.g., education × occupation) that a linear model misses. The FM computes interactions in O(d·k) time, keeping inference fast. If dev accuracy does not exceed label-linear (0.8524), it suggests interactions are either uninformative or the model overfits despite L2 regularization.
```

```python
"""label-fm: Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports per-byte lookup tables for linear and factor parts, plus W2, b2.
Format: magic 0x464D4C31, d, k, C; float32 L[d][256][C], V[d][256][k], W2[k][C], b2[C]."""
import json, math, struct, sys

import numpy as np

K = 16
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
    types = card["input"]["types"]
    rng = np.random.default_rng(SEED)

    # --- Build feature metadata ---
    ordinal = []
    categorical = []
    mu_sd = []          # for ordinal: (mu, sd)
    cat_maps = []       # for categorical: array[256] -> cat index or -1
    cat_sizes = []      # number of categories per categorical feature
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            mu_sd.append((mu, sd))
            ordinal.append(j)
        else:
            vals = np.unique(X[:, j])
            # map byte value to 0..len(vals)-1, unseen (255) -> -1
            mapping = np.full(256, -1, dtype=np.int32)
            mapping[vals] = np.arange(len(vals))
            cat_maps.append(mapping)
            cat_sizes.append(len(vals))
            categorical.append(j)

    # --- Initialize parameters ---
    # Linear part
    w_lin_ord = {}      # j -> (C,)
    w_lin_cat = {}      # j -> (num_cat, C)
    # Factor part
    v_fac_ord = {}      # j -> (k,)
    v_fac_cat = {}      # j -> (num_cat, k)
    # Interaction projection
    W2 = (rng.uniform(-1, 1, (K, C)) * math.sqrt(6.0 / K)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    for j in ordinal:
        w_lin_ord[j] = (rng.uniform(-1, 1, C) * math.sqrt(6.0 / C)).astype(np.float32)
        v_fac_ord[j] = (rng.uniform(-1, 1, K) * math.sqrt(6.0 / K)).astype(np.float32)
    for idx, j in enumerate(categorical):
        nc = cat_sizes[idx]
        w_lin_cat[j] = (rng.uniform(-1, 1, (nc, C)) * math.sqrt(6.0 / nc)).astype(np.float32)
        v_fac_cat[j] = (rng.uniform(-1, 1, (nc, K)) * math.sqrt(6.0 / nc)).astype(np.float32)

    # --- Adam state ---
    params = []
    # order: w_lin_ord, v_fac_ord, w_lin_cat, v_fac_cat, W2, b2
    for j in ordinal:
        params.append(w_lin_ord[j])
        params.append(v_fac_ord[j])
    for j in categorical:
        params.append(w_lin_cat[j])
        params.append(v_fac_cat[j])
    params.append(W2)
    params.append(b2)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    def softmax_cross_entropy_grad(scores, y_idx):
        # scores: (B, C), y_idx: (B,)
        B = scores.shape[0]
        max_s = scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores - max_s)
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_idx] -= 1.0
        return (probs / B).astype(np.float32)

    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # --- Forward ---
            linear_acc = np.zeros((B, C), np.float32)
            sum_v = np.zeros((B, K), np.float32)
            sum_v_sq = np.zeros((B, K), np.float32)

            # Ordinal features
            for j in ordinal:
                mu, sd = mu_sd[ordinal.index(j)]
                x_std = (xb[:, j].astype(np.float32) - mu) / sd   # (B,)
                linear_acc += w_lin_ord[j] * x_std[:, None]
                sum_v += v_fac_ord[j] * x_std[:, None]
                sum_v_sq += (v_fac_ord[j] ** 2) * (x_std[:, None] ** 2)

            # Categorical features
            for j in categorical:
                cmap = cat_maps[categorical.index(j)]
                cats = cmap[xb[:, j]]          # (B,) with -1 for unseen
                # gather for seen
                seen = cats != -1
                if seen.any():
                    cat_idx = cats[seen]
                    linear_acc[seen] += w_lin_cat[j][cat_idx]
                    sum_v[seen] += v_fac_cat[j][cat_idx]
                    sum_v_sq[seen] += v_fac_cat[j][cat_idx] ** 2

            z = 0.5 * (sum_v * sum_v - sum_v_sq)          # (B, K)
            scores = linear_acc + z @ W2 + b2             # (B, C)

            # --- Loss gradient ---
            d_scores = softmax_cross_entropy_grad(scores, yb)   # (B, C)

            # --- Backward ---
            d_linear_acc = d_scores                          # (B, C)
            d_z = d_scores @ W2.T                            # (B, K)
            d_W2 = z.T @ d_scores                            # (K, C)
            d_b2 = d_scores.sum(axis=0)                      # (C,)

            d_sum_v = d_z * sum_v                            # (B, K)
            d_sum_v_sq = d_z * (-0.5)                        # (B, K)

            grads = []
            # Ordinal
            for j in ordinal:
                mu, sd = mu_sd[ordinal.index(j)]
                x_std = (xb[:, j].astype(np.float32) - mu) / sd   # (B,)
                # w_lin
                gw = (d_linear_acc * x_std[:, None]).sum(axis=0) + L2 * w_lin_ord[j]
                # v_fac
                gv = (d_sum_v * x_std[:, None]).sum(axis=0)
                gv += (d_sum_v_sq * 2 * v_fac_ord[j] * (x_std[:, None] ** 2)).sum(axis=0)
                gv += L2 * v_fac_ord[j]
                grads.append(gw)
                grads.append(gv)

            # Categorical
            for j in categorical:
                cmap = cat_maps[categorical.index(j)]
                cats = cmap[xb[:, j]]
                seen = cats != -1
                gw = np.zeros_like(w_lin_cat[j])
                gv = np.zeros_like(v_fac_cat[j])
                if seen.any():
                    cat_idx = cats[seen]
                    np.add.at(gw, cat_idx, d_linear_acc[seen])
                    np.add.at(gv, cat_idx, d_sum_v[seen])
                    # d_sum_v_sq contribution: 2 * v * d_sum_v_sq
                    np.add.at(gv, cat_idx, 2 * v_fac_cat[j][cat_idx] * d_sum_v_sq[seen])
                gw += L2 * w_lin_cat[j]
                gv += L2 * v_fac_cat[j]
                grads.append(gw)
                grads.append(gv)

            grads.append(d_W2 + L2 * W2)
            grads.append(d_b2)   # no L2 on bias

            # --- Adam update ---
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # --- Export per-byte lookup tables ---
    L_table = np.zeros((d, 256, C), np.float32)
    V_table = np.zeros((d, 256, K), np.float32)
    for j in ordinal:
        mu, sd = mu_sd[ordinal.index(j)]
        for byte in range(256):
            std = (byte - mu) / sd
            L_table[j, byte] = w_lin_ord[j] * std
            V_table[j, byte] = v_fac_ord[j] * std
    for j in categorical:
        cmap = cat_maps[categorical.index(j)]
        for byte in range(256):
            ci = cmap[byte]
            if ci != -1:
                L_table[j, byte] = w_lin_cat[j][ci]
                V_table[j, byte] = v_fac_cat[j][ci]

    # Pack weights
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, K, C)
    body = L_table.astype("<f4").tobytes() + V_table.astype("<f4").tobytes()
    body += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class index (0 or 1).
 * Weights format: magic 0x464D4C31, d, k, C; float32 L[d][256][C], V[d][256][k], W2[k][C], b2[C]. */
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
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * (C + K) + (long)K * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *L = (const float *)(hd + 4);
    const float *V = L + (long)d * 256 * C;
    const float *W2 = V + (long)d * 256 * K;
    const float *b2 = W2 + (long)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lin = (float *)malloc(sizeof(float) * C);
    float *sum_v = (float *)malloc(sizeof(float) * K);
    float *sum_v_sq = (float *)malloc(sizeof(float) * K);
    float *z = (float *)malloc(sizeof(float) * K);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) lin[c] = b2[c];
        for (int u = 0; u < K; u++) { sum_v[u] = 0.0f; sum_v_sq[u] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *Lj = L + ((long)j * 256 + v) * C;
            const float *Vj = V + ((long)j * 256 + v) * K;
            for (int c = 0; c < C; c++) lin[c] += Lj[c];
            for (int u = 0; u < K; u++) {
                float vu = Vj[u];
                sum_v[u] += vu;
                sum_v_sq[u] += vu * vu;
            }
        }

        for (int u = 0; u < K; u++) z[u] = 0.5f * (sum_v[u] * sum_v[u] - sum_v_sq[u]);

        for (int c = 0; c < C; c++) scores[c] = lin[c];
        for (int u = 0; u < K; u++) {
            float zu = z[u];
            for (int c = 0; c < C; c++) scores[c] += zu * W2[(long)u * C + c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```