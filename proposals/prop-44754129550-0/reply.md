```hypothesis
Family: label-fm (Factorization Machine). Adult benefits from pairwise feature interactions (e.g., education × occupation, marital-status × relationship) which linear models miss. A rank-8 FM adds only ~130 KB weights and O(d·K) inference (d=14, K=8), keeping CPU cost near label-linear while capturing interactions. If dev accuracy ≤ 0.8524 (label-linear) or instructions/item > 2× label-linear, the hypothesis fails.
```
```python
import json, math, struct, sys
import numpy as np

K = 8
SEED, LR, L2, BATCH, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 50

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
    n, d = X.shape
    C = card["classes"]
    assert C == 2
    rng = np.random.default_rng(SEED)

    encoders = []
    Z_cols = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j]
        if t == "ordinal":
            mu = v.mean()
            sd = v.std() or 1.0
            zvals = (np.arange(256, dtype=np.float64) - mu) / sd
            zcol = zvals[v].astype(np.float32).reshape(-1, 1)
            Z_cols.append(zcol)
            encoders.append(("ordinal", mu, sd, zvals.astype(np.float32)))
        else:
            vals = np.unique(v)
            n_vals = len(vals)
            val_to_idx = {val: i for i, val in enumerate(vals)}
            zcol = np.zeros((n, n_vals), dtype=np.float32)
            for i, val in enumerate(v):
                if val in val_to_idx:
                    zcol[i, val_to_idx[val]] = 1.0
            Z_cols.append(zcol)
            encoders.append(("categorical", vals, val_to_idx, n_vals))

    Z = np.concatenate(Z_cols, axis=1)
    n_features = Z.shape[1]

    w = rng.normal(0, 0.01, n_features).astype(np.float32)
    V = rng.normal(0, 0.01, (n_features, K)).astype(np.float32)
    bias = np.float32(0.0)

    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    m_b, v_b = 0.0, 0.0
    t = 0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            z = Z[idx]
            yy = y[idx]
            linear = z @ w + bias
            zV = z @ V
            z_sq = z * z
            V_sq = V * V
            zV_sq = z_sq @ V_sq
            interaction = 0.5 * np.sum(zV * zV - zV_sq, axis=1)
            score = linear + interaction
            p = 1.0 / (1.0 + np.exp(-score))
            grad_score = p - yy
            grad_w = z.T @ grad_score + L2 * w
            grad_bias = np.sum(grad_score)
            grad_V = z.T @ (grad_score[:, None] * zV) - (grad_score[:, None] * z_sq).T @ V
            grad_V += L2 * V

            t += 1
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * grad_w * grad_w
            m_V = beta1 * m_V + (1 - beta1) * grad_V
            v_V = beta2 * v_V + (1 - beta2) * grad_V * grad_V
            m_b = beta1 * m_b + (1 - beta1) * grad_bias
            v_b = beta2 * v_b + (1 - beta2) * grad_bias * grad_bias

            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)

            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)
            V -= LR * m_V_hat / (np.sqrt(v_V_hat) + eps)
            bias -= LR * m_b_hat / (np.sqrt(v_b_hat) + eps)

    feat_starts = [0]
    for zcol in Z_cols:
        feat_starts.append(feat_starts[-1] + zcol.shape[1])

    linear_table = np.zeros((d, 256), dtype=np.float32)
    latent_table = np.zeros((d, 256, K), dtype=np.float32)

    for j in range(d):
        enc = encoders[j]
        start, end = feat_starts[j], feat_starts[j+1]
        w_col = w[start:end]
        V_col = V[start:end, :]
        if enc[0] == "ordinal":
            _, mu, sd, zvals = enc
            for b in range(256):
                if b < 255:
                    z = zvals[b]
                    linear_table[j, b] = w_col[0] * z
                    latent_table[j, b, :] = V_col[0, :] * z
                else:
                    linear_table[j, b] = 0.0
                    latent_table[j, b, :] = 0.0
        else:
            _, vals, val_to_idx, n_vals = enc
            for b in range(256):
                if b in val_to_idx:
                    idx = val_to_idx[b]
                    linear_table[j, b] = w_col[idx]
                    latent_table[j, b, :] = V_col[idx, :]
                else:
                    linear_table[j, b] = 0.0
                    latent_table[j, b, :] = 0.0

    magic = 0x464D4C31
    out = struct.pack("<4i f", magic, d, K, C, bias)
    out += linear_table.astype("<f4").tobytes()
    out += latent_table.astype("<f4").tobytes()
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || C != 2 || K < 1 || K > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    float *fptr = (float *)(hd + 4);
    float bias = fptr[0];
    float *linear_table = fptr + 1;
    float *latent_table = linear_table + (long)d * 256;
    long expected = 20 + (long)d * 256 * 4 + (long)d * 256 * K * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *latent_sum = calloc(K, sizeof(float));
    float *latent_sq_sum = calloc(K, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float linear_sum = bias;
        for (int kk = 0; kk < K; kk++) {
            latent_sum[kk] = 0.0f;
            latent_sq_sum[kk] = 0.0f;
        }
        for (int j = 0; j < d; j++) {
            unsigned char b = xi[j];
            linear_sum += linear_table[(long)j * 256 + b];
            float *lat = latent_table + ((long)j * 256 + b) * K;
            for (int kk = 0; kk < K; kk++) {
                float v = lat[kk];
                latent_sum[kk] += v;
                latent_sq_sum[kk] += v * v;
            }
        }
        float interaction = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            float s = latent_sum[kk];
            interaction += s * s - latent_sq_sum[kk];
        }
        interaction *= 0.5f;
        float score = linear_sum + interaction;
        unsigned char best = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```