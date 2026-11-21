```hypothesis
Factorization Machine (FM) with low-rank embeddings (K=8) captures pairwise feature interactions crucial for tabular data (e.g., education×occupation) while maintaining O(d·K) inference cost. It should exceed label-linear's 0.8524 accuracy with similar CPU cost. Failure would be dev accuracy ≤0.8524 or instructions/item >1000.
```
```python
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    enc = []
    enc_sizes = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append((mu, sd))
            enc_sizes.append(1)
        else:
            vals = np.unique(X[:, j])
            enc.append(vals)
            enc_sizes.append(len(vals))

    d = sum(enc_sizes)
    col_start = []
    s = 0
    for sz in enc_sizes:
        col_start.append(s)
        s += sz

    n = X.shape[0]
    Z = np.zeros((n, d), dtype=np.float32)
    for j in range(X.shape[1]):
        t = card["input"]["types"][j]
        cs = col_start[j]
        if t == "ordinal":
            mu, sd = enc[j]
            Z[:, cs] = (X[:, j].astype(np.float32) - mu) / sd
        else:
            vals = enc[j]
            for idx, val in enumerate(vals):
                Z[X[:, j] == val, cs + idx] = 1.0

    w = np.zeros(d, np.float32)
    V = (rng.normal(0, 0.01, (d, K))).astype(np.float32)
    bias = np.float32(0.0)

    mw, mV, mb = np.zeros_like(w), np.zeros_like(V), np.float32(0.0)
    vw, vV, vb = np.zeros_like(w), np.zeros_like(V), np.float32(0.0)
    t = 0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]
            yy = y[idx]

            linear = z @ w + bias
            zV = z @ V
            sum_zV_sq = np.sum(zV * zV, axis=1)
            z_sq = z * z
            V_sq = V * V
            sum_z_sq_V_sq = z_sq @ V_sq
            sum_sum_z_sq_V_sq = np.sum(sum_z_sq_V_sq, axis=1)
            fm = 0.5 * (sum_zV_sq - sum_sum_z_sq_V_sq)
            logit = linear + fm

            p = 1.0 / (1.0 + np.exp(-logit))
            grad = (p - yy).astype(np.float32)

            gr_w = z.T @ grad + L2 * w
            gr_bias = grad.sum()
            gr_V = z.T @ (grad[:, None] * zV) - z_sq.T @ (grad[:, None] * V) + L2 * V

            t += 1
            for param, grad, m, v in [(w, gr_w, mw, vw), (V, gr_V, mV, vV), (bias, gr_bias, mb, vb)]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    d_orig = X.shape[1]
    linear_table = np.zeros((d_orig, 256), dtype=np.float32)
    emb_table = np.zeros((d_orig, 256, K), dtype=np.float32)

    for j in range(d_orig):
        t = card["input"]["types"][j]
        cs = col_start[j]
        if t == "ordinal":
            mu, sd = enc[j]
            for v in range(256):
                z_val = (v - mu) / sd
                linear_table[j, v] = w[cs] * z_val
                emb_table[j, v, :] = V[cs, :] * z_val
        else:
            vals = enc[j]
            for idx, val in enumerate(vals):
                linear_table[j, val] = w[cs + idx]
                emb_table[j, val, :] = V[cs + idx, :]

    out = struct.pack("<4i", 0x464D464D, d_orig, K, C)
    out += struct.pack("<f", bias)
    out += linear_table.astype("<f4").tobytes()
    out += emb_table.astype("<f4").tobytes()

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
    if (nw < 16 || hd[0] != 0x464D464D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d_orig = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K < 1 || d_orig != 14) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16 + 4 + 4L * d_orig * 256 + 4L * d_orig * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear_table = bias + 1;
    const float *emb_table = linear_table + d_orig * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d_orig) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d_orig); return 3; }

        float linear_sum = *bias;
        float emb_sum[16];
        float emb_sq_sum = 0.0f;
        for (int f = 0; f < K; f++) emb_sum[f] = 0.0f;

        for (int j = 0; j < d_orig; j++) {
            unsigned char v = xi[j];
            linear_sum += linear_table[j * 256 + v];
            const float *emb_row = emb_table + (j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float e = emb_row[f];
                emb_sum[f] += e;
                emb_sq_sum += e * e;
            }
        }

        float sum_emb_sq = 0.0f;
        for (int f = 0; f < K; f++) sum_emb_sq += emb_sum[f] * emb_sum[f];
        float fm = 0.5f * (sum_emb_sq - emb_sq_sum);
        float logit = linear_sum + fm;

        unsigned char best = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```