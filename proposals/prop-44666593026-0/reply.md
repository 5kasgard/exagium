```hypothesis
Factorization Machines (FM) capture pairwise feature interactions via low-rank latent vectors, which are crucial for the Adult dataset's mix of ordinal and categorical features. With a small latent dimension (k=8), the model remains fast at inference (similar to linear) while significantly improving accuracy over logistic regression. If dev accuracy does not exceed label-linear (0.8524), the hypothesis fails.
```

```python
"""Factorization Machine for tabular classification with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: int32 magic 0x464D3132, d, k, C; float32 w0, linear[d][256], latent[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8                 # latent dimension
SEED = 1
LR = 0.01
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
BATCH = 64
EPOCHS = 24           # ceil(600000 / 26048) ≈ 24


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2, "FM binary only"

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # parameters
    w0 = np.float32(0.0)
    linear = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    latent = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_lat, v_lat = np.zeros_like(latent), np.zeros_like(latent)

    t = 0
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            b = xb.shape[0]

            # forward
            lin_term = w0 + np.sum(linear[np.arange(d), xb.T], axis=0)  # (b,)
            lat_sum = np.zeros((b, K), np.float32)
            lat_sq_sum = np.zeros((b, K), np.float32)
            for j in range(d):
                lv = latent[j, xb[:, j], :]
                lat_sum += lv
                lat_sq_sum += lv * lv
            inter = 0.5 * (np.sum(lat_sum * lat_sum, axis=1) - np.sum(lat_sq_sum, axis=1))
            logit = lin_term + inter
            prob = sigmoid(logit)
            # loss (not used for gradients directly)
            # backward
            dlogit = (prob - yb) / b  # (b,)

            # gradients
            g_w0 = np.sum(dlogit)
            g_lin = np.zeros_like(linear)
            g_lat = np.zeros_like(latent)
            for j in range(d):
                np.add.at(g_lin[j], xb[:, j], dlogit)
                for i in range(b):
                    v = xb[i, j]
                    g_lat[j, v] += dlogit[i] * (lat_sum[i] - latent[j, v])

            t += 1
            # Adam updates
            # w0
            m_w0 = BETA1 * m_w0 + (1 - BETA1) * g_w0
            v_w0 = BETA2 * v_w0 + (1 - BETA2) * g_w0 * g_w0
            m_w0_hat = m_w0 / (1 - BETA1 ** t)
            v_w0_hat = v_w0 / (1 - BETA2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + EPS)

            # linear
            m_lin = BETA1 * m_lin + (1 - BETA1) * g_lin
            v_lin = BETA2 * v_lin + (1 - BETA2) * g_lin * g_lin
            m_lin_hat = m_lin / (1 - BETA1 ** t)
            v_lin_hat = v_lin / (1 - BETA2 ** t)
            linear -= LR * m_lin_hat / (np.sqrt(v_lin_hat) + EPS)

            # latent
            m_lat = BETA1 * m_lat + (1 - BETA1) * g_lat
            v_lat = BETA2 * v_lat + (1 - BETA2) * g_lat * g_lat
            m_lat_hat = m_lat / (1 - BETA1 ** t)
            v_lat_hat = v_lat / (1 - BETA2 ** t)
            latent -= LR * m_lat_hat / (np.sqrt(v_lat_hat) + EPS)

    # write weights
    magic = 0x464D3132  # "FM12"
    out = struct.pack("<4i", magic, d, K, C)
    out += struct.pack("<f", w0)
    out += linear.astype("<f4").tobytes()
    out += latent.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (one per feature). Output: one byte class (0/1). */
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
    if (hd[0] != 0x464D3132) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d <= 0 || K <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    size_t expected = 16 + 4 + (size_t)d * 256 * 4 + (size_t)d * 256 * K * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = *p++;                     // bias
    const float *linear = p; p += (size_t)d * 256;
    const float *latent = p;             // [d][256][K]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lat_sum = malloc(sizeof(float) * K);
    float *lat_sq_sum = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin_term = w0;
        for (int j = 0; j < K; j++) { lat_sum[j] = 0.0f; lat_sq_sum[j] = 0.0f; }

        for (int j = 0; j < d; j++) {
            int v = xi[j];
            lin_term += linear[(size_t)j * 256 + v];
            const float *lv = latent + ((size_t)j * 256 + v) * K;
            for (int u = 0; u < K; u++) {
                float val = lv[u];
                lat_sum[u] += val;
                lat_sq_sum[u] += val * val;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < K; u++) {
            inter += lat_sum[u] * lat_sum[u] - lat_sq_sum[u];
        }
        inter *= 0.5f;
        float logit = lin_term + inter;
        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```