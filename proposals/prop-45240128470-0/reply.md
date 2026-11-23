```hypothesis
Factorization Machine (degree 2) with 8-d embeddings captures pairwise feature interactions crucial for tabular data like Adult, while keeping inference lightweight (only 14*8 embeddings and 91 dot products per item). It should exceed label-linear's 0.8524 accuracy. If it fails to beat label-linear, it indicates interactions are not beneficial here or the linear model already saturates the signal.
```

```python
"""Factorization Machine (degree 2) for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Exports: magic, d, k, C, global_bias, linear[d][256], emb[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 100

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
    C = card["classes"]
    assert C == 2
    rng = np.random.default_rng(SEED)

    # Parameters: global_bias (1), linear[d][256], emb[d][256][K]
    global_bias = np.float32(0.0)
    linear = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    emb = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)

    # Adam state
    m_gb, v_gb = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)

            # Gather parameters for this batch
            lin_vals = linear[np.arange(d)[:, None], xb.T]          # (d, B)
            emb_vals = emb[np.arange(d)[:, None], xb.T]             # (d, B, K)

            # Linear part
            linear_sum = global_bias + lin_vals.sum(axis=0)         # (B,)

            # FM interaction: 0.5 * ( (sum emb)^2 - sum(emb^2) )
            sum_emb = emb_vals.sum(axis=0)                          # (B, K)
            sum_emb_sq = (emb_vals ** 2).sum(axis=0)                # (B, K)
            interaction = 0.5 * ((sum_emb ** 2 - sum_emb_sq).sum(axis=1))  # (B,)

            logits = linear_sum + interaction                       # (B,)
            # Sigmoid and gradient of binary cross-entropy
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = (p - yb).astype(np.float32) / len(yb)            # (B,)

            # Gradients
            g_gb = grad.sum()
            g_lin = np.zeros_like(linear)
            g_emb = np.zeros_like(emb)

            # For each feature, scatter-add gradients
            for j in range(d):
                vals = xb[:, j]                                     # (B,)
                np.add.at(g_lin[j], vals, grad)
                # emb grad: grad * (sum_emb - emb_j)
                # sum_emb is (B, K), emb_vals[j] is (B, K)
                g_emb_j = grad[:, None] * (sum_emb - emb_vals[j])   # (B, K)
                np.add.at(g_emb[j], vals, g_emb_j)

            # L2 regularization
            g_lin += L2 * linear
            g_emb += L2 * emb

            t += 1
            # Adam updates
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            # global bias
            m_gb = beta1 * m_gb + (1 - beta1) * g_gb
            v_gb = beta2 * v_gb + (1 - beta2) * g_gb * g_gb
            m_gb_hat = m_gb / (1 - beta1 ** t)
            v_gb_hat = v_gb / (1 - beta2 ** t)
            global_bias -= LR * m_gb_hat / (np.sqrt(v_gb_hat) + eps)

            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            m_lin_hat = m_lin / (1 - beta1 ** t)
            v_lin_hat = v_lin / (1 - beta2 ** t)
            linear -= LR * m_lin_hat / (np.sqrt(v_lin_hat) + eps)

            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            m_emb_hat = m_emb / (1 - beta1 ** t)
            v_emb_hat = v_emb / (1 - beta2 ** t)
            emb -= LR * m_emb_hat / (np.sqrt(v_emb_hat) + eps)

    # Write weights.bin
    # Format: magic(4), d(4), K(4), C(4), global_bias(4), linear[d*256](4), emb[d*256*K](4)
    magic = 0x464D3231  # "FM21"
    total_floats = 1 + d * 256 + d * 256 * K
    expected_bytes = 16 + 4 * total_floats
    out = struct.pack("<5i", magic, d, K, C, 0)  # placeholder for global_bias
    out = out[:16] + struct.pack("<f", global_bias) + out[20:]
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    assert len(out) == expected_bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine (degree 2) inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class index (0 or 1). */
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
    if (nw < 20 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || K < 1 || d < 1) { fprintf(stderr, "bad header\n"); return 3; }
    long total_floats = 1L + d * 256L + d * 256L * K;
    if (nw != 16 + 4L * total_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    float global_bias = p[0];
    const float *linear = p + 1;
    const float *emb = p + 1 + d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * K);
    float *sum_emb_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float lin_sum = global_bias;
        for (int j = 0; j < K; j++) { sum_emb[j] = 0.0f; sum_emb_sq[j] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            lin_sum += linear[(long)j * 256 + v];
            const float *e = emb + ((long)j * 256 + v) * K;
            for (int u = 0; u < K; u++) {
                float val = e[u];
                sum_emb[u] += val;
                sum_emb_sq[u] += val * val;
            }
        }

        float interaction = 0.0f;
        for (int u = 0; u < K; u++) {
            interaction += sum_emb[u] * sum_emb[u] - sum_emb_sq[u];
        }
        interaction *= 0.5f;

        float logit = lin_sum + interaction;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```