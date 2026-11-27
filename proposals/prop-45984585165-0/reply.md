```hypothesis
Factorization Machine (FM) with latent dimension 8 captures pairwise feature interactions
(e.g., education×occupation) that the linear model misses, while staying small and fast.
If dev accuracy does not exceed label-linear (0.8524), interactions are not beneficial
or rank 8 is insufficient for this task.
```

```python
"""label-fm: Factorization Machine for tabular label tasks with byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Exports per-byte lookup tables for w and V."""
import json, math, struct, sys

import numpy as np

K = 8                 # latent dimension
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)   # 0 or 1
    d = card["input"]["bytes"]                      # 14
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Map each (feature, value) to a flat index: 0 .. d*256-1
    flat_idx = X + (np.arange(d, dtype=np.uint16) * 256)   # shape (n, d)
    n_feat = d * 256

    # Parameters
    w0 = np.float32(0.0)
    w = np.zeros(n_feat, dtype=np.float32)
    V = (rng.normal(0, 0.01, (n_feat, K))).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            idx_flat = flat_idx[idx]           # (B, d)
            yy = y[idx]                        # (B,)

            # Linear term
            lin = w0 + w[idx_flat].sum(axis=1)                 # (B,)

            # Interaction term: 0.5 * sum_l ( (sum_i V_i,l)^2 - sum_i V_i,l^2 )
            V_batch = V[idx_flat]                              # (B, d, K)
            sum_V = V_batch.sum(axis=1)                        # (B, K)
            sum_V2 = (V_batch * V_batch).sum(axis=1)           # (B, K)
            inter = 0.5 * ((sum_V * sum_V - sum_V2).sum(axis=1))   # (B,)

            logits = lin + inter                               # (B,)

            # Logistic loss gradients
            # p = sigmoid(logits) for class 1
            p = 1.0 / (1.0 + np.exp(-logits))
            grad_logit = (p - yy).astype(np.float32)           # (B,)

            # Gradient w.r.t w0
            g_w0 = grad_logit.mean()

            # Gradient w.r.t w: each active feature gets grad_logit
            g_w = np.zeros_like(w)
            np.add.at(g_w, idx_flat.ravel(), np.repeat(grad_logit, d))
            g_w /= BATCH
            g_w += L2 * w

            # Gradient w.r.t V
            # For each example b and feature j: V_ij contributes to inter via sum_V and sum_V2
            # d_inter/d_V_ij,l = sum_V[b,l] - V_ij,l
            g_V = np.zeros_like(V)
            # sum_V[b,l] broadcast to (B, d, K)
            coeff = sum_V[:, None, :] - V_batch                # (B, d, K)
            # Multiply by grad_logit[:, None, None]
            coeff *= grad_logit[:, None, None]
            np.add.at(g_V, idx_flat.ravel(), coeff.reshape(-1, K))
            g_V /= BATCH
            g_V += L2 * V

            t += 1
            # Adam updates
            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_w0_hat / (np.sqrt(v_w0_hat) + eps)

            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)

            # V
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            V -= LR * m_V_hat / (np.sqrt(v_V_hat) + eps)

    # Export weights as per-byte lookup tables
    # Layout: magic, d, K, C, w0, w[d][256], V[d][256][K]
    C = card["classes"]
    magic = 0x464D424C  # 'LFML' little-endian
    w_table = w.reshape(d, 256).astype(np.float32)
    V_table = V.reshape(d, 256, K).astype(np.float32)

    out = struct.pack("<4i", magic, d, K, C)
    out += struct.pack("<f", w0)
    out += w_table.tobytes()
    out += V_table.tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class (0 or 1). */
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
    uint32_t magic = wb[0] | wb[1]<<8 | wb[2]<<16 | wb[3]<<24;
    if (magic != 0x464D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t d = *(int32_t*)(wb+4);
    int32_t K = *(int32_t*)(wb+8);
    int32_t C = *(int32_t*)(wb+12);
    if (C != 2 || d != 14 || K != 8) { fprintf(stderr, "unexpected dims\n"); return 3; }

    size_t expected = 16 + 4 + 4*d*256 + 4*d*256*K;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float*)(wb + 16);
    float w0 = *p++;
    const float *w_table = p; p += d * 256;
    const float *V_table = p; /* p += d * 256 * K; */

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_V = calloc(K, sizeof(float));
    float *sum_V2 = calloc(K, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = w0;
        memset(sum_V, 0, K * sizeof(float));
        memset(sum_V2, 0, K * sizeof(float));

        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            linear += w_table[j * 256 + val];
            const float *Vj = V_table + (j * 256 + val) * K;
            for (int l = 0; l < K; l++) {
                float v = Vj[l];
                sum_V[l] += v;
                sum_V2[l] += v * v;
            }
        }

        float inter = 0.0f;
        for (int l = 0; l < K; l++) {
            inter += sum_V[l] * sum_V[l] - sum_V2[l];
        }
        inter *= 0.5f;

        float score1 = linear + inter;
        unsigned char best = (score1 > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }

    free(sum_V); free(sum_V2);
    return 0;
}
```