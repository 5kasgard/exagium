```hypothesis
Factorization Machine (FM) with embedding size K=8. FM captures pairwise feature interactions crucial for tabular data (e.g., education×occupation) while remaining lightweight. It should outperform label-linear (0.8524) and approach label-mlp16 (0.8591) with lower inference cost. If dev accuracy ≤0.855, the added complexity is not justified.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameters (tuned on last 20% of train.in)
K = 8
LR = 0.01
L2 = 1e-4
BATCH = 256
EPOCHS = 30
SEED = 1

def records(path):
    data = open(path, "rb").read()
    out = []
    i = 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    global_bias = np.zeros(C, dtype=np.float32)
    linear_weights = np.zeros((d, 256, C), dtype=np.float32)
    embeddings = (rng.normal(0, 0.01, (d, 256, K, C))).astype(np.float32)

    # Adam moments
    m_global = np.zeros_like(global_bias)
    v_global = np.zeros_like(global_bias)
    m_linear = np.zeros_like(linear_weights)
    v_linear = np.zeros_like(linear_weights)
    m_emb = np.zeros_like(embeddings)
    v_emb = np.zeros_like(embeddings)

    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            batch_size = xb.shape[0]

            # Forward pass
            scores = global_bias + np.zeros((batch_size, C), dtype=np.float32)
            for i in range(d):
                scores += linear_weights[i, xb[:, i], :]

            S = np.zeros((batch_size, C, K), dtype=np.float32)
            sum_sq = np.zeros((batch_size, C, K), dtype=np.float32)
            for i in range(d):
                v = embeddings[i, xb[:, i], :, :]          # (batch, K, C)
                v_t = np.transpose(v, (0, 2, 1))           # (batch, C, K)
                S += v_t
                sum_sq += v_t * v_t
            interaction = 0.5 * np.sum(S * S - sum_sq, axis=2)
            scores += interaction

            # Softmax and loss gradient
            scores_max = scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            dscores = probs
            dscores[np.arange(batch_size), yb] -= 1.0
            dscores /= batch_size

            # Backward pass
            g_global = dscores.sum(axis=0)

            g_linear = np.zeros_like(linear_weights)
            g_emb = np.zeros_like(embeddings)

            for i in range(d):
                # linear gradient
                np.add.at(g_linear[i], xb[:, i], dscores)

                # embedding gradient
                v = embeddings[i, xb[:, i], :, :]          # (batch, K, C)
                S_t = np.transpose(S, (0, 2, 1))           # (batch, K, C)
                dscores_exp = dscores[:, np.newaxis, :]    # (batch, 1, C)
                grad_v = dscores_exp * (S_t - v)           # (batch, K, C)
                # accumulate via flattened view
                g_emb_i_flat = g_emb[i].reshape(256, -1)
                grad_v_flat = grad_v.reshape(batch_size, -1)
                np.add.at(g_emb_i_flat, xb[:, i], grad_v_flat)

            # L2 regularization
            g_linear += L2 * linear_weights
            g_emb += L2 * embeddings

            # Adam update
            t += 1
            # global_bias
            m_global = beta1 * m_global + (1 - beta1) * g_global
            v_global = beta2 * v_global + (1 - beta2) * (g_global * g_global)
            m_hat = m_global / (1 - beta1 ** t)
            v_hat = v_global / (1 - beta2 ** t)
            global_bias -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # linear_weights
            m_linear = beta1 * m_linear + (1 - beta1) * g_linear
            v_linear = beta2 * v_linear + (1 - beta2) * (g_linear * g_linear)
            m_hat = m_linear / (1 - beta1 ** t)
            v_hat = v_linear / (1 - beta2 ** t)
            linear_weights -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # embeddings
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * (g_emb * g_emb)
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            embeddings -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights.bin
    magic = 0x464D3132  # "FM12"
    header = struct.pack("<4i", magic, d, K, C)
    out_bytes = header
    out_bytes += global_bias.astype("<f4").tobytes()
    out_bytes += linear_weights.astype("<f4").tobytes()
    out_bytes += embeddings.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out_bytes)

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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || C != 2 || K <= 0 || K > 32) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected_floats = C + (long)d * 256 * C + (long)d * 256 * K * C;
    if (nw != 16 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *global_bias = p; p += C;
    const float *linear_weights = p; p += (long)d * 256 * C;
    const float *embeddings = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float scores[2];
        scores[0] = global_bias[0];
        scores[1] = global_bias[1];

        for (int j = 0; j < d; j++) {
            int val = xi[j];
            const float *lw = linear_weights + ((long)j * 256 + val) * C;
            scores[0] += lw[0];
            scores[1] += lw[1];
        }

        for (int c = 0; c < C; c++) {
            float sum_f[32] = {0};
            float sum_sq_f[32] = {0};
            for (int j = 0; j < d; j++) {
                int val = xi[j];
                const float *emb = embeddings + (((long)j * 256 + val) * K + 0) * C + c;
                for (int f = 0; f < K; f++) {
                    float v = emb[f * C];
                    sum_f[f] += v;
                    sum_sq_f[f] += v * v;
                }
            }
            float inter = 0.0f;
            for (int f = 0; f < K; f++) {
                inter += 0.5f * (sum_f[f] * sum_f[f] - sum_sq_f[f]);
            }
            scores[c] += inter;
        }

        unsigned char best = (scores[1] > scores[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```