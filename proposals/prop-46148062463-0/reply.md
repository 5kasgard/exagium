```hypothesis
Family: label-fm (Factorization Machine with per-feature latent vectors).
Why: Adult data has 14 mixed ordinal/categorical features; pairwise interactions capture non-linearities missed by linear models.
A rank-8 FM adds only ~14*256*8 ≈ 28k parameters, keeping inference fast (O(d·k) per item). If dev accuracy ≤ label-linear (0.8524) or cost exceeds label-linear by >2×, the approach fails.
```
```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Exports: magic 0x464D31, d, k, C; float32 T_lin[d][256][C], V[d][256][k], W2[k][C], b[C]."""
import json, math, struct, sys

import numpy as np

K = 8
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

    # Determine ordinal stats and categorical cardinalities
    ordinal_info = []
    cat_sizes = []
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            ordinal_info.append((mu, sd))
            cat_sizes.append(1)  # placeholder
        else:
            ordinal_info.append(None)
            vals = np.unique(X[:, j])
            cat_sizes.append(max(vals) + 1 if len(vals) else 1)

    # Parameters
    # For ordinal: base linear weight (C), base latent vector (K)
    # For categorical: full tables (256, C) and (256, K)
    base_w_lin = np.zeros((d, C), np.float32)
    base_v = np.zeros((d, K), np.float32)
    W_lin_cat = np.zeros((d, 256, C), np.float32)
    V_cat = np.zeros((d, 256, K), np.float32)

    # Initialize categorical tables with small random values
    for j in range(d):
        if types[j] == "categorical":
            W_lin_cat[j] = (rng.uniform(-1, 1, (256, C)) * math.sqrt(6.0 / 256)).astype(np.float32)
            V_cat[j] = (rng.uniform(-1, 1, (256, K)) * math.sqrt(6.0 / 256)).astype(np.float32)
        else:
            base_w_lin[j] = (rng.uniform(-1, 1, C) * math.sqrt(6.0 / 1)).astype(np.float32)
            base_v[j] = (rng.uniform(-1, 1, K) * math.sqrt(6.0 / 1)).astype(np.float32)

    W2 = (rng.uniform(-1, 1, (K, C)) * math.sqrt(6.0 / K)).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Collect all parameters for Adam
    params = []
    if any(t == "ordinal" for t in types):
        params.append(base_w_lin)
        params.append(base_v)
    params.append(W_lin_cat)
    params.append(V_cat)
    params.append(W2)
    params.append(b)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    steps_per_epoch = (n + BATCH - 1) // BATCH
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    def softmax_grad(scores, yy):
        # scores: (batch, C), yy: (batch,)
        probs = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return probs / len(yy)  # (batch, C)

    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward
            linear = np.zeros((B, C), np.float32)
            sum_v = np.zeros((B, K), np.float32)
            sum_v_sq = np.zeros((B, K), np.float32)

            for j in range(d):
                col = xb[:, j]
                if types[j] == "ordinal":
                    mu, sd = ordinal_info[j]
                    scale = ((col.astype(np.float32) - mu) / sd)[:, None]  # (B,1)
                    linear += scale * base_w_lin[j]          # (B,C)
                    v_contrib = scale * base_v[j]            # (B,K)
                    sum_v += v_contrib
                    sum_v_sq += v_contrib * v_contrib
                else:
                    linear += W_lin_cat[j, col]              # (B,C)
                    v_contrib = V_cat[j, col]                # (B,K)
                    sum_v += v_contrib
                    sum_v_sq += v_contrib * v_contrib

            interaction = 0.5 * (sum_v * sum_v - sum_v_sq)   # (B,K)
            scores = linear + interaction @ W2 + b           # (B,C)

            # Backward
            g = softmax_grad(scores, yb)                     # (B,C)

            # Gradients for output layer
            g_W2 = interaction.T @ g                         # (K,C)
            g_b = g.sum(axis=0)                              # (C,)

            # Gradient w.r.t interaction
            g_inter = g @ W2.T                               # (B,K)

            # Gradient w.r.t sum_v and sum_v_sq
            g_sum_v = g_inter * sum_v                        # (B,K)
            g_sum_v_sq = -0.5 * g_inter                      # (B,K)

            # Accumulate gradients for latent vectors
            g_V_cat = np.zeros_like(V_cat)
            g_base_v = np.zeros_like(base_v)
            g_W_lin_cat = np.zeros_like(W_lin_cat)
            g_base_w_lin = np.zeros_like(base_w_lin)

            for j in range(d):
                col = xb[:, j]
                if types[j] == "ordinal":
                    mu, sd = ordinal_info[j]
                    scale = ((col.astype(np.float32) - mu) / sd)[:, None]  # (B,1)
                    # linear part
                    g_base_w_lin[j] += (scale * g).sum(axis=0)
                    # latent part
                    g_v = scale * (g_sum_v - 2 * scale * sum_v * g_sum_v_sq)  # (B,K)
                    g_base_v[j] += g_v.sum(axis=0)
                else:
                    # linear part
                    np.add.at(g_W_lin_cat[j], col, g)
                    # latent part
                    g_v = g_sum_v - 2 * V_cat[j, col] * g_sum_v_sq
                    np.add.at(g_V_cat[j], col, g_v)

            # L2 regularization gradients
            if any(t == "ordinal" for t in types):
                g_base_w_lin += L2 * base_w_lin
                g_base_v += L2 * base_v
            g_W_lin_cat += L2 * W_lin_cat
            g_V_cat += L2 * V_cat
            g_W2 += L2 * W2

            # Adam update
            grads = []
            if any(t == "ordinal" for t in types):
                grads.append(g_base_w_lin)
                grads.append(g_base_v)
            grads.append(g_W_lin_cat)
            grads.append(g_V_cat)
            grads.append(g_W2)
            grads.append(g_b)

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build export tables T_lin[d][256][C] and V_table[d][256][K]
    T_lin = np.zeros((d, 256, C), np.float32)
    V_table = np.zeros((d, 256, K), np.float32)
    for j in range(d):
        if types[j] == "ordinal":
            mu, sd = ordinal_info[j]
            for val in range(256):
                scale = (val - mu) / sd
                T_lin[j, val] = scale * base_w_lin[j]
                V_table[j, val] = scale * base_v[j]
        else:
            T_lin[j] = W_lin_cat[j]
            V_table[j] = V_cat[j]

    # Write weights.bin
    magic = 0x464D31  # "FM1"
    header = struct.pack("<4i", magic, d, K, C)
    body = (T_lin.astype("<f4").tobytes() +
            V_table.astype("<f4").tobytes() +
            W2.astype("<f4").tobytes() +
            b.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + (long)d * 256 * K + (long)K * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T_lin = (const float *)(hd + 4);
    const float *V_table = T_lin + (long)d * 256 * C;
    const float *W2 = V_table + (long)d * 256 * K;
    const float *b = W2 + (long)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v_sq = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int u = 0; u < K; u++) { sum_v[u] = 0.0f; sum_v_sq[u] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char byte = xi[j];
            const float *tlin = T_lin + ((long)j * 256 + byte) * C;
            for (int c = 0; c < C; c++) scores[c] += tlin[c];

            const float *v = V_table + ((long)j * 256 + byte) * K;
            for (int u = 0; u < K; u++) {
                float vu = v[u];
                sum_v[u] += vu;
                sum_v_sq[u] += vu * vu;
            }
        }

        for (int u = 0; u < K; u++) {
            float inter = 0.5f * (sum_v[u] * sum_v[u] - sum_v_sq[u]);
            for (int c = 0; c < C; c++) scores[c] += inter * W2[(long)u * C + c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```