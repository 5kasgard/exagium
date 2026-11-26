```hypothesis
Family: label-fm (Factorization Machine with shared interaction layer, K=4).
Why: Adult income depends on feature crosses (e.g., education×occupation, marital×sex) that linear models miss. A low-rank FM captures all pairwise interactions in O(K·d) time/params. Sharing the K=4 interaction layer across the 2 classes keeps weights tiny (~140 KB) and inference fast (≈14·6=84 FLOPs/item). Precomputed per-byte lookups mirror label-linear's speed.
Failure signal: dev accuracy ≤ 0.8524 (label-linear) or instructions/item ≥ label-linear.
```
```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: per-card types (ordinal z-scored, categorical one-hot).
Model: shared K=4 latent factors + per-class linear weights + bias.
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Export: magic 0x464D3131, d, K, C; float32 bias[C], lin[d][256][C], lat[d][256][K], latsq[d][256][K]."""
import json, math, struct, sys

import numpy as np

K = 4
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
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    # Build encoders per original feature
    enc = []
    enc_dim = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)  # (256, 1)
            enc.append(E)
            enc_dim.append(1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
            enc_dim.append(len(vals))

    # Total encoded dimension D
    D = sum(enc_dim)
    offsets = np.cumsum([0] + enc_dim)  # length d+1

    # Initialize parameters
    # V: (D, K) shared latent factors
    # W_lin: (D, C) per-class linear weights
    # bias: (C,)
    scale = math.sqrt(6.0 / (D + K))
    V = (rng.uniform(-1, 1, (D, K)) * scale).astype(np.float32)
    W_lin = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    bias = np.zeros(C, np.float32)

    params = [V, W_lin, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Encode full training set to Z (n, D)
    Z_list = []
    for j in range(d):
        Z_list.append(enc[j][X[:, j]])
    Z = np.concatenate(Z_list, axis=1).astype(np.float32)  # (n, D)
    n = Z.shape[0]

    # Precompute Z squared for latsq term
    Z_sq = Z * Z

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]          # (B, D)
            z_sq = Z_sq[idx]    # (B, D)
            yy = y[idx]         # (B,)

            # Forward: compute logits for each class
            # Interaction part (shared): 0.5 * sum_f ( (z @ V[:,f])^2 - z_sq @ (V[:,f]^2) )
            zV = z @ V          # (B, K)
            zV_sq = zV * zV
            z_sq_V_sq = z_sq @ (V * V)  # (B, K)
            inter = 0.5 * np.sum(zV_sq - z_sq_V_sq, axis=1, keepdims=True)  # (B, 1)

            # Linear part per class
            lin = z @ W_lin     # (B, C)

            logits = lin + inter + bias  # (B, C)

            # Softmax cross-entropy gradient
            max_logit = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_logits = (probs / len(yy)).astype(np.float32)  # (B, C)

            # Gradients
            # grad_bias = sum grad_logits
            grad_bias = grad_logits.sum(axis=0)

            # grad_W_lin = z.T @ grad_logits + L2 * W_lin
            grad_W_lin = z.T @ grad_logits + L2 * W_lin

            # grad_V: interaction gradient
            # d(inter)/dV = z.T @ (grad_logits_sum * zV) - z_sq.T @ (grad_logits_sum * V)
            # where grad_logits_sum = sum_c grad_logits[:,c] (B,)
            g_sum = grad_logits.sum(axis=1)  # (B,)
            # (B, K) * (B, 1) -> (B, K)
            g_zV = g_sum[:, None] * zV
            g_V = z.T @ g_zV - z_sq.T @ (g_sum[:, None] * V) + L2 * V

            grads = [g_V, grad_W_lin, grad_bias]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export per-byte lookup tables
    # bias[C]
    # lin[d][256][C]
    # lat[d][256][K]
    # latsq[d][256][K]
    lin_table = np.zeros((d, 256, C), np.float32)
    lat_table = np.zeros((d, 256, K), np.float32)
    latsq_table = np.zeros((d, 256, K), np.float32)

    for j in range(d):
        off = offsets[j]
        dim = enc_dim[j]
        E = enc[j]  # (256, dim)
        W_slice = W_lin[off:off+dim]      # (dim, C)
        V_slice = V[off:off+dim]          # (dim, K)
        V_sq_slice = V_slice * V_slice    # (dim, K)

        # Linear contribution: E @ W_slice -> (256, C)
        lin_table[j] = E @ W_slice
        # Latent contribution: E @ V_slice -> (256, K)
        lat_table[j] = E @ V_slice
        # Latent squared contribution: E_sq @ V_sq_slice
        # For ordinal: E is (256,1), E_sq = E*E. For categorical: E is one-hot, E_sq = E.
        E_sq = E * E
        latsq_table[j] = E_sq @ V_sq_slice

    # Pack: magic, d, K, C, bias[C], lin[d*256*C], lat[d*256*K], latsq[d*256*K]
    magic = 0x464D3131  # 'FM11'
    header = struct.pack("<4i", magic, d, K, C)
    out = header
    out += bias.astype("<f4").tobytes()
    out += lin_table.astype("<f4").tobytes()
    out += lat_table.astype("<f4").tobytes()
    out += latsq_table.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class index (argmax over C logits).
 * Weights layout (little-endian):
 *   int32 magic=0x464D3131, d, K, C
 *   float32 bias[C]
 *   float32 lin[d][256][C]
 *   float32 lat[d][256][K]
 *   float32 latsq[d][256][K] */
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
    if (nw < 16 || hd[0] != 0x464D3131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 256) { fprintf(stderr, "bad dims\n"); return 3; }

    size_t expected = 16
        + (size_t)C * 4
        + (size_t)d * 256 * C * 4
        + (size_t)d * 256 * K * 4
        + (size_t)d * 256 * K * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *lin = bias + C;
    const float *lat = lin + (size_t)d * 256 * C;
    const float *latsq = lat + (size_t)d * 256 * K;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = calloc(K, sizeof(float));
    float *sum_sq = calloc(K, sizeof(float));
    float *score = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) score[c] = bias[c];
        for (int f = 0; f < K; f++) { acc[f] = 0.0f; sum_sq[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_j = lin + ((size_t)j * 256 + v) * C;
            const float *lat_j = lat + ((size_t)j * 256 + v) * K;
            const float *latsq_j = latsq + ((size_t)j * 256 + v) * K;
            for (int c = 0; c < C; c++) score[c] += lin_j[c];
            for (int f = 0; f < K; f++) {
                acc[f] += lat_j[f];
                sum_sq[f] += latsq_j[f];
            }
        }
        for (int f = 0; f < K; f++) {
            float inter = 0.5f * (acc[f] * acc[f] - sum_sq[f]);
            for (int c = 0; c < C; c++) score[c] += inter;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```