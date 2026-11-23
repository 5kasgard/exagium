```hypothesis
Family: label-fm (Factorization Machine with per-column lookup tables).
Why: The Adult dataset has mixed ordinal/categorical features with moderate dimensionality (~100 after one-hot). A degree-2 FM captures pairwise interactions efficiently via a low-rank latent space (k=8), which often outperforms linear models on tabular data while keeping inference fast (O(d·k) per item). The per-byte lookup tables avoid branching and make inference cache-friendly.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or inference cost exceeds label-linear by >2×, the FM does not justify its complexity.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Encoding: ordinal -> z-scored (1 dim), categorical -> one-hot over training values (255=unseen -> zero).
Model: two-class FM with latent dim k=8. Scores: w0 + w·x + 0.5*∑_f((∑_i V_{i,f}x_i)^2 - ∑_i (V_{i,f}x_i)^2).
Exported as per-column lookup tables: magic 0x464D5F31, d_cols, k, C; float32 bias[C]; float32 T[C][d_cols][256][1+k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d_col = card["input"]["bytes"]
    types = card["input"]["types"]
    n = X_bytes.shape[0]
    rng = np.random.default_rng(SEED)

    # Build feature mapping: for each column, list of feature indices and value functions
    col_feat_indices = []          # list of lists of absolute feature indices
    ordinal_mu = np.zeros(d_col, np.float64)
    ordinal_sd = np.ones(d_col, np.float64)
    cat_val_to_idx = [np.full(256, -1, dtype=np.int32) for _ in range(d_col)]
    d = 0
    for j, t in enumerate(types):
        col_vals = X_bytes[:, j]
        if t == "ordinal":
            mu, sd = col_vals.mean(), col_vals.std()
            if sd == 0:
                sd = 1.0
            ordinal_mu[j], ordinal_sd[j] = mu, sd
            col_feat_indices.append([d])
            d += 1
        else:
            uniq = np.unique(col_vals)
            mapping = {int(v): idx for idx, v in enumerate(uniq)}
            for v, idx in mapping.items():
                cat_val_to_idx[j][v] = d + idx
            col_feat_indices.append(list(range(d, d + len(uniq))))
            d += len(uniq)

    # Initialize FM parameters
    bias = np.zeros(C, np.float32)
    w = (rng.normal(0, 0.01, (C, d))).astype(np.float32)
    V = (rng.normal(0, 0.01, (C, d, K))).astype(np.float32)

    # Adam state
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    m_b, v_b = np.zeros_like(bias), np.zeros_like(bias)
    t = 0

    # Helper: compute scores for a batch of feature matrices Z (batch x d)
    def compute_scores(Z):
        # Z: (b, d) float32
        scores = np.zeros((Z.shape[0], C), np.float32)
        for c in range(C):
            linear = bias[c] + Z @ w[c]                       # (b,)
            ZV = Z @ V[c]                                     # (b, K)
            Z2V2 = (Z * Z) @ (V[c] * V[c])                    # (b, K)
            interaction = 0.5 * np.sum(ZV * ZV - Z2V2, axis=1)  # (b,)
            scores[:, c] = linear + interaction
        return scores

    # Training loop
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            # Build Z for this batch
            bsz = len(idx)
            Z = np.zeros((bsz, d), np.float32)
            for j in range(d_col):
                col_data = X_bytes[idx, j]
                if types[j] == "ordinal":
                    feat_idx = col_feat_indices[j][0]
                    Z[:, feat_idx] = (col_data - ordinal_mu[j]) / ordinal_sd[j]
                else:
                    for v in range(256):
                        fi = cat_val_to_idx[j][v]
                        if fi != -1:
                            mask = (col_data == v)
                            Z[mask, fi] = 1.0
            # Forward
            scores = compute_scores(Z)
            # Softmax + cross-entropy gradient
            max_s = scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), y[idx]] -= 1.0
            probs /= bsz  # (bsz, C)
            # Gradients
            grad_bias = probs.sum(axis=0).astype(np.float32)  # (C,)
            grad_w = np.zeros_like(w)
            grad_V = np.zeros_like(V)
            for c in range(C):
                g = probs[:, c]  # (bsz,)
                grad_w[c] = (Z.T @ g).astype(np.float32) + L2 * w[c]
                ZV = Z @ V[c]  # (bsz, K)
                # d/dV = Z.T @ (g[:,None] * (ZV - Z * V[c]?))
                # Actually: interaction = 0.5 * sum_f ( (ZV_f)^2 - Z2V2_f )
                # d/dV_{i,f} = sum_b g_b * (ZV_{b,f} * Z_{b,i} - Z_{b,i}^2 * V_{i,f})
                # = Z.T @ (g * ZV_f) - (Z^2).T @ (g * V_f)  ... let's derive carefully.
                # Let S_f = sum_i V_{i,f} x_i. Then interaction = 0.5 sum_f (S_f^2 - sum_i V_{i,f}^2 x_i^2).
                # d interaction / d V_{i,f} = S_f * x_i - V_{i,f} * x_i^2.
                # So for batch: dL/dV_{i,f} = sum_b g_b * (S_{b,f} * x_{b,i} - V_{i,f} * x_{b,i}^2) + L2 * V_{i,f}
                # We can compute S = ZV (bsz, K). Then:
                # term1 = Z.T @ (g[:,None] * S)   # (d, K)
                # term2 = (Z*Z).T @ (g[:,None] * V[c][None,:,:])  # (d, K)
                # grad_V[c] = term1 - term2 + L2 * V[c]
                term1 = (Z.T @ (g[:, None] * ZV)).astype(np.float32)
                term2 = ((Z * Z).T @ (g[:, None] * V[c])).astype(np.float32)
                grad_V[c] = term1 - term2 + L2 * V[c]
            # Adam update
            t += 1
            for param, grad, m, v in [(w, grad_w, m_w, v_w), (V, grad_V, m_V, v_V), (bias, grad_bias, m_b, v_b)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build lookup table T[C][d_col][256][1+K]
    T = np.zeros((C, d_col, 256, 1 + K), np.float32)
    for c in range(C):
        for j in range(d_col):
            for v in range(256):
                lin = 0.0
                V_sum = np.zeros(K, np.float32)
                if types[j] == "ordinal":
                    fi = col_feat_indices[j][0]
                    val = (v - ordinal_mu[j]) / ordinal_sd[j]
                    lin += w[c, fi] * val
                    V_sum += V[c, fi] * val
                else:
                    fi = cat_val_to_idx[j][v]
                    if fi != -1:
                        lin += w[c, fi]
                        V_sum += V[c, fi]
                T[c, j, v, 0] = lin
                T[c, j, v, 1:] = V_sum

    # Write weights.bin
    magic = 0x464D5F31  # "FM_1"
    hdr = struct.pack("<4i", magic, d_col, K, C)
    out = hdr + bias.astype("<f4").tobytes() + T.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d_col bytes (14 for adult); output is one byte class index. */
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
    if (nw < 16 || hd[0] != 0x464D5F31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d_col = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 16 + 4L * C + 4L * C * d_col * 256 * (1 + K);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *T = bias + C;  // [C][d_col][256][1+K]
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *sum_Vx = (float *)malloc(sizeof(float) * K);
    float *sum_Vx_sq = (float *)malloc(sizeof(float) * K);
    float score[256];  // C <= 256

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d_col) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d_col); return 3; }

        for (int c = 0; c < C; c++) {
            score[c] = bias[c];
            memset(sum_Vx, 0, sizeof(float) * K);
            memset(sum_Vx_sq, 0, sizeof(float) * K);
            for (int j = 0; j < d_col; j++) {
                unsigned char v = xi[j];
                size_t offset = ((size_t)c * d_col + j) * 256 * (1 + K) + v * (1 + K);
                score[c] += T[offset];
                for (int f = 0; f < K; f++) {
                    float a = T[offset + 1 + f];
                    sum_Vx[f] += a;
                    sum_Vx_sq[f] += a * a;
                }
            }
            float inter = 0.0f;
            for (int f = 0; f < K; f++) {
                inter += sum_Vx[f] * sum_Vx[f] - sum_Vx_sq[f];
            }
            score[c] += 0.5f * inter;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(sum_Vx); free(sum_Vx_sq);
    return 0;
}
```