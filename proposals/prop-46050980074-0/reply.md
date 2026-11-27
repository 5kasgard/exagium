```hypothesis
Family: label-poly2 (Polynomial Logistic Regression with 2nd-order ordinal interactions).
Why: The Adult dataset has 6 ordinal (quantile-binned numeric) features. A linear model misses pairwise interactions like age×hours-per-week. Adding only 15 interaction features (z_i * z_j) gives the simplest non-linearity, keeping inference at ~60 FLOPs/item (near linear cost) while capturing curvature. The MLP (0.8591) shows non-linearity helps; this adds it with minimal overhead.
Failure: If dev accuracy ≤ 0.8524 (linear baseline), interactions don't help or overfit; if inference cost > 2× linear, the approach is too heavy.
```
```python
"""label-poly2: Linear model + 2nd-order ordinal interactions.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: int32 magic=0x504F4C32, d, H=0, C, n_ord, ord_idx[6];
float32 mu[6], sd[6]; float32 T[d][256][C]; float32 bias[C]; float32 inter_w[15][C]."""
import json, math, struct, sys, itertools

import numpy as np

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
    types = card["input"]["types"]
    d = len(types)

    # Identify ordinal/categorical indices
    ord_indices = [j for j, t in enumerate(types) if t == "ordinal"]
    cat_indices = [j for j, t in enumerate(types) if t == "categorical"]
    n_ord = len(ord_indices)  # 6
    n_cat = len(cat_indices)  # 8

    rng = np.random.default_rng(SEED)

    # Compute mu, sd for ordinal features
    mu = np.zeros(n_ord, np.float64)
    sd = np.zeros(n_ord, np.float64)
    for i, j in enumerate(ord_indices):
        v = X[:, j].astype(np.float64)
        mu[i] = v.mean()
        sd[i] = v.std() or 1.0

    # Build main-effect encoding matrices E_j (256 x K_j)
    enc = []  # list of (256, K_j) for each feature in order 0..d-1
    cat_offsets = {}  # feature j -> (start_col, K_j)
    col = 0
    for j in range(d):
        if j in ord_indices:
            # ordinal: single column, encoding is z-score
            idx = ord_indices.index(j)
            z = (np.arange(256, dtype=np.float64) - mu[idx]) / sd[idx]
            E = z[:, None]  # (256, 1)
            enc.append(E)
            cat_offsets[j] = (col, 1)
            col += 1
        else:
            # categorical: one-hot over training values
            vals = np.unique(X[:, j])
            K = len(vals)
            E = np.zeros((256, K), dtype=np.float64)
            E[vals, np.arange(K)] = 1.0
            enc.append(E)
            cat_offsets[j] = (col, K)
            col += K
    k_main = col

    # Build Z_main (n x k_main)
    Z_main = np.empty((X.shape[0], k_main), dtype=np.float32)
    for j in range(d):
        start, K = cat_offsets[j]
        Z_main[:, start:start+K] = enc[j][X[:, j]]

    # Build interaction features: 15 pairs of ordinal features
    ord_pairs = list(itertools.combinations(range(n_ord), 2))  # 15 pairs
    Z_inter = np.empty((X.shape[0], len(ord_pairs)), dtype=np.float32)
    for p, (a, b) in enumerate(ord_pairs):
        # z-scored values for ordinal features a and b
        ja, jb = ord_indices[a], ord_indices[b]
        za = (X[:, ja].astype(np.float32) - mu[a]) / sd[a]
        zb = (X[:, jb].astype(np.float32) - mu[b]) / sd[b]
        Z_inter[:, p] = za * zb

    Z = np.hstack([Z_main, Z_inter]).astype(np.float32)
    n, k = Z.shape

    # Initialize weights
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # softmax gradient
            S = z @ W + b
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            # gradients
            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Split weights
    W_main = W[:k_main]
    W_inter = W[k_main:]

    # Build per-byte lookup tables T[j][v] = contribution to logits (C-dim)
    T = np.empty((d, 256, C), dtype=np.float32)
    for j in range(d):
        start, K = cat_offsets[j]
        if K == 1:  # ordinal
            w = W_main[start]  # (C,)
            idx = ord_indices.index(j)
            z_vals = (np.arange(256, dtype=np.float32) - mu[idx]) / sd[idx]
            T[j] = z_vals[:, None] * w[None, :]
        else:  # categorical
            # W_main block is (K, C), one-hot picks row
            # For unseen (255), contribution is zero (already zero-initialized)
            T[j] = 0.0
            vals = np.unique(X[:, j])
            for v_idx, v in enumerate(vals):
                T[j, v] = W_main[start + v_idx]

    # Write weights.bin
    magic = 0x504F4C32  # 'POL2'
    header = struct.pack("<6i", magic, d, 0, C, n_ord, *ord_indices)
    mu_arr = mu.astype(np.float32)
    sd_arr = sd.astype(np.float32)
    bias_arr = b.astype(np.float32)
    inter_arr = W_inter.astype(np.float32)  # (15, C)
    # Flatten T: d * 256 * C floats
    T_flat = T.astype("<f4").tobytes()
    out = (header +
           mu_arr.astype("<f4").tobytes() +
           sd_arr.astype("<f4").tobytes() +
           T_flat +
           bias_arr.astype("<f4").tobytes() +
           inter_arr.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-poly2 infer: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class (argmax).
 * Weights format (little-endian):
 *   int32 magic=0x504F4C32, d, H=0, C, n_ord, ord_idx[6]
 *   float32 mu[6], sd[6]
 *   float32 T[d][256][C]
 *   float32 bias[C]
 *   float32 inter_w[15][C] */
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 24 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], n_ord = hd[4];
    if (H != 0 || C < 1 || C > 256 || n_ord != 6 || d != 14) { fprintf(stderr, "bad header\n"); return 3; }
    int32_t *ord_idx = hd + 5;
    float *ptr = (float *)(ord_idx + n_ord);
    float *mu = ptr; ptr += n_ord;
    float *sd = ptr; ptr += n_ord;
    float *T = ptr; ptr += (long)d * 256 * C;
    float *bias = ptr; ptr += C;
    float *inter_w = ptr; ptr += 15 * C;
    if ((unsigned char *)ptr != wb + nw) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[2];  // C=2
    float z_ord[6];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // init scores with bias
        scores[0] = bias[0]; scores[1] = bias[1];

        // main effects
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((long)j * 256 + xi[j]) * C;
            scores[0] += tj[0]; scores[1] += tj[1];
        }

        // compute z for ordinal features
        for (int o = 0; o < 6; o++) {
            int j = ord_idx[o];
            z_ord[o] = (xi[j] - mu[o]) / sd[o];
        }

        // 15 interactions
        int p = 0;
        for (int a = 0; a < 6; a++) {
            for (int b = a + 1; b < 6; b++) {
                float prod = z_ord[a] * z_ord[b];
                const float *iw = inter_w + p * C;
                scores[0] += prod * iw[0];
                scores[1] += prod * iw[1];
                p++;
            }
        }

        unsigned char best = (scores[1] > scores[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```