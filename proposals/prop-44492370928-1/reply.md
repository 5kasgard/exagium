```hypothesis
Factorization Machine (FM) with k=8 latent factors captures pairwise feature interactions
efficiently, crucial for Adult where combinations (education×occupation, marital×sex) predict
income. It adds only O(k·d) params over linear, keeping inference ~200 Ir/item. If dev
accuracy ≤ 0.8524 (label-linear) or latency ≥ 2× label-linear, the hypothesis fails.
```
```python
"""Factorization Machine for tabular label tasks: FM(w, V) with k factors.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encoding: ordinal z-scored, categorical one-hot over training values (255=unseen→zero).
Exports per-input-byte lookup tables for linear, factor, and squared-factor contributions."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 256, 300000, 50


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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.float32)  # 0/1
    types = card["input"]["types"]
    d_in = X.shape[1]
    rng = np.random.default_rng(SEED)

    # Build encoders per column and column->feature index map
    enc = []
    col_feat = []
    feat_idx = 0
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)  # (256,1)
            enc.append(E)
            col_feat.append([feat_idx])
            feat_idx += 1
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
            col_feat.append(list(range(feat_idx, feat_idx + len(vals))))
            feat_idx += len(vals)
    d = feat_idx

    # Concatenate to design matrix Z (n, d)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1)

    n = Z.shape[0]
    # FM parameters
    w0 = 0.0
    w = rng.normal(0, 0.01, d).astype(np.float32)
    V = rng.normal(0, 0.01, (d, K)).astype(np.float32)

    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = Z[idx]                 # (B, d)
            yb = y[idx]                # (B,)
            B = len(idx)

            # Forward
            linear = x @ w                          # (B,)
            S = x @ V                               # (B, K)
            x2 = x * x
            V2 = V * V
            Q = x2 @ V2                             # (B, K)
            inter = 0.5 * np.sum(S * S - Q, axis=1) # (B,)
            score = w0 + linear + inter             # (B,)

            # Logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-score))
            g = (p - yb).astype(np.float32)         # (B,)

            # Gradients
            g_w0 = g.mean()
            g_w = (x.T @ g) / B + L2 * w
            GS = g[:, None] * S                     # (B, K)
            term1 = x.T @ GS                        # (d, K)
            x2Tg = x2.T @ g                         # (d,)
            term2 = V * x2Tg[:, None]               # (d, K)
            g_V = (term1 - term2) / B + L2 * V

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            w0 -= LR * (m_w0 / (1 - beta1 ** t)) / (math.sqrt(v_w0 / (1 - beta2 ** t)) + 1e-8)

            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            w -= LR * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + 1e-8)

            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            V -= LR * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + 1e-8)

    # Build per-input-byte lookup tables: (d_in, 256) for linear, (d_in, 256, K) for factor & sqfactor
    T_lin = np.zeros((d_in, 256), np.float32)
    T_fac = np.zeros((d_in, 256, K), np.float32)
    T_sqf = np.zeros((d_in, 256, K), np.float32)

    for j in range(d_in):
        idxs = col_feat[j]
        E = enc[j]  # (256, width)
        for b in range(256):
            x_i = E[b]  # (width,)
            if len(idxs) == 1:  # ordinal
                val = x_i[0]
                T_lin[j, b] = w[idxs[0]] * val
                T_fac[j, b] = V[idxs[0]] * val
                T_sqf[j, b] = (V[idxs[0]] ** 2) * (val * val)
            else:  # categorical
                nz = np.nonzero(x_i)[0]
                if nz.size:
                    loc = nz[0]
                    gi = idxs[loc]
                    T_lin[j, b] = w[gi]
                    T_fac[j, b] = V[gi]
                    T_sqf[j, b] = V[gi] ** 2

    # Write weights.bin: magic, d_in, K, C; then T_lin, T_fac, T_sqf, w0
    C = card["classes"]
    out = struct.pack("<4i", 0x464D3130, d_in, K, C)
    out += T_lin.astype("<f4").tobytes()
    out += T_fac.astype("<f4").tobytes()
    out += T_sqf.astype("<f4").tobytes()
    out += struct.pack("<f", w0)
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record = d_in bytes (14 for adult). Output = 1 byte class (0/1).
 * Weights: magic 0x464D3130, d_in, K, C; float T_lin[d_in][256], T_fac[d_in][256][K],
 * T_sqf[d_in][256][K], w0. Score = w0 + Σ_j T_lin[j][x_j] + ½ Σ_f ( (Σ_j T_fac[j][x_j][f])² - Σ_j T_sqf[j][x_j][f] ). */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d_in = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d_in != 14 || K < 1 || K > 64) { fprintf(stderr, "bad header dims\n"); return 3; }

    long expected = 16L + 4L * ( (long)d_in * 256 * (1 + 2 * K) + 1 );
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *T_lin = (const float *)(hd + 4);
    const float *T_fac = T_lin + (long)d_in * 256;
    const float *T_sqf = T_fac + (long)d_in * 256 * K;
    const float w0 = T_sqf[(long)d_in * 256 * K];

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *S = calloc(K, sizeof(float));
    float *Q = calloc(K, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d_in) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d_in); return 3; }

        float lin = 0.0f;
        for (int f = 0; f < K; f++) { S[f] = 0.0f; Q[f] = 0.0f; }

        for (int j = 0; j < d_in; j++) {
            unsigned char b = xi[j];
            long base = (long)j * 256;
            lin += T_lin[base + b];
            long baseK = base * K;
            for (int f = 0; f < K; f++) {
                S[f] += T_fac[baseK + b * K + f];
                Q[f] += T_sqf[baseK + b * K + f];
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < K; f++) inter += S[f] * S[f] - Q[f];
        inter *= 0.5f;
        float score = w0 + lin + inter;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```