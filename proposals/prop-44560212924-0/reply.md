```hypothesis
Family: label-fm2 (Factorization Machine with 2 latent factors per class).
Why: Adult has strong pairwise interactions (e.g., education×occupation, marital-status×sex). A degree-2 FM captures all pairwise interactions with only O(k·d) parameters (k≈108 features, d=2), adding ~2× the weights of linear but modelling non-linearities. Inference stays fast: 14 columns × 2 classes × 2 factors = 112 multiply-adds per item.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or Ir/item > 1500, the interaction gain doesn't justify the cost.
```
```python
"""Factorization Machine (degree 2, 2 latent factors) for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Weights format: int32 magic(0x464D3231), d, D, C; then float32 bias[C], linear_table[d*256*C], fm_table[d*256*D*C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 1e-2
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 20
D = 2  # latent factors per class

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
    d = card["input"]["bytes"]
    C = card["classes"]
    types = card["input"]["types"]
    n = X.shape[0]

    # Build feature mapping: for each column j, value v (0..255) -> (feat_idx, feat_val)
    # feat_idx = -1 for unseen categorical (value 255 not in training)
    feat_idx = []   # list of length d, each is array[256] of int32
    feat_val = []   # list of length d, each is array[256] of float32
    k = 0
    for j, t in enumerate(types):
        col = X[:, j]
        if t == "ordinal":
            mu = col.mean()
            sd = col.std() or 1.0
            idx = np.full(256, k, dtype=np.int32)
            val = (np.arange(256, dtype=np.float32) - mu) / sd
            feat_idx.append(idx)
            feat_val.append(val)
            k += 1
        else:
            uniq = np.unique(col)
            # unseen (255) maps to -1, value 0
            idx = np.full(256, -1, dtype=np.int32)
            val = np.zeros(256, dtype=np.float32)
            for v in uniq:
                if v == 255:
                    continue
                idx[v] = k
                val[v] = 1.0
                k += 1
            feat_idx.append(idx)
            feat_val.append(val)

    rng = np.random.default_rng(SEED)
    # Parameters
    bias = np.zeros(C, dtype=np.float32)
    w_lin = (rng.normal(0, 0.01, (k, C)) * math.sqrt(2.0 / k)).astype(np.float32)
    V = (rng.normal(0, 0.01, (k, D, C)) * math.sqrt(2.0 / k)).astype(np.float32)

    params = [bias, w_lin, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    # Precompute per-column mapping for fast batch conversion
    # We'll convert a batch of X (b, d) to Z (b, k) on the fly
    def batch_to_Z(Xb):
        b = Xb.shape[0]
        Z = np.zeros((b, k), dtype=np.float32)
        for j in range(d):
            v = Xb[:, j]
            idx = feat_idx[j][v]
            vl = feat_val[j][v]
            mask = idx != -1
            Z[mask, idx[mask]] = vl[mask]
        return Z

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X[idx]
            yb = y[idx]
            Z = batch_to_Z(Xb)  # (b, k)

            # Forward
            linear_logits = Z @ w_lin + bias  # (b, C)
            # FM interaction
            inter = np.zeros((len(yb), C), dtype=np.float32)
            for f in range(D):
                v_f = V[:, f, :]  # (k, C)
                sum_v = Z @ v_f   # (b, C)
                sum_v2 = (Z**2) @ (v_f**2)
                inter += 0.5 * (sum_v**2 - sum_v2)
            logits = linear_logits + inter

            # Softmax cross-entropy gradient
            max_logit = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = probs / len(yb)  # (b, C)

            # Gradients
            grad_bias = grad_logits.sum(axis=0).astype(np.float32)
            grad_w_lin = (Z.T @ grad_logits).astype(np.float32) + L2 * w_lin
            grad_V = np.zeros_like(V)
            for f in range(D):
                v_f = V[:, f, :]
                sum_v = Z @ v_f
                # d(inter)/dV[i,f,c] = x_i * (sum_v_c - V[i,f,c] * x_i)
                # Vectorized over batch:
                # For each sample, we need outer product of Z[b] and (sum_v[b] - v_f * Z[b])
                # We can do: grad_V[:, f, :] = Z.T @ (grad_logits * (sum_v - Z @ v_f))? Not exactly.
                # Let's do loop over batch for clarity (batch size 64, k~108, D=2, C=2 -> fine)
                for b in range(len(yb)):
                    zb = Z[b]
                    g = grad_logits[b]
                    sv = sum_v[b]
                    # non-zero features
 nz = zb != 0
                    if not np.any(nz):
                        continue
                    zb_nz = zb[nz]
                    sv_nz = sv[nz]  # This is wrong: sv is per class, not per feature.
                    # Correct: sv is (C,). For each feature i, we need sv_c - V[i,f,c]*zb_i.
                    # We'll loop over non-zero features.
                    nz_idx = np.where(nz)[0]
                    for i in nz_idx:
                        zi = zb[i]
                        for c in range(C):
                            grad_V[i, f, c] += g[c] * zi * (sv[c] - v_f[i, c] * zi)
            grad_V += L2 * V

            grads = [grad_bias, grad_w_lin, grad_V]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)

    # Build per-byte lookup tables for inference
    linear_table = np.zeros((d, 256, C), dtype=np.float32)
    fm_table = np.zeros((d, 256, D, C), dtype=np.float32)
    for j in range(d):
        for v in range(256):
            idx = feat_idx[j][v]
            if idx == -1:
                continue
            vl = feat_val[j][v]
            linear_table[j, v, :] = w_lin[idx] * vl
            fm_table[j, v, :, :] = V[idx] * vl

    # Write weights.bin
    magic = 0x464D3231  # "FM21"
    header = struct.pack("<4i", magic, d, D, C)
    out = header
    out += bias.astype("<f4").tobytes()
    out += linear_table.astype("<f4").tobytes()
    out += fm_table.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine (degree 2, D latent factors) inference for label tasks.
 * ./infer weights.bin in.rec N
 * Weights format: int32 magic(0x464D3231), d, D, C; float32 bias[C], linear_table[d*256*C], fm_table[d*256*D*C] */
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
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C != 2 || D != 2 || d != 14) { fprintf(stderr, "unexpected dims\n"); return 3; }
    size_t bias_sz = C;
    size_t lin_sz = (size_t)d * 256 * C;
    size_t fm_sz = (size_t)d * 256 * D * C;
    size_t total_floats = bias_sz + lin_sz + fm_sz;
    if (nw != 16 + (long)total_floats * 4) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *lin_tab = bias + bias_sz;
    const float *fm_tab = lin_tab + lin_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score[2] = {bias[0], bias[1]};
        float sum_v[2][2] = {{0}};  // [class][factor]
        float sum_v2[2][2] = {{0}};

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *lin_ptr = lin_tab + (j * 256 + v) * C;
            score[0] += lin_ptr[0];
            score[1] += lin_ptr[1];
            const float *fm_ptr = fm_tab + ((j * 256 + v) * D + 0) * C;
            for (int f = 0; f < D; f++) {
                for (int c = 0; c < C; c++) {
                    float val = fm_ptr[f * C + c];
                    sum_v[c][f] += val;
                    sum_v2[c][f] += val * val;
                }
            }
        }

        for (int c = 0; c < C; c++) {
            for (int f = 0; f < D; f++) {
                score[c] += 0.5f * (sum_v[c][f] * sum_v[c][f] - sum_v2[c][f]);
            }
        }

        out_byte = (score[1] > score[0]) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```