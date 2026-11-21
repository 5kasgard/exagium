```hypothesis
Family: label-fm (Factorization Machine with binary cross-entropy).
Why: Adult has strong pairwise interactions (e.g., education×occupation, marital×sex) that linear models miss. FM captures all pairwise interactions via low-rank embeddings (k=8) with only O(k·d) extra inference cost, keeping CPU low. The one-hot/z-score encoding matches label-linear, so the only added parameters are k embeddings per feature.
Failure signal: dev accuracy ≤ 0.8524 (label-linear) or instructions/item > 1000.
```

```python
"""Factorization Machine for tabular label tasks (binary).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encoding: ordinal z-scored, categorical one-hot over training values (255 = unseen).
FM: logit = w0 + Σ w_i x_i + ½ Σ_f [(Σ v_{i,f} x_i)² - Σ v_{i,f}² x_i²].
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Exports: magic 0x464D4C31, d, k, C=1, bias, linear_table[d][256], emb_table[d][256][k] (float32 LE)."""
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
    assert C == 2, "FM binary only"
    rng = np.random.default_rng(SEED)

    # Build encoding tables exactly like label-linear
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, D = Z.shape

    # FM parameters
    w = np.zeros(D, np.float32)
    V = (rng.normal(0, 1.0 / math.sqrt(K), (D, K))).astype(np.float32)
    bias = np.float32(0.0)
    params = [w, V, bias[None]]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # forward
            linear = z @ w
            a = z @ V                      # (B, K)
            b = z @ (V * V)                # (B, K)
            pairwise = 0.5 * np.sum(a * a - b, axis=1)
            logit = linear + pairwise + bias
            p = 1.0 / (1.0 + np.exp(-logit))
            g = p - yy                     # (B,)

            # gradients
            grad_w = z.T @ g + L2 * w
            g_exp = g[:, None]
            term1 = z.T @ (g_exp * a)
            term2 = V * (z.T @ g)[:, None]
            grad_V = term1 - term2 + L2 * V
            grad_bias = np.array([g.sum()], np.float32)
            grads = [grad_w, grad_V, grad_bias]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Build lookup tables for inference
    linear_table = np.zeros((X.shape[1], 256), np.float32)
    emb_table = np.zeros((X.shape[1], 256, K), np.float32)
    off = 0
    for j, E in enumerate(enc):
        Dj = E.shape[1]
        w_j = w[off:off + Dj]
        V_j = V[off:off + Dj]
        linear_table[j] = E @ w_j
        emb_table[j] = E @ V_j
        off += Dj

    # Write weights
    magic, d, k, C_out = 0x464D4C31, X.shape[1], K, 1
    out = struct.pack("<4i", magic, d, k, C_out)
    out += np.array([bias], dtype="<f4").tobytes()
    out += linear_table.astype("<f4").tobytes()
    out += emb_table.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (d=14). Output: one byte class (0/1). */
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = (int)hd[1], K = (int)hd[2], C = (int)hd[3];
    if (d != 14 || K < 1 || K > 64 || C != 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *lin_tab = bias + 1;
    const float *emb_tab = lin_tab + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float emb_sum[64], emb_sq_sum[64];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float linear_sum = *bias;
        for (int f = 0; f < K; f++) { emb_sum[f] = 0.0f; emb_sq_sum[f] = 0.0f; }
        for (int j = 0; j < d; j++) {
            unsigned char b = xi[j];
            linear_sum += lin_tab[(long)j * 256 + b];
            const float *emb = emb_tab + ((long)j * 256 + b) * K;
            for (int f = 0; f < K; f++) {
                float e = emb[f];
                emb_sum[f] += e;
                emb_sq_sum[f] += e * e;
            }
        }
        float pairwise = 0.0f;
        for (int f = 0; f < K; f++) pairwise += emb_sum[f] * emb_sum[f] - emb_sq_sum[f];
        pairwise *= 0.5f;
        float logit = linear_sum + pairwise;
        unsigned char cls = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```