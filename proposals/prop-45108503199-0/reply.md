```hypothesis
Factorization Machines (FM) model pairwise feature interactions via low-rank latent vectors.
For this tabular task with 14 mixed-type features (ordinal quantile bins, categorical indices),
FMs are a natural fit: they capture cross-feature patterns that linear models miss while staying
extremely fast at inference (O(d*K) with d=14, K=8). With deterministic Adam training on the
full one-hot/z-scored encoding, label-fm should exceed label-linear (0.8524) and rival
label-mlp16 (0.8591) at lower CPU cost. If dev accuracy stays ≤0.8524, pairwise interactions
alone are insufficient for this dataset.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Encoding: ordinal -> z-scored continuous; categorical -> one-hot over training values.
FM: w0 + sum_j w_j x_j + 0.5 * ( (sum_j v_j x_j)^2 - sum_j (v_j x_j)^2 )
Exported as: magic 0x464D4C31, d, K, C; float32 w0, w[d][256], V[d][256][K] (C=2, one logit)."""
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
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    # Build per-column encoders: for ordinal -> z-scored continuous (1 dim), for categorical -> one-hot
    encoders = []
    offsets = []
    off = 0
    for j, t in enumerate(card["input"]["types"]):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            # continuous feature: single value per example, we'll treat as one dimension
            encoders.append(("cont", mu, sd))
            offsets.append(off)
            off += 1
        else:
            vals = np.unique(X[:, j])
            mapping = {v: i for i, v in enumerate(vals)}
            encoders.append(("cat", mapping, len(vals)))
            offsets.append(off)
            off += len(vals)
    D = off  # total expanded dimensions

    # Expanded design matrix Z (n x D)
    n = X.shape[0]
    Z = np.zeros((n, D), dtype=np.float32)
    for j in range(d):
        t = encoders[j]
        if t[0] == "cont":
            _, mu, sd = t
            Z[:, offsets[j]] = (X[:, j].astype(np.float32) - mu) / sd
        else:
            _, mapping, _ = t
            for idx, val in enumerate(X[:, j]):
                if val in mapping:
                    Z[idx, offsets[j] + mapping[val]] = 1.0

    # FM parameters
    w0 = np.float32(0.0)
    w = (rng.normal(0, 0.01, D)).astype(np.float32)
    V = (rng.normal(0, 0.01, (D, K))).astype(np.float32)

    params = [w0, w, V]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t_step = 0

    def forward(Zb):
        # Zb: (b, D)
        linear = w0 + Zb @ w                       # (b,)
        ZV = Zb @ V                                 # (b, K)
        inter = 0.5 * np.sum(ZV * ZV - (Zb * Zb) @ (V * V), axis=1)  # (b,)
        return linear + inter                       # logit

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Zb, yb = Z[idx], y[idx]
            logit = forward(Zb)
            p = sigmoid(logit)
            # gradient of BCE w.r.t logit
            g = (p - yb).astype(np.float32) / len(yb)   # (b,)

            # gradients
            gw0 = g.sum()
            gw = Zb.T @ g + L2 * w
            # d(inter)/dV = Zb.T @ (g[:,None] * (ZV - Zb[:,:,None]*V))  but careful:
            # inter = 0.5 * sum_k ( (sum_i z_ik v_ik)^2 - sum_i z_ik^2 v_ik^2 )
            # d inter / d v_jk = g * ( z_jk * (sum_i z_ik v_ik) - z_jk^2 v_jk )
            ZV = Zb @ V
            g_exp = g[:, None]  # (b,1)
            gV = Zb.T @ (g_exp * ZV) - (Zb * Zb).T @ (g_exp * V) + L2 * V

            grads = [np.array([gw0], np.float32), gw, gV]

            t_step += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] *= 0.9; m[i] += 0.1 * gr
                v[i] *= 0.999; v[i] += 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t_step)
                v_hat = v[i] / (1 - 0.999 ** t_step)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights in a flat layout matching infer.c
    # We need to write per original feature (0..d-1) per value (0..255) the linear weight and K latent weights.
    # For continuous (ordinal) features, we have only one expanded dimension but 256 possible input byte values.
    # We map each byte value to its z-scored contribution: linear weight * z + latent * z.
    # But the FM formula uses x_j as the expanded feature. For continuous, x_j is a single continuous value.
    # In the input record, the byte is the bin index. We need to convert it to the continuous value (z-score).
    # So we precompute for each ordinal feature j and each byte value v (0..255):
    #   z = (v - mu) / sd
    #   w_jv = w[off] * z
    #   V_jv = V[off] * z
    # For categorical, we have one-hot: if byte value v is in mapping, use w[off+mapping[v]] and V[off+mapping[v]],
    # else zero.
    w_flat = np.zeros((d, 256), dtype=np.float32)
    V_flat = np.zeros((d, 256, K), dtype=np.float32)
    for j in range(d):
        t = encoders[j]
        if t[0] == "cont":
            _, mu, sd = t
            off = offsets[j]
            for v in range(256):
                z = (v - mu) / sd
                w_flat[j, v] = w[off] * z
                V_flat[j, v] = V[off] * z
        else:
            _, mapping, _ = t
            for v in range(256):
                if v in mapping:
                    off = offsets[j] + mapping[v]
                    w_flat[j, v] = w[off]
                    V_flat[j, v] = V[off]

    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, K, 2)
    out += struct.pack("<f", w0)
    out += w_flat.astype("<f4").tobytes()
    out += V_flat.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index 0 or 1). */
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d < 1 || d > 256 || K < 1 || K > 256) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sumV = (float *)malloc(K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear = w0;
        for (int kk = 0; kk < K; kk++) sumV[kk] = 0.0f;
        float inter_acc = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            float wjv = w[(long)j * 256 + v];
            linear += wjv;
            const float *Vjv = V + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) sumV[kk] += Vjv[kk];
        }

        for (int kk = 0; kk < K; kk++) inter_acc += sumV[kk] * sumV[kk];
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *Vjv = V + ((long)j * 256 + v) * K;
            for (int kk = 0; kk < K; kk++) inter_acc -= Vjv[kk] * Vjv[kk];
        }
        inter_acc *= 0.5f;

        float logit = linear + inter_acc;
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    free(sumV);
    free(wb);
    free(x);
    return 0;
}
```