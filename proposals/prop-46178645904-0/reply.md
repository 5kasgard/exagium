```hypothesis
Family: label-fm (Factorization Machine with 8 factors). FM captures pairwise feature interactions efficiently, crucial for tabular data with mixed ordinal/categorical columns. It has more capacity than linear but far fewer parameters than MLP/GBDT, so it should beat label-linear (0.8524) and approach label-mlp16 (0.8591) at lower inference cost (table lookups + small vector ops). Failure if dev accuracy ≤ 0.8524 or inference cost > label-linear (719 Ir/item).
```
```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Encodes ordinal as normalized, categorical as one-hot.
Trains FM (linear + pairwise factorized) with Adam. Exports per-byte lookup tables.
Weights format: int32 magic 0x464D4C31, d, D, C; float32 bias; then for each column j (0..d-1):
  float32 linear[256], factor[256][D], factor_sq[256][D]."""
import json, math, struct, sys

import numpy as np

D = 8          # factor dimension
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 50


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
    rng = np.random.default_rng(SEED)

    # Build encodings per column (same as label-linear)
    enc = []          # list of (256, k_j) float64
    offsets = []      # start index in global feature vector
    total_k = 0
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None]          # (256, 1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
        offsets.append(total_k)
        total_k += E.shape[1]

    # Expand to global feature matrix Z (n, total_k)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape

    # FM parameters
    w0 = np.float32(0.0)
    w = (rng.uniform(-0.01, 0.01, k)).astype(np.float32)
    V = (rng.uniform(-0.01, 0.01, (k, D))).astype(np.float32)
    params = [w0, w, V]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # FM forward
            linear = z @ w + w0                     # (batch,)
            # interaction term: 0.5 * sum_f ( (sum_i V_{i,f} z_i)^2 - sum_i V_{i,f}^2 z_i^2 )
            zV = z @ V                              # (batch, D)
            z2 = z * z
            V2 = V * V
            zV2 = z2 @ V2                           # (batch, D)
            inter = 0.5 * np.sum(zV * zV - zV2, axis=1)
            logits = linear + inter                 # (batch,)
            p1 = sigmoid(logits)                    # P(y=1)
            # gradient of logistic loss: (p1 - y) for each sample
            grad_logits = (p1 - yy).astype(np.float32)          # (batch,)
            # gradients
            gw0 = grad_logits.mean()
            gw = (z.T @ grad_logits) / len(yy) + L2 * w
            # d(inter)/dV = z[:,None] * zV[None,:] - z2[:,None] * V[None,:]  (but batched)
            # For each factor f: sum over batch of grad_logits * (z_i * zV_f - z_i^2 * V_{i,f})
            # Vectorized: (z.T @ (grad_logits[:,None] * zV)) - (z2.T @ (grad_logits[:,None] * V))
            gV = (z.T @ (grad_logits[:, None] * zV)) / len(yy) - (z2.T @ (grad_logits[:, None] * V)) / len(yy) + L2 * V
            grads = [gw0, gw, gV]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Build per-byte lookup tables
    linear_tables = []   # list of (256,) float32
    factor_tables = []   # list of (256, D) float32
    factor_sq_tables = []# list of (256, D) float32
    for j, E in enumerate(enc):
        off = offsets[j]
        kj = E.shape[1]
        w_slice = w[off:off + kj]
        V_slice = V[off:off + kj]
        V2_slice = V_slice * V_slice
        linear_tables.append((E @ w_slice).astype(np.float32))
        factor_tables.append((E @ V_slice).astype(np.float32))
        factor_sq_tables.append((E @ V2_slice).astype(np.float32))

    # Write weights
    # magic, d, D, C
    out = struct.pack("<4i", 0x464D4C31, d, D, C)
    out += struct.pack("<f", w0)
    for j in range(d):
        out += linear_tables[j].tobytes()
        out += factor_tables[j].tobytes()
        out += factor_sq_tables[j].tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: one byte class index (0 or 1).
 * Weights format: int32 magic 0x464D4C31, d, D, C; float32 bias;
 *   for each column j: float32 linear[256], factor[256][D], factor_sq[256][D]. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C != 2 || d < 1 || D < 1) { fprintf(stderr, "bad header\n"); return 3; }
    float *p = (float *)(hd + 4);
    float bias = *p++;
    // expected size: 4*4 + 4 + d * 256 * (1 + 2*D) * 4
    long expected = 16 + 4 + (long)d * 256 * (1 + 2 * D) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *factor_sum = calloc(D, sizeof(float));
    float *factor_sq_sum = calloc(D, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float linear_sum = bias;
        memset(factor_sum, 0, D * sizeof(float));
        memset(factor_sq_sum, 0, D * sizeof(float));
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            linear_sum += p[v];
            p += 256;
            for (int f = 0; f < D; f++) factor_sum[f] += p[v * D + f];
            p += 256 * D;
            for (int f = 0; f < D; f++) factor_sq_sum[f] += p[v * D + f];
            p += 256 * D;
        }
        float inter = 0.0f;
        for (int f = 0; f < D; f++) {
            float fs = factor_sum[f];
            inter += 0.5f * (fs * fs - factor_sq_sum[f]);
        }
        float score = linear_sum + inter;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```