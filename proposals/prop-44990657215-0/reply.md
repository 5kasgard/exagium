```hypothesis
Family: label-fm (factorization machine with rank-8 latent vectors). It models pairwise feature interactions via low-rank dot products, which are crucial for tabular data with mixed categorical/ordinal features. The model adds only ~1800 parameters over linear, and inference costs O(d*rank) ~ 112 multiply-adds per item, keeping CPU low. If dev accuracy does not exceed label-linear (0.8524) or Ir/item exceeds label-mlp16 (2325), the approach fails.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal as z-scored, categorical as one-hot over training values.
FM: w0 + sum(w_i x_i) + 0.5*((sum v_i x_i)^2 - sum (v_i^2 x_i^2)).
Rank=8, Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Exports per-byte lookup tables: int32 magic 0x464D4C31, d, rank, C; float32 bias;
float32 linear[d][256]; float32 latent[d][256][rank]."""
import json, math, struct, sys

import numpy as np

RANK = 8
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
    rng = np.random.default_rng(SEED)

    # Build encoding matrices per raw feature
    enc = []
    offsets = []
    off = 0
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None]  # (256, 1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
        offsets.append(off)
        off += E.shape[1]
    M = off

    # Build dense design matrix Z (n, M)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, _ = Z.shape

    # Initialize parameters
    bias = np.float32(0.0)
    W = np.zeros(M, dtype=np.float32)
    V = (rng.normal(0, 0.01, (M, RANK)) * math.sqrt(2.0 / M)).astype(np.float32)  # He init for ReLU? but linear
    params = [W, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            bsz = len(yy)

            # Forward
            linear_part = z @ W + bias
            ZV = z @ V
            Z_sq = z * z
            V_sq = V * V
            ZV_sq = Z_sq @ V_sq
            interaction = 0.5 * np.sum(ZV * ZV - ZV_sq, axis=1)
            scores = linear_part + interaction
            probs = 1.0 / (1.0 + np.exp(-scores))
            # Gradient of loss w.r.t scores
            dL_ds = (probs - yy) / bsz  # (bsz,)

            # Backward
            d_linear = dL_ds
            d_inter = dL_ds

            # d_inter w.r.t ZV and ZV_sq
            d_ZV = (ZV * d_inter[:, None]).astype(np.float32)
            d_ZV_sq = (-0.5 * d_inter[:, None]).astype(np.float32)

            # Gradients for V
            grad_V = z.T @ d_ZV + (Z_sq.T @ d_ZV_sq) * (2.0 * V)
            grad_V += L2 * V

            # Gradient for W
            grad_W = z.T @ d_linear + L2 * W

            # Gradient for bias
            grad_b = float(d_linear.sum())

            grads = [grad_W, grad_V]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

            bias -= LR * grad_b / (1 - 0.9 ** t) * 0.9  # bias uses same Adam but no L2; simplified: treat as param with m,v
            # Actually we should include bias in Adam. Let's do it properly by adding bias to params.
            # But for simplicity and to match label-linear (which didn't have bias in Adam), we'll just SGD bias.
            # However, label-linear had bias in b1 and included in Adam. We'll do the same: add bias to params.
            # We'll restructure: include bias in params list.

    # The above loop missed bias in Adam. Let's redo with bias in params.
    # We'll restart training with proper param list.

    # Re-initialize
    bias = np.float32(0.0)
    W = np.zeros(M, dtype=np.float32)
    V = (rng.normal(0, 0.01, (M, RANK)) * math.sqrt(2.0 / M)).astype(np.float32)
    params = [W, V, bias[None]]  # bias as array of size 1
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            bsz = len(yy)

            linear_part = z @ W + bias
            ZV = z @ V
            Z_sq = z * z
            V_sq = V * V
            ZV_sq = Z_sq @ V_sq
            interaction = 0.5 * np.sum(ZV * ZV - ZV_sq, axis=1)
            scores = linear_part + interaction
            probs = 1.0 / (1.0 + np.exp(-scores))
            dL_ds = (probs - yy) / bsz

            d_linear = dL_ds
            d_inter = dL_ds

            d_ZV = (ZV * d_inter[:, None]).astype(np.float32)
            d_ZV_sq = (-0.5 * d_inter[:, None]).astype(np.float32)

            grad_V = z.T @ d_ZV + (Z_sq.T @ d_ZV_sq) * (2.0 * V)
            grad_V += L2 * V

            grad_W = z.T @ d_linear + L2 * W

            grad_b = np.array([float(d_linear.sum())], dtype=np.float32)

            grads = [grad_W, grad_V, grad_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

            W, V, bias = params[0], params[1], params[2][0]

    # Build lookup tables for inference
    linear_table = np.zeros((card["input"]["bytes"], 256), dtype=np.float32)
    latent_table = np.zeros((card["input"]["bytes"], 256, RANK), dtype=np.float32)
    for j, E in enumerate(enc):
        start = offsets[j]
        end = start + E.shape[1]
        W_j = W[start:end]
        V_j = V[start:end]
        # For each raw value v, compute contribution
        linear_table[j] = E @ W_j
        latent_table[j] = E @ V_j  # (256, RANK)

    # Write weights.bin
    # Magic: 0x464D4C31 ("FML1")
    header = struct.pack("<4i", 0x464D4C31, card["input"]["bytes"], RANK, C)
    out = header
    out += np.array([bias], dtype="<f4").tobytes()
    out += linear_table.astype("<f4").tobytes()
    out += latent_table.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index 0 or 1).
 * Weights format: int32 magic 0x464D4C31, d, rank, C; float32 bias;
 * float32 linear[d][256]; float32 latent[d][256][rank]. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], rank = hd[2], C = hd[3];
    if (C != 2 || rank <= 0 || d <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }

    // Expected size: header(16) + bias(4) + linear(d*256*4) + latent(d*256*rank*4)
    long expected = 16 + 4 + (long)d * 256 * 4 + (long)d * 256 * rank * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + 1;
    const float *latent = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_vec = (float *)malloc(sizeof(float) * rank);
    if (!sum_vec) { fprintf(stderr, "malloc failed\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float linear_sum = *bias;
        for (int r = 0; r < rank; r++) sum_vec[r] = 0.0f;
        float sum_sq = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            linear_sum += linear[(long)j * 256 + v];
            const float *lv = latent + ((long)j * 256 + v) * rank;
            for (int r = 0; r < rank; r++) {
                float val = lv[r];
                sum_vec[r] += val;
                sum_sq += val * val;
            }
        }

        float dot = 0.0f;
        for (int r = 0; r < rank; r++) dot += sum_vec[r] * sum_vec[r];
        float interaction = 0.5f * (dot - sum_sq);
        float score = linear_sum + interaction;
        float prob = 1.0f / (1.0f + expf(-score));
        unsigned char pred = (prob > 0.5f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }

    free(sum_vec);
    return 0;
}
```