```hypothesis
Family: label-emb (embedding-based linear model). Categorical features map to learned 4-d embeddings instead of one-hot, ordinal features are z-scored. The dense 38-d representation (6 ordinal + 8*4 embeddings) feeds a linear classifier. This should match or exceed label-linear's 0.8524 accuracy by sharing statistical strength across categories, while cutting inference cost: 8 table lookups + one 38x2 matvec vs label-linear's ~1000-d sparse dot product. Failure: dev accuracy < 0.850 or Ir/item > 700.
```

```python
"""label-emb: embedding-based linear model for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Encodes ordinal columns as z-scored scalars, categorical columns as learned embeddings (dim E=4).
Exports: magic(0x454D424C), d, E, C, n_cat, n_ord, cat_indices[14], ord_indices[14],
         embed[sum_cat*256*E], ord_mean[n_ord], ord_std[n_ord], W[(n_ord+n_cat*E)*C], b[C] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

E = 4
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
    d = card["input"]["bytes"]

    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]
    ord_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    n_cat, n_ord = len(cat_idx), len(ord_idx)

    rng = np.random.default_rng(SEED)

    # Ordinal normalization stats
    ord_mean = X[:, ord_idx].mean(axis=0).astype(np.float32)
    ord_std = X[:, ord_idx].std(axis=0).astype(np.float32)
    ord_std[ord_std == 0] = 1.0

    # Embedding tables: one per categorical feature, each 256 x E
    embed = (rng.uniform(-1, 1, (n_cat, 256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)

    # Linear layer
    D = n_ord + n_cat * E
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [embed.reshape(-1), W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Build dense features
            z = np.empty((bs, D), np.float32)
            # Ordinal part
            if n_ord:
                z[:, :n_ord] = ((xb[:, ord_idx].astype(np.float32) - ord_mean) / ord_std)
            # Categorical embeddings
            off = n_ord
            for ci, cj in enumerate(cat_idx):
                z[:, off:off+E] = embed[ci, xb[:, cj]]
                off += E

            # Forward + loss grad
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_logits = (probs / bs).astype(np.float32)

            # Gradients
            grad_W = z.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_z = grad_logits @ W.T

            grad_embed = np.zeros_like(embed)
            if n_ord:
                grad_ord = grad_z[:, :n_ord] / ord_std  # dL/dx_ord
            off = n_ord
            for ci, cj in enumerate(cat_idx):
                # scatter-add into embedding gradients
                np.add.at(grad_embed[ci], xb[:, cj], grad_z[:, off:off+E])
                off += E

            grads = [grad_embed.reshape(-1), grad_W, grad_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Pack weights
    magic = 0x454D424C  # 'L','B','M','E' little-endian
    hdr = struct.pack("<6i", magic, d, E, C, n_cat, n_ord)
    cat_idx_arr = np.array(cat_idx, dtype=np.int32)
    ord_idx_arr = np.array(ord_idx, dtype=np.int32)
    out = (hdr +
           cat_idx_arr.tobytes() +
           ord_idx_arr.tobytes() +
           embed.astype("<f4").tobytes() +
           ord_mean.astype("<f4").tobytes() +
           ord_std.astype("<f4").tobytes() +
           W.astype("<f4").tobytes() +
           b.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (ordinal + categorical indices). Output: 1 byte class index.
 * Weights layout (little-endian):
 *   int32 magic(0x454D424C), d, E, C, n_cat, n_ord
 *   int32 cat_idx[n_cat], ord_idx[n_ord]
 *   float embed[n_cat*256*E], ord_mean[n_ord], ord_std[n_ord], W[(n_ord+n_cat*E)*C], b[C] */
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
    if (nw < 24 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3], n_cat = hd[4], n_ord = hd[5];
    if (C < 1 || C > 256 || n_cat + n_ord != d) { fprintf(stderr, "bad header\n"); return 3; }

    int32_t *cat_idx = hd + 6;
    int32_t *ord_idx = cat_idx + n_cat;
    float *embed = (float *)(ord_idx + n_ord);
    float *ord_mean = embed + (long)n_cat * 256 * E;
    float *ord_std = ord_mean + n_ord;
    float *W = ord_std + n_ord;
    float *b = W + (long)(n_ord + n_cat * E) * C;

    long expected = 24L + 4L * (n_cat + n_ord) + 4L * ((long)n_cat * 256 * E + n_ord + n_ord + (long)(n_ord + n_cat * E) * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * (n_ord + n_cat * E));
    float *s = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        int off = 0;
        for (int j = 0; j < n_ord; j++) {
            float v = (float)xi[ord_idx[j]];
            z[off++] = (v - ord_mean[j]) / ord_std[j];
        }
        for (int ci = 0; ci < n_cat; ci++) {
            const float *e = embed + ((long)ci * 256 + xi[cat_idx[ci]]) * E;
            memcpy(z + off, e, E * sizeof(float));
            off += E;
        }

        for (int c = 0; c < C; c++) s[c] = b[c];
        for (int j = 0; j < off; j++) {
            float zj = z[j];
            for (int c = 0; c < C; c++) s[c] += zj * W[(long)j * C + c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```