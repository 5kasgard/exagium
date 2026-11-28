```hypothesis
Family: label-emb (embedding-based linear model). Categorical features get learned embeddings (dim E=4), ordinal features are z-scored, then a single linear layer predicts the two classes. This reduces the one-hot dimensionality (10 cats × 256 → 10×4=40) while learning feature similarities, so it should beat label-linear (0.8524) with similar or lower inference cost (≈1k params vs 10k). If dev accuracy ≤0.8524 or instructions/item ≥ label-linear, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# hyper-parameters (chosen on the last 20% of train.in)
E = 4                # embedding dimension per categorical feature
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

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

    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]
    ord_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    n_cat, n_ord = len(cat_idx), len(ord_idx)

    rng = np.random.default_rng(SEED)

    # ordinal standardization
    ord_mean = np.zeros(n_ord, np.float64)
    ord_std  = np.ones(n_ord, np.float64)
    for k, j in enumerate(ord_idx):
        v = X[:, j].astype(np.float64)
        ord_mean[k] = v.mean()
        ord_std[k] = v.std() or 1.0

    # embeddings: (n_cat, 256, E)
    emb = (rng.uniform(-1, 1, (n_cat, 256, E)) * math.sqrt(6.0 / (256 * E))).astype(np.float32)

    # linear layer: input dim = n_cat*E + n_ord
    in_dim = n_cat * E + n_ord
    W = np.zeros((in_dim, C), np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # precompute ordinal part of design matrix (constant)
    Z_ord = np.empty((X.shape[0], n_ord), np.float32)
    for k, j in enumerate(ord_idx):
        Z_ord[:, k] = (X[:, j].astype(np.float32) - ord_mean[k]) / ord_std[k]

    t = 0
    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            batch_emb = emb[:, X[idx][:, cat_idx], :]          # (n_cat, batch, E)
            batch_emb = batch_emb.transpose(1, 0, 2).reshape(len(idx), -1)  # (batch, n_cat*E)
            z = np.concatenate([batch_emb, Z_ord[idx]], axis=1)            # (batch, in_dim)
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            probs[np.arange(len(idx)), y[idx]] -= 1.0
            grad_out = (probs / len(idx)).astype(np.float32)

            # gradients
            gW = z.T @ grad_out + L2 * W
            gb = grad_out.sum(0)
            gz = grad_out @ W.T                                          # (batch, in_dim)
            g_emb = gz[:, :n_cat*E].reshape(len(idx), n_cat, E)          # (batch, n_cat, E)
            g_emb = g_emb.transpose(1, 0, 2)                             # (n_cat, batch, E)
            gemb = np.zeros_like(emb)
            for k in range(n_cat):
                np.add.at(gemb[k], X[idx, cat_idx[k]], g_emb[k])

            grads = [gemb, gW, gb]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # pack weights.bin
    # magic 0x454D4231 ("EMB1"), d, E, C, n_cat, n_ord
    # cat_flags: d bytes (1=categorical, 0=ordinal)
    # ord_means[n_ord], ord_stds[n_ord] (float32)
    # embeddings: n_cat * 256 * E floats
    # W: (in_dim, C) floats
    # b: C floats
    cat_flags = bytearray(d)
    for j in cat_idx:
        cat_flags[j] = 1

    out = bytearray()
    out += struct.pack("<6i", 0x454D4231, d, E, C, n_cat, n_ord)
    out += cat_flags
    out += ord_mean.astype("<f4").tobytes()
    out += ord_std.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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
    if (nw < 24 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3], n_cat = hd[4], n_ord = hd[5];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }

    unsigned char *cat_flags = wb + 24;
    float *ord_mean = (float *)(cat_flags + d);
    float *ord_std  = ord_mean + n_ord;
    float *emb = ord_std + n_ord;
    int in_dim = n_cat * E + n_ord;
    float *W = emb + (long)n_cat * 256 * E;
    float *b = W + (long)in_dim * C;

    long expected = 24 + d + 4L * (n_ord * 2 + (long)n_cat * 256 * E + (long)in_dim * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * in_dim);
    float *s = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        int zidx = 0;
        for (int j = 0; j < d; j++) {
            if (cat_flags[j]) {
                const float *e = emb + ((long)zidx / E * 256 + xi[j]) * E;
                for (int u = 0; u < E; u++) z[zidx++] = e[u];
            } else {
                int o = 0;
                for (int jj = 0; jj < j; jj++) if (!cat_flags[jj]) o++;
                z[zidx++] = (xi[j] - ord_mean[o]) / ord_std[o];
            }
        }
        for (int c = 0; c < C; c++) s[c] = b[c];
        for (int j = 0; j < in_dim; j++) {
            float zv = z[j];
            for (int c = 0; c < C; c++) s[c] += zv * W[(long)j * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```