```hypothesis
Family: label-emb-linear (embedding-based linear model). Categorical features get learned embeddings (size 8) instead of one-hot, reducing dimensionality and capturing category similarity; ordinal features are z-scored. A single linear layer on top yields logits. This should match or beat label-linear (0.8524) on dev accuracy with similar inference cost (same lookup-table structure, just smaller tables). If dev accuracy does not exceed label-linear, the hypothesis fails.
```

```python
"""label-emb-linear: embedding-based linear model for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports per-byte lookup tables in the same format as label-linear (H=0) but with a different magic number."""
import json, math, struct, sys

import numpy as np

E = 8                 # embedding size for categorical features
SEED, LR, L2, BATCH, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 100
MAGIC = 0x4C424D45    # 'EMBL' in little-endian

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
    types = card["input"]["types"]
    d = len(types)

    # Identify ordinal / categorical features
    ord_idx = [i for i, t in enumerate(types) if t == "ordinal"]
    cat_idx = [i for i, t in enumerate(types) if t == "categorical"]
    n_ord, n_cat = len(ord_idx), len(cat_idx)

    rng = np.random.default_rng(SEED)

    # Ordinal: z-score on training data
    X_ord = X_bytes[:, ord_idx].astype(np.float32)
    ord_mean = X_ord.mean(axis=0)
    ord_std = X_ord.std(axis=0)
    ord_std[ord_std == 0] = 1.0
    X_ord = (X_ord - ord_mean) / ord_std

    # Categorical: map each feature's values to 0..num_cat, with 255 -> unseen index
    cat_maps = []
    cat_sizes = []
    X_cat_idx = np.zeros((X_bytes.shape[0], n_cat), dtype=np.int32)
    for ci, j in enumerate(cat_idx):
        vals = X_bytes[:, j]
        uniq = np.unique(vals[vals != 255])          # training categories (exclude unseen marker)
        mapping = {v: i for i, v in enumerate(uniq)}
        unseen_idx = len(uniq)
        cat_maps.append(mapping)
        cat_sizes.append(unseen_idx + 1)
        mapped = np.array([mapping.get(v, unseen_idx) for v in vals], dtype=np.int32)
        X_cat_idx[:, ci] = mapped

    # Parameters
    # Embeddings: list of (cat_size, E) for each categorical feature
    embeddings = [rng.normal(0, 0.01, (sz, E)).astype(np.float32) for sz in cat_sizes]
    # Linear weights: (n_ord + n_cat * E, C)
    in_dim = n_ord + n_cat * E
    W = rng.normal(0, math.sqrt(2.0 / in_dim), (in_dim, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    m_emb = [np.zeros_like(e) for e in embeddings]
    v_emb = [np.zeros_like(e) for e in embeddings]
    t = 0

    n = X_bytes.shape[0]
    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            bs = len(idx)
            # Build batch input
            x_ord = X_ord[idx]                              # (bs, n_ord)
            x_cat_emb = np.zeros((bs, n_cat * E), dtype=np.float32)
            for ci in range(n_cat):
                emb = embeddings[ci]
                indices = X_cat_idx[idx, ci]
                x_cat_emb[:, ci * E:(ci + 1) * E] = emb[indices]
            x = np.concatenate([x_ord, x_cat_emb], axis=1)  # (bs, in_dim)

            # Forward
            logits = x @ W + b                              # (bs, C)
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            yy = y[idx]
            probs[np.arange(bs), yy] -= 1.0
            grad_logits = probs / bs                        # (bs, C)

            # Gradients
            grad_W = x.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_x = grad_logits @ W.T                      # (bs, in_dim)

            # Split grad_x
            grad_ord = grad_x[:, :n_ord]
            grad_cat = grad_x[:, n_ord:].reshape(bs, n_cat, E)

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            # Linear weights
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * grad_W * grad_W
            m_hat_W = m_W / (1 - beta1 ** t)
            v_hat_W = v_W / (1 - beta2 ** t)
            W -= LR * m_hat_W / (np.sqrt(v_hat_W) + eps)
            # Linear bias
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b * grad_b
            m_hat_b = m_b / (1 - beta1 ** t)
            v_hat_b = v_b / (1 - beta2 ** t)
            b -= LR * m_hat_b / (np.sqrt(v_hat_b) + eps)
            # Embeddings
            for ci in range(n_cat):
                # Accumulate gradients per embedding index
                g_emb = np.zeros_like(embeddings[ci])
                np.add.at(g_emb, X_cat_idx[idx, ci], grad_cat[:, ci, :])
                g_emb += L2 * embeddings[ci]
                m_emb[ci] = beta1 * m_emb[ci] + (1 - beta1) * g_emb
                v_emb[ci] = beta2 * v_emb[ci] + (1 - beta2) * g_emb * g_emb
                m_hat = m_emb[ci] / (1 - beta1 ** t)
                v_hat = v_emb[ci] / (1 - beta2 ** t)
                embeddings[ci] -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Build per-feature per-byte lookup tables (d x 256 x C)
    T = np.zeros((d, 256, C), dtype=np.float32)
    # Ordinal features
    for oi, j in enumerate(ord_idx):
        w_slice = W[oi:oi+1, :]  # (1, C)
        for v in range(256):
            if v < 255:
                norm = (v - ord_mean[oi]) / ord_std[oi]
            else:
                norm = 0.0
            T[j, v, :] = norm * w_slice
    # Categorical features
    for ci, j in enumerate(cat_idx):
        emb = embeddings[ci]          # (cat_size, E)
        w_slice = W[n_ord + ci * E : n_ord + (ci + 1) * E, :]  # (E, C)
        for v in range(256):
            if v in cat_maps[ci]:
                e = emb[cat_maps[ci][v]]
            else:
                e = emb[-1]           # unseen embedding
            T[j, v, :] = e @ w_slice

    # Write weights file
    out = struct.pack("<4i", MAGIC, d, 0, C)
    out += T.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-linear inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index). */
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
    if (nw < 16 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (H != 0 || C < 1 || C > 256) { fprintf(stderr, "bad weights config\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *b = T + (long)d * 256 * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) sc[c] = b[c];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) sc[c] += t[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```