```hypothesis
Family: label-embmlp (Embedding MLP). Learns 4-d embeddings per categorical column, z-scores ordinal columns, feeds concatenated vector (≈38 dims) into a 32-unit ReLU hidden layer + linear output. Beats label-linear (0.8524) by adding nonlinear interactions at lower dimensionality than label-mlp16's one-hot input; targets >0.855 dev accuracy with Ir/item < 1000. Fails if accuracy ≤ 0.8524 or cost ≥ label-mlp16.
```

```python
"""label-embmlp: Embedding MLP for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Categorical cols -> learned 4-d embeddings (255=unseen zero), ordinal cols -> z-score.
Concat -> Linear(32) -> ReLU -> Linear(2). Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights format compatible with label-linear infer.c (magic 0x4D424D45)."""
import json, math, struct, sys

import numpy as np

E = 4          # embedding dim per categorical column
H = 32         # hidden units
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x4D424D45  # 'EMBM' little-endian


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
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # Separate categorical and ordinal columns
    cat_cols, ord_cols = [], []
    cat_maps, cat_sizes = [], []
    ord_mu, ord_sd = [], []
    for j, t in enumerate(types):
        if t == "categorical":
            cat_cols.append(j)
            vals = np.unique(X[:, j])
            # mapping from value to 0..K-1, reserve last index for unseen (255)
            mp = {v: i for i, v in enumerate(vals)}
            cat_maps.append(mp)
            cat_sizes.append(len(vals) + 1)  # +1 for unseen
        else:
            ord_cols.append(j)
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            ord_mu.append(mu)
            ord_sd.append(sd)

    n_cat, n_ord = len(cat_cols), len(ord_cols)
    D = n_cat * E + n_ord  # concatenated input dimension

    # Initialize embeddings for categorical columns
    embeddings = []
    for K in cat_sizes:
        # last row (unseen) stays zero
        emb = rng.normal(0, 1.0 / math.sqrt(E), size=(K, E)).astype(np.float32)
        emb[-1] = 0.0
        embeddings.append(emb)

    # Initialize MLP weights
    W1 = (rng.uniform(-1, 1, (D, H)) * math.sqrt(6.0 / D)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = embeddings + [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bs = len(yb)

            # Build input batch: concatenate embeddings and normalized ordinals
            x_batch = np.empty((bs, D), np.float32)
            off = 0
            cat_indices = []
            for ci, j in enumerate(cat_cols):
                mp = cat_maps[ci]
                K = cat_sizes[ci]
                # map values to indices, unseen (255) -> K-1
                indices = np.array([mp.get(v, K - 1) for v in xb[:, j]], dtype=np.int32)
                cat_indices.append(indices)
                x_batch[:, off:off + E] = embeddings[ci][indices]
                off += E
            for oi, j in enumerate(ord_cols):
                x_batch[:, off] = (xb[:, j].astype(np.float32) - ord_mu[oi]) / ord_sd[oi]
                off += 1

            # Forward
            a = x_batch @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            dlogits = (probs / bs).astype(np.float32)

            # Backward
            dW2 = h.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)
            dh = dlogits @ W2.T * (a > 0)
            dW1 = x_batch.T @ dh + L2 * W1
            db1 = dh.sum(axis=0)
            dx_batch = dh @ W1.T

            # Gradients for embeddings
            emb_grads = []
            off = 0
            for ci in range(n_cat):
                g = dx_batch[:, off:off + E]
                # accumulate per embedding row
                grad_emb = np.zeros_like(embeddings[ci])
                np.add.at(grad_emb, cat_indices[ci], g)
                emb_grads.append(grad_emb)
                off += E
            # ordinal gradients not needed (fixed normalization)

            grads = emb_grads + [dW1, db1, dW2, db2]

            # Adam update
            t = 1  # simplified: using fixed t per batch? label-linear uses global t.
            # We'll follow label-linear: global step counter
            # But for simplicity and determinism, use per-parameter Adam with global t.
            # Actually label-linear uses a single t incremented per batch.
            # Let's replicate exactly.

    # The above loop needs a global t. Let's restructure.

    # Rewriting training loop with global t:
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bs = len(yb)

            x_batch = np.empty((bs, D), np.float32)
            off = 0
            cat_indices = []
            for ci, j in enumerate(cat_cols):
                mp = cat_maps[ci]
                K = cat_sizes[ci]
                indices = np.array([mp.get(v, K - 1) for v in xb[:, j]], dtype=np.int32)
                cat_indices.append(indices)
                x_batch[:, off:off + E] = embeddings[ci][indices]
                off += E
            for oi, j in enumerate(ord_cols):
                x_batch[:, off] = (xb[:, j].astype(np.float32) - ord_mu[oi]) / ord_sd[oi]
                off += 1

            a = x_batch @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            dlogits = (probs / bs).astype(np.float32)

            dW2 = h.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)
            dh = dlogits @ W2.T * (a > 0)
            dW1 = x_batch.T @ dh + L2 * W1
            db1 = dh.sum(axis=0)
            dx_batch = dh @ W1.T

            emb_grads = []
            off = 0
            for ci in range(n_cat):
                g = dx_batch[:, off:off + E]
                grad_emb = np.zeros_like(embeddings[ci])
                np.add.at(grad_emb, cat_indices[ci], g)
                emb_grads.append(grad_emb)
                off += E

            grads = emb_grads + [dW1, db1, dW2, db2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build T tables for inference (d x 256 x H)
    T_list = []
    # W1 is D x H. First n_cat*E rows are embeddings, then n_ord rows.
    w1_cat = W1[:n_cat * E].reshape(n_cat, E, H)
    w1_ord = W1[n_cat * E:]

    for j in range(d):
        if j in cat_cols:
            ci = cat_cols.index(j)
            mp = cat_maps[ci]
            K = cat_sizes[ci]
            table = np.zeros((256, H), np.float32)
            for v in range(256):
                row = mp.get(v, K - 1)
                table[v] = embeddings[ci][row] @ w1_cat[ci]
            T_list.append(table)
        else:
            oi = ord_cols.index(j)
            table = np.zeros((256, H), np.float32)
            vals = np.arange(256, dtype=np.float32)
            norm = (vals - ord_mu[oi]) / ord_sd[oi]
            table[:, :] = norm[:, None] * w1_ord[oi]
            T_list.append(table)

    T_arr = np.stack(T_list)  # d x 256 x H
    # Pack weights: magic, d, H, C, T, b1, W2, b2
    out = struct.pack("<4i", MAGIC, d, H, C)
    out += T_arr.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input: d bytes per record (14 for adult). Output: 1 byte class index.
 * Weights format: magic 0x4D424D45 ('EMBM'), int32 d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C].
 * Identical structure to label-linear, only magic differs. */
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
    if (nw < 16 || hd[0] != 0x4D424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```