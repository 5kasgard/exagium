```hypothesis
We propose label-factmlp: a one-hidden-layer MLP with factorized embeddings for categorical features. Each categorical feature gets a rank-4 embedding (E=4) projected to H=16 hidden units, reducing parameters vs. label-mlp16's full 16-dim embeddings and acting as regularization. Ordinal features are z-scored and linearly projected. The model exports the same per-byte lookup tables as label-linear, so inference cost matches label-mlp16 (~2300 Ir/item). If dev accuracy ≤ 0.8591 (label-mlp16), the factorization does not help.
```

```python
"""label-factmlp: factorized embedding MLP for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports weights in label-linear's table format (magic 0x54424C31, d, H, C)."""
import json, math, struct, sys

import numpy as np

E = 4          # embedding rank for categorical features
H = 16         # hidden layer size
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
    d = card["input"]["bytes"]
    types = card["input"]["types"]

    rng = np.random.default_rng(SEED)

    # --- Precompute feature statistics / mappings ---
    ordinal_idx = []
    cat_idx = []
    ord_mu = []
    ord_sd = []
    cat_maps = []
    cat_unseen = []
    cat_sizes = []

    for j, t in enumerate(types):
        col = X[:, j]
        if t == "ordinal":
            ordinal_idx.append(j)
            mu = col.mean()
            sd = col.std() or 1.0
            ord_mu.append(float(mu))
            ord_sd.append(float(sd))
        else:
            cat_idx.append(j)
            uniq = np.unique(col)
            # map observed values to 0..V-2, reserve V-1 for unseen (255)
            mapping = {int(v): i for i, v in enumerate(uniq)}
            unseen_idx = len(uniq)
            cat_maps.append(mapping)
            cat_unseen.append(unseen_idx)
            cat_sizes.append(unseen_idx + 1)

    n_ord = len(ordinal_idx)
    n_cat = len(cat_idx)

    # --- Initialize parameters ---
    # Ordinal: weight vector per feature (H,)
    ord_W = [rng.uniform(-1, 1, H).astype(np.float32) * math.sqrt(2.0 / H) for _ in range(n_ord)]
    # Categorical: Emb (V, E), Proj (E, H)
    cat_Emb = []
    cat_Proj = []
    for V in cat_sizes:
        emb = rng.uniform(-1, 1, (V, E)).astype(np.float32) * math.sqrt(6.0 / (V + E))
        proj = rng.uniform(-1, 1, (E, H)).astype(np.float32) * math.sqrt(6.0 / (E + H))
        cat_Emb.append(emb)
        cat_Proj.append(proj)
    # Hidden bias
    b1 = np.zeros(H, np.float32)
    # Output layer
    W2 = rng.uniform(-1, 1, (H, C)).astype(np.float32) * math.sqrt(6.0 / (H + C))
    b2 = np.zeros(C, np.float32)

    params = ord_W + cat_Emb + cat_Proj + [b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Precompute normalized ordinal values for all 256 bins (for export later)
    # But during training we compute on the fly per batch.

    # Adam step counter
    t = 0
    n = X.shape[0]
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bs = len(yb)

            # Forward
            h = np.tile(b1, (bs, 1))  # (bs, H)

            # Ordinal contributions
            for k, j in enumerate(ordinal_idx):
                vals = xb[:, j].astype(np.float32)
                norm = (vals - ord_mu[k]) / ord_sd[k]
                h += norm[:, None] * ord_W[k]

            # Categorical contributions
            for k, j in enumerate(cat_idx):
                vals = xb[:, j]
                # map to indices
                indices = np.array([cat_maps[k].get(int(v), cat_unseen[k]) for v in vals], dtype=np.int32)
                # Emb[indices] @ Proj -> (bs, H)
                h += cat_Emb[k][indices] @ cat_Proj[k]

            a = np.maximum(h, 0, dtype=np.float32)
            logits = a @ W2 + b2

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            g = probs / bs  # (bs, C)

            # Backward
            gh = (g @ W2.T) * (h > 0)  # (bs, H)

            # Gradients for output layer
            grad_W2 = a.T @ g + L2 * W2
            grad_b2 = g.sum(axis=0)

            # Grad for b1
            grad_b1 = gh.sum(axis=0)

            # Grads for ordinal weights
            grad_ord = []
            for k, j in enumerate(ordinal_idx):
                vals = xb[:, j].astype(np.float32)
                norm = (vals - ord_mu[k]) / ord_sd[k]
                gw = (norm[:, None] * gh).sum(axis=0) + L2 * ord_W[k]
                grad_ord.append(gw)

            # Grads for categorical
            grad_Emb = []
            grad_Proj = []
            for k, j in enumerate(cat_idx):
                vals = xb[:, j]
                indices = np.array([cat_maps[k].get(int(v), cat_unseen[k]) for v in vals], dtype=np.int32)
                Emb_k = cat_Emb[k]
                Proj_k = cat_Proj[k]
                # grad_Proj = Emb_k[indices].T @ gh
                gp = Emb_k[indices].T @ gh + L2 * Proj_k
                # grad_Emb: scatter-add gh @ Proj_k.T
                ge = np.zeros_like(Emb_k)
                np.add.at(ge, indices, gh @ Proj_k.T)
                ge += L2 * Emb_k
                grad_Proj.append(gp)
                grad_Emb.append(ge)

            # Assemble grads in same order as params
            grads = grad_ord + grad_Emb + grad_Proj + [grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # --- Export weights in label-linear table format ---
    # Build T[d][256][H]
    T = np.zeros((d, 256, H), dtype=np.float32)
    ord_ptr = 0
    cat_ptr = 0
    for j, tp in enumerate(types):
        if tp == "ordinal":
            mu, sd = ord_mu[ord_ptr], ord_sd[ord_ptr]
            w = ord_W[ord_ptr]
            for v in range(256):
                if v < 255:
                    T[j, v] = (v - mu) / sd * w
                else:
                    T[j, v] = 0.0
            ord_ptr += 1
        else:
            mapping = cat_maps[cat_ptr]
            unseen = cat_unseen[cat_ptr]
            effective = cat_Emb[cat_ptr] @ cat_Proj[cat_ptr]  # (V, H)
            for v in range(256):
                if v == 255:
                    idx = unseen
                else:
                    idx = mapping.get(v, unseen)
                T[j, v] = effective[idx]
            cat_ptr += 1

    # Pack
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
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