```hypothesis
Family: label-embedding (linear on learned categorical embeddings + normalized ordinals).
Why: The Adult dataset has 8 categorical features with up to 256 levels each. One-hot encoding (label-linear) creates a high-dimensional sparse input (~3000 dims) that cannot capture similarities between categories. Learned embeddings (dim=4) project each category into a dense space where related categories can be close, while ordinal features are z-scored. A linear classifier on top (38 dims total) is fast at inference and has far fewer parameters than one-hot linear, yet can model interactions via the embedding geometry. This often beats one-hot linear on tabular data with many categories.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or inference cost exceeds label-linear significantly, the embedding hypothesis does not hold for this task.
```

```python
"""label-embedding: linear classifier on learned categorical embeddings + normalized ordinals.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only.
Weights format (little-endian):
  magic: 0x454D4244 ('EMBD')
  d: int32 (14)
  n_cat: int32 (8)
  emb_dim: int32 (4)
  C: int32 (2)
  ordinal_mean[d_ord]: float32
  ordinal_std[d_ord]: float32
  embeddings[n_cat][256][emb_dim]: float32
  W[ (d_ord + n_cat*emb_dim) ][C]: float32
  b[C]: float32
"""
import json, math, struct, sys

import numpy as np

EMB_DIM = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x454D4244  # 'EMBD'

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

    # Identify ordinal and categorical columns
    ord_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]
    n_ord, n_cat = len(ord_idx), len(cat_idx)

    rng = np.random.default_rng(SEED)

    # Ordinal normalization stats
    ord_mean = X[:, ord_idx].mean(axis=0).astype(np.float32)
    ord_std = X[:, ord_idx].std(axis=0).astype(np.float32)
    ord_std[ord_std == 0] = 1.0

    # Embeddings: [n_cat, 256, EMB_DIM]
    emb = rng.normal(0, 0.01, size=(n_cat, 256, EMB_DIM)).astype(np.float32)

    # Linear layer on concatenated [ordinal_norm, embeddings...]
    in_dim = n_ord + n_cat * EMB_DIM
    W = rng.normal(0, math.sqrt(2.0 / in_dim), size=(in_dim, C)).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # [B, d]
            yb = y[idx]          # [B]

            # Forward
            # Ordinal part
            ord_part = ((xb[:, ord_idx].astype(np.float32) - ord_mean) / ord_std)  # [B, n_ord]
            # Embedding part
            emb_part = np.zeros((len(idx), n_cat * EMB_DIM), np.float32)
            for k, col in enumerate(cat_idx):
                emb_part[:, k*EMB_DIM:(k+1)*EMB_DIM] = emb[k, xb[:, col]]
            z = np.concatenate([ord_part, emb_part], axis=1)  # [B, in_dim]

            logits = z @ W + b  # [B, C]
            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = probs / len(yb)  # [B, C]

            # Gradients
            grad_W = z.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_z = grad_logits @ W.T  # [B, in_dim]

            # Split grad_z to ordinal and embedding parts
            grad_ord = grad_z[:, :n_ord]
            grad_emb_flat = grad_z[:, n_ord:]  # [B, n_cat*EMB_DIM]

            grad_emb = np.zeros_like(emb)
            for k in range(n_cat):
                g = grad_emb_flat[:, k*EMB_DIM:(k+1)*EMB_DIM]  # [B, EMB_DIM]
                cats = xb[:, cat_idx[k]]
                # Accumulate gradients per category
                for i in range(256):
                    mask = (cats == i)
                    if np.any(mask):
                        grad_emb[k, i] = g[mask].sum(axis=0)

            # L2 on embeddings
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W, grad_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<5i", MAGIC, d, n_cat, EMB_DIM, C)
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
/* label-embedding inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (d=14). Output: 1 byte class index (0 or 1).
 * Weights format (little-endian):
 *   magic(0x454D4244), d, n_cat, emb_dim, C
 *   ordinal_mean[n_ord], ordinal_std[n_ord]
 *   embeddings[n_cat][256][emb_dim]
 *   W[in_dim][C], b[C]
 *   where in_dim = n_ord + n_cat*emb_dim
 */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], n_cat = hd[2], emb_dim = hd[3], C = hd[4];
    if (d != 14 || C != 2 || n_cat != 8 || emb_dim != 4) { fprintf(stderr, "unexpected dims\n"); return 3; }

    // Fixed indices from card
    int ord_idx[6] = {0,2,4,10,11,12};
    int cat_idx[8] = {1,3,5,6,7,8,9,13};
    int n_ord = 6;
    int in_dim = n_ord + n_cat * emb_dim; // 6 + 32 = 38

    // Expected weight size:
    // 5 int32 header + n_ord*2 float32 + n_cat*256*emb_dim + in_dim*C + C
    long expected = 20
        + (long)n_ord * 2 * 4
        + (long)n_cat * 256 * emb_dim * 4
        + (long)in_dim * C * 4
        + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    const float *ptr = (const float *)(hd + 5);
    const float *ord_mean = ptr; ptr += n_ord;
    const float *ord_std = ptr; ptr += n_ord;
    const float *emb = ptr; ptr += (long)n_cat * 256 * emb_dim;
    const float *W = ptr; ptr += (long)in_dim * C;
    const float *b = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * in_dim);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Ordinal part
        for (int j = 0; j < n_ord; j++) {
            float val = (float)xi[ord_idx[j]];
            z[j] = (val - ord_mean[j]) / ord_std[j];
        }

        // Embedding part
        int off = n_ord;
        for (int c = 0; c < n_cat; c++) {
            const float *e = emb + ((long)c * 256 + xi[cat_idx[c]]) * emb_dim;
            for (int v = 0; v < emb_dim; v++) {
                z[off++] = e[v];
            }
        }

        // Linear layer
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < in_dim; j++) {
            float zj = z[j];
            const float *w = W + (long)j * C;
            for (int c = 0; c < C; c++) scores[c] += zj * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(z); free(scores);
    return 0;
}
```