```hypothesis
Family: label-emb-mlp. Uses learned embeddings for categorical features (dim 4) and normalized ordinal features, followed by a single ReLU hidden layer (32 units). This reduces input dimensionality vs one-hot, allowing richer interactions with fewer parameters. Should beat label-linear (0.8524) on Adult dev; if dev accuracy <= 0.8524, the hypothesis fails.
```

```python
"""label-emb-mlp: Embedding + MLP for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Architecture: categorical features -> embeddings (E=4), ordinal -> z-scored scalars.
Concatenated -> ReLU hidden layer (H=32) -> linear output (C=2).
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights format: magic 0x454D424D, d, n_ord, n_cat, E, H, C, d bytes feature_type (0=ord,1=cat),
then for each cat: 256*E float32 embeddings, for each ord: 2 float32 (mean,std),
then W1[input_dim,H], b1[H], W2[H,C], b2[C] (all float32 little-endian)."""
import json, math, struct, sys

import numpy as np

E = 4
H = 32
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

    # Separate ordinal and categorical indices
    ord_idx = [i for i, t in enumerate(types) if t == "ordinal"]
    cat_idx = [i for i, t in enumerate(types) if t == "categorical"]
    n_ord, n_cat = len(ord_idx), len(cat_idx)
    input_dim = n_ord + n_cat * E

    rng = np.random.default_rng(SEED)

    # Normalize ordinal features
    ord_mean = np.zeros(n_ord, np.float32)
    ord_std = np.ones(n_ord, np.float32)
    for k, j in enumerate(ord_idx):
        v = X[:, j].astype(np.float64)
        mu, sd = v.mean(), v.std()
        ord_mean[k] = mu
        ord_std[k] = sd if sd > 0 else 1.0

    # Initialize embeddings for categorical features (256 categories each)
    emb = (rng.normal(0, math.sqrt(2.0 / (256 + E)), (n_cat, 256, E))).astype(np.float32)

    # Build input matrix Z (n, input_dim)
    n = X.shape[0]
    Z = np.empty((n, input_dim), dtype=np.float32)
    # Ordinal part
    for k, j in enumerate(ord_idx):
        Z[:, k] = (X[:, j].astype(np.float32) - ord_mean[k]) / ord_std[k]
    # Categorical part: embeddings
    off = n_ord
    for k, j in enumerate(cat_idx):
        Z[:, off:off+E] = emb[k, X[:, j]]
        off += E

    # Initialize MLP weights
    W1 = (rng.normal(0, math.sqrt(2.0 / (input_dim + H)), (input_dim, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / (H + C)), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # Forward
            a1 = z @ W1 + b1
            h = np.maximum(a1, 0, dtype=np.float32)
            logits = h @ W2 + b2
            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g2 = probs / len(yy)  # (batch, C)
            # Backprop
            grad_W2 = h.T @ g2 + L2 * W2
            grad_b2 = g2.sum(axis=0)
            gh = g2 @ W2.T * (a1 > 0)
            grad_W1 = z.T @ gh + L2 * W1
            grad_b1 = gh.sum(axis=0)
            grads = [grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Header: magic, d, n_ord, n_cat, E, H, C
    header = struct.pack("<7I", 0x454D424D, d, n_ord, n_cat, E, H, C)
    # Feature types: 0=ordinal, 1=categorical
    feat_types = bytearray(d)
    for j in cat_idx:
        feat_types[j] = 1
    # Embeddings
    emb_bytes = emb.astype("<f4").tobytes()
    # Ordinal stats
    ord_stats = np.stack([ord_mean, ord_std], axis=1).astype("<f4").tobytes()
    # MLP weights
    W1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    W2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()

    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(feat_types)
        f.write(emb_bytes)
        f.write(ord_stats)
        f.write(W1_bytes)
        f.write(b1_bytes)
        f.write(W2_bytes)
        f.write(b2_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (0 or 1).
 * Weights format: see train.py. */
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
    unsigned char *w = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Parse header
    if (nw < 28) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t *hd = (uint32_t *)w;
    if (hd[0] != 0x454D424D) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], n_ord = hd[2], n_cat = hd[3], E = hd[4], H = hd[5], C = hd[6];
    if (d != 14 || C != 2 || n_ord + n_cat != d) { fprintf(stderr, "bad header dims\n"); return 3; }
    unsigned char *feat_types = w + 28;
    // Pointers to data sections
    float *emb = (float *)(feat_types + d);
    float *ord_stats = emb + n_cat * 256 * E;
    float *W1 = ord_stats + n_ord * 2;
    float *b1 = W1 + (n_ord + n_cat * E) * H;
    float *W2 = b1 + H;
    float *b2 = W2 + H * C;

    // Validate total size
    long expected = 28 + d + (long)n_cat * 256 * E * 4 + (long)n_ord * 2 * 4
                    + (long)(n_ord + n_cat * E) * H * 4 + H * 4 + (long)H * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    int input_dim = n_ord + n_cat * E;
    float *hidden = malloc(H * sizeof(float));
    float *output = malloc(C * sizeof(float));
    float *in_vec = malloc(input_dim * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build input vector
        int ord_ptr = 0, cat_ptr = 0, vec_ptr = 0;
        for (int j = 0; j < d; j++) {
            if (feat_types[j] == 0) { // ordinal
                float mean = ord_stats[ord_ptr * 2];
                float std = ord_stats[ord_ptr * 2 + 1];
                in_vec[vec_ptr++] = (xi[j] - mean) / std;
                ord_ptr++;
            } else { // categorical
                const float *e = emb + (cat_ptr * 256 + xi[j]) * E;
                for (int e_i = 0; e_i < E; e_i++) in_vec[vec_ptr++] = e[e_i];
                cat_ptr++;
            }
        }

        // Hidden layer: ReLU(in_vec @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int v = 0; v < input_dim; v++) sum += in_vec[v] * W1[v * H + h];
            hidden[h] = fmaxf(sum, 0.0f);
        }

        // Output layer: hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[h * C + c];
            output[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (output[c] > output[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(hidden); free(output); free(in_vec);
    free(w); if (x) free(x);
    return 0;
}
```