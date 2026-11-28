```hypothesis
Family: label-embedlin (embedding + linear). Replaces one-hot for categorical features with learned embeddings (dim E=8), keeps ordinal as normalized scalars. This reduces input dimension from ~200 (one-hot) to 6+8*8=70, allowing a denser linear layer that can generalize across similar categories. Expect dev accuracy >0.855 (beating label-linear's 0.8524) with similar or lower inference cost (smaller matrix multiply). If accuracy <=0.852 or inference cost >label-linear, the hypothesis fails.
```

```python
"""label-embedlin: embedding + linear for tabular classification.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format (little-endian):
  magic(0x454D424C), d, E, C, n_ord, n_cat,
  feature_types[d] (uint8: 0=ordinal,1=categorical),
  for each feature: if ordinal: mean,std (float32); if cat: 256*E float32 embedding,
  linear_weight[(n_ord + n_cat*E) * C] float32,
  linear_bias[C] float32.
"""
import json, math, struct, sys

import numpy as np

E = 8                # embedding dimension for categorical features
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

    # Split train/tune (last 20% for tuning)
    n = X.shape[0]
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)

    rng = np.random.default_rng(SEED)

    # Identify ordinal/categorical columns
    ord_cols = [j for j, t in enumerate(types) if t == "ordinal"]
    cat_cols = [j for j, t in enumerate(types) if t == "categorical"]
    n_ord = len(ord_cols)
    n_cat = len(cat_cols)

    # Normalize ordinal features (using train split only)
    ord_mean = np.zeros(n_ord, np.float32)
    ord_std = np.zeros(n_ord, np.float32)
    for i, j in enumerate(ord_cols):
        v = X[train_idx, j].astype(np.float32)
        ord_mean[i] = v.mean()
        ord_std[i] = v.std()
        if ord_std[i] == 0:
            ord_std[i] = 1.0

    # Initialize embeddings for categorical features (256 categories each, including unseen=255)
    emb = np.zeros((n_cat, 256, E), np.float32)
    for i in range(n_cat):
        emb[i] = rng.uniform(-0.01, 0.01, (256, E)).astype(np.float32)

    # Linear layer input dimension
    in_dim = n_ord + n_cat * E
    W = rng.uniform(-0.01, 0.01, (in_dim, C)).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(X_batch):
        """X_batch: (B, d) uint8 -> logits (B, C)"""
        B = X_batch.shape[0]
        # Ordinal part
        ord_part = np.zeros((B, n_ord), np.float32)
        for i, j in enumerate(ord_cols):
            ord_part[:, i] = (X_batch[:, j].astype(np.float32) - ord_mean[i]) / ord_std[i]
        # Categorical embeddings
        cat_part = np.zeros((B, n_cat * E), np.float32)
        for i, j in enumerate(cat_cols):
            cat_part[:, i*E:(i+1)*E] = emb[i, X_batch[:, j]]
        # Concatenate
        feat = np.concatenate([ord_part, cat_part], axis=1)
        logits = feat @ W + b
        return logits, feat

    def softmax_grad(logits, y_batch):
        """Return gradient of cross-entropy w.r.t logits"""
        P = np.exp(logits - logits.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(y_batch)), y_batch] -= 1.0
        return P / len(y_batch)

    t = 0
    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(len(train_idx))
        for s in range(0, len(train_idx), BATCH):
            idx = train_idx[order[s:s+BATCH]]
            xb, yb = X[idx], y[idx]
            logits, feat = forward(xb)
            g_logits = softmax_grad(logits, yb)          # (B, C)
            # Gradient w.r.t linear layer
            g_W = feat.T @ g_logits + L2 * W
            g_b = g_logits.sum(axis=0)
            # Gradient w.r.t embeddings
            g_emb = np.zeros_like(emb)
            for i, j in enumerate(cat_cols):
                # g_feat for this cat block: (B, E)
                g_feat_cat = g_logits @ W[n_ord + i*E : n_ord + (i+1)*E].T
                np.add.at(g_emb[i], xb[:, j], g_feat_cat)
            # No L2 on embeddings (optional, but keep simple)
            grads = [g_emb, g_W, g_b]

            t += 1
            for p, gr, (m, v) in zip(params, grads, state):
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # magic, d, E, C, n_ord, n_cat
    header = struct.pack("<6i", 0x454D424C, d, E, C, n_ord, n_cat)
    # feature types
    feat_types = bytearray(1 if t == "categorical" else 0 for t in types)
    # ordinal stats
    ord_stats = b""
    for i in range(n_ord):
        ord_stats += struct.pack("<2f", ord_mean[i], ord_std[i])
    # embeddings
    emb_bytes = emb.astype("<f4").tobytes()
    # linear weights
    lin_w = W.astype("<f4").tobytes()
    lin_b = b.astype("<f4").tobytes()

    out = header + feat_types + ord_stats + emb_bytes + lin_w + lin_b
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embedlin inference: ./infer weights.bin in.rec N
 * Input record: d bytes (uint8 per feature). Output: 1 byte class index.
 * Weights format (little-endian):
 *   int32 magic(0x454D424C), d, E, C, n_ord, n_cat
 *   uint8 feature_types[d] (0=ordinal,1=categorical)
 *   for each feature in order:
 *     if ordinal: float32 mean, float32 std
 *     if categorical: 256*E float32 embedding table
 *   float32 linear_weight[(n_ord + n_cat*E) * C]
 *   float32 linear_bias[C]
 */
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
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    if (nw < 24) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hdr = (int32_t *)wb;
    if (hdr[0] != 0x454D424C) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], E = hdr[2], C = hdr[3], n_ord = hdr[4], n_cat = hdr[5];
    if (d <= 0 || E <= 0 || C <= 0 || C > 256 || n_ord < 0 || n_cat < 0 || n_ord + n_cat != d) {
        fprintf(stderr, "bad header values\n"); return 3;
    }

    unsigned char *ptr = wb + 24;
    unsigned char *feat_type = ptr; ptr += d;
    float *ord_mean = (float *)ptr; ptr += n_ord * 2 * sizeof(float);
    float *ord_std = ord_mean + n_ord;
    float *emb = (float *)ptr; ptr += (long)n_cat * 256 * E * sizeof(float);
    float *W = (float *)ptr; ptr += (long)(n_ord + n_cat * E) * C * sizeof(float);
    float *b = (float *)ptr;

    // Validate total size
    long expected = 24 + d + (long)n_ord * 2 * 4 + (long)n_cat * 256 * E * 4 +
                    (long)(n_ord + n_cat * E) * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    float *feat = malloc(sizeof(float) * (n_ord + n_cat * E));
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build feature vector
        float *f = feat;
        // Ordinal features
        for (int j = 0, o = 0; j < d; j++) {
            if (feat_type[j] == 0) {
                *f++ = (xi[j] - ord_mean[o]) / ord_std[o];
                o++;
            }
        }
        // Categorical embeddings
        for (int j = 0, c = 0; j < d; j++) {
            if (feat_type[j] == 1) {
                const float *e = emb + (long)c * 256 * E + (long)xi[j] * E;
                memcpy(f, e, E * sizeof(float));
                f += E;
                c++;
            }
        }

        // Linear layer
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int idx = 0; idx < n_ord + n_cat * E; idx++) {
            float v = feat[idx];
            if (v != 0.0f) {
                const float *w = W + (long)idx * C;
                for (int c = 0; c < C; c++) scores[c] += v * w[c];
            }
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(feat); free(scores); free(wb); if (x) free(x);
    return 0;
}
```