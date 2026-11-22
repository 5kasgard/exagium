```hypothesis
Family: label-embedding. Learn a small dense embedding (8 dims) for each categorical column, z-score ordinal columns, then a linear classifier on the concatenated vector. Embeddings capture category similarities that one-hot misses, with far fewer parameters than one-hot (8 cats × 256 × 8 = 16 KB vs one-hot's ~10 KB but sparse). Inference is pure table lookups + a dot product, so CPU cost should match or beat label-linear. If dev accuracy ≤ 0.8524 or Ir/item ≥ 800, the hypothesis fails.
```
```python
"""label-embedding: embeddings for categorical, linear on top.
python3 -s -P train.py card.json train.in train.key weights.bin"""
import json, math, struct, sys

import numpy as np

EMB_DIM = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-5, 256, 800000, 100


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

    # Identify ordinal and categorical columns
    ord_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]

    # Ordinal: compute mean/std on training data
    ord_mu = X[:, ord_idx].mean(axis=0).astype(np.float32) if ord_idx else np.array([], np.float32)
    ord_sd = X[:, ord_idx].std(axis=0).astype(np.float32) if ord_idx else np.array([], np.float32)
    ord_sd[ord_sd == 0] = 1.0

    # Categorical: learn embeddings (256 values per column, last is unseen=255)
    n_cat = len(cat_idx)
    emb = (rng.uniform(-0.01, 0.01, (n_cat, 256, EMB_DIM)) * math.sqrt(6.0 / 256)).astype(np.float32)

    # Linear layer on top: input dim = n_ord + n_cat * EMB_DIM
    n_ord = len(ord_idx)
    in_dim = n_ord + n_cat * EMB_DIM
    W = np.zeros((in_dim, C), np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    n = X.shape[0]
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yy = y[idx]
            bs = len(idx)

            # Build features
            feat = np.zeros((bs, in_dim), np.float32)
            if n_ord:
                feat[:, :n_ord] = (xb[:, ord_idx].astype(np.float32) - ord_mu) / ord_sd
            off = n_ord
            for ci, j in enumerate(cat_idx):
                e = emb[ci, xb[:, j]]  # (bs, EMB_DIM)
                feat[:, off:off + EMB_DIM] = e
                off += EMB_DIM

            # Forward
            logits = feat @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yy] -= 1.0
            grad_logits = (probs / bs).astype(np.float32)

            # Gradients
            grad_W = feat.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_feat = grad_logits @ W.T

            grad_emb = np.zeros_like(emb)
            off = n_ord
            for ci, j in enumerate(cat_idx):
                g = grad_feat[:, off:off + EMB_DIM]  # (bs, EMB_DIM)
                np.add.at(grad_emb[ci], xb[:, j], g)
                off += EMB_DIM
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights: magic, d, n_ord, n_cat, EMB_DIM, C
    # Then: ord_mu[d_ord], ord_sd[d_ord], emb[n_cat][256][EMB_DIM], W[in_dim][C], b[C]
    # All float32 little-endian
    magic = 0x454D4244  # "DBME"
    hdr = struct.pack("<5i", magic, d, n_ord, n_cat, C)
    out = hdr
    out += ord_mu.astype("<f4").tobytes()
    out += ord_sd.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-embedding inference: ./infer weights.bin in.rec N */
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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x454D4244) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], n_ord = hd[2], n_cat = hd[3], C = hd[4];
    if (C < 1 || C > 256 || n_ord < 0 || n_cat < 0 || n_ord + n_cat != d) { fprintf(stderr, "bad dims\n"); return 3; }
    const int EMB_DIM = 8;
    int in_dim = n_ord + n_cat * EMB_DIM;
    long expected = 20L + 4L * (n_ord + n_ord + (long)n_cat * 256 * EMB_DIM + (long)in_dim * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *ord_mu = p; p += n_ord;
    const float *ord_sd = p; p += n_ord;
    const float *emb = p; p += (long)n_cat * 256 * EMB_DIM;
    const float *W = p; p += (long)in_dim * C;
    const float *b = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * in_dim);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        int off = 0;
        for (int j = 0, oi = 0; j < d; j++) {
            // We need to know which columns are ordinal vs categorical.
            // The training code uses the card's types array order.
            // But we don't have the card at inference. We must encode the column types in weights.
            // PROBLEM: The current weight format doesn't store which columns are ordinal/categorical.
            // Need to fix this. Let me add a bitmap or array of column types.
            // Since we can't change train.py now, I'll assume the card order is fixed for adult.
            // But the contract says the candidate must work for any card with same schema.
            // The card is only available at train time. infer.c must be self-contained.
            // So we MUST store the column type info in the weights file.
            // I'll add a byte array of length d after the header indicating type (0=ord, 1=cat).
            // But the train.py above doesn't do that. I need to fix train.py to include it.
            // Since I can't edit previous block, I'll assume the adult task column types are fixed
            // and hardcode them here. But that violates the "any card" requirement.
            // Actually, the example label-linear encodes per-byte lookup tables that implicitly
            // handle the type via the encoding. It stores T[d][256][W] where for ordinal it's
            // the z-scored value times weights, for categorical it's one-hot.
            // For embeddings, we need to know which columns are which.
            // I'll store a bitmask of categorical columns in the header.
            // Let me redesign: add a uint8_t array of length d after C in header.
            // But I already wrote train.py without it. The inference will fail.
            // Since this is a thought exercise, I'll proceed assuming we can read the types
            // from a known schema. For the purpose of this answer, I'll hardcode adult's types.
            // In reality, train.py must write the column types.
        }
        // This is incomplete due to the missing column type info.
        // Let me provide a corrected version that includes types in weights.
    }
    return 0;
}
```

Wait, I made a critical error - the infer.c doesn't have the column type information. I need to fix both train.py and infer.c to include the column types in the weights file. Let me provide a corrected complete answer.```hypothesis
Family: label-embedding. Learn a small dense embedding (8 dims) for each categorical column, z-score ordinal columns, then a linear classifier on the concatenated vector. Embeddings capture category similarities that one-hot misses, with far fewer parameters than one-hot (8 cats × 256 × 8 = 16 KB vs one-hot's ~10 KB but sparse). Inference is pure table lookups + a dot product, so CPU cost should match or beat label-linear. If dev accuracy ≤ 0.8524 or Ir/item ≥ 800, the hypothesis fails.
```
```python
"""label-embedding: embeddings for categorical, linear on top.
python3 -s -P train.py card.json train.in train.key weights.bin"""
import json, math, struct, sys

import numpy as np

EMB_DIM = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-5, 256, 800000, 100


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

    # Column types: 0=ordinal, 1=categorical
    col_type = np.array([0 if t == "ordinal" else 1 for t in types], dtype=np.uint8)
    ord_idx = np.where(col_type == 0)[0]
    cat_idx = np.where(col_type == 1)[0]
    n_ord = len(ord_idx)
    n_cat = len(cat_idx)

    # Ordinal: compute mean/std on training data
    ord_mu = X[:, ord_idx].mean(axis=0).astype(np.float32) if n_ord else np.array([], np.float32)
    ord_sd = X[:, ord_idx].std(axis=0).astype(np.float32) if n_ord else np.array([], np.float32)
    if n_ord:
        ord_sd[ord_sd == 0] = 1.0

    # Categorical: learn embeddings (256 values per column, last is unseen=255)
    emb = (rng.uniform(-0.01, 0.01, (n_cat, 256, EMB_DIM)) * math.sqrt(6.0 / 256)).astype(np.float32)

    # Linear layer on top: input dim = n_ord + n_cat * EMB_DIM
    in_dim = n_ord + n_cat * EMB_DIM
    W = np.zeros((in_dim, C), np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    n = X.shape[0]
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yy = y[idx]
            bs = len(idx)

            # Build features
            feat = np.zeros((bs, in_dim), np.float32)
            if n_ord:
                feat[:, :n_ord] = (xb[:, ord_idx].astype(np.float32) - ord_mu) / ord_sd
            off = n_ord
            for ci, j in enumerate(cat_idx):
                e = emb[ci, xb[:, j]]  # (bs, EMB_DIM)
                feat[:, off:off + EMB_DIM] = e
                off += EMB_DIM

            # Forward
            logits = feat @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yy] -= 1.0
            grad_logits = (probs / bs).astype(np.float32)

            # Gradients
            grad_W = feat.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_feat = grad_logits @ W.T

            grad_emb = np.zeros_like(emb)
            off = n_ord
            for ci, j in enumerate(cat_idx):
                g = grad_feat[:, off:off + EMB_DIM]  # (bs, EMB_DIM)
                np.add.at(grad_emb[ci], xb[:, j], g)
                off += EMB_DIM
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights:
    # magic(4), d(4), n_ord(4), n_cat(4), C(4), EMB_DIM(4)
    # col_type[d] (uint8)
    # ord_mu[n_ord], ord_sd[n_ord] (float32)
    # emb[n_cat][256][EMB_DIM] (float32)
    # W[in_dim][C], b[C] (float32)
    magic = 0x454D4244  # "DBME"
    hdr = struct.pack("<6i", magic, d, n_ord, n_cat, C, EMB_DIM)
    out = hdr
    out += col_type.astype(np.uint8).tobytes()
    out += ord_mu.astype("<f4").tobytes()
    out += ord_sd.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-embedding inference: ./infer weights.bin in.rec N */
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
    if (nw < 24) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x454D4244) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], n_ord = hd[2], n_cat = hd[3], C = hd[4], EMB_DIM = hd[5];
    if (C < 1 || C > 256 || n_ord < 0 || n_cat < 0 || n_ord + n_cat != d || EMB_DIM != 8) {
        fprintf(stderr, "bad dims\n"); return 3;
    }
    const unsigned char *col_type = wb + 24;
    const float *p = (const float *)(col_type + d);
    const float *ord_mu = p; p += n_ord;
    const float *ord_sd = p; p += n_ord;
    const float *emb = p; p += (long)n_cat * 256 * EMB_DIM;
    int in_dim = n_ord + n_cat * EMB_DIM;
    const float *W = p; p += (long)in_dim * C;
    const float *b = p;

    long expected = 24L + d + 4L * (n_ord + n_ord + (long)n_cat * 256 * EMB_DIM + (long)in_dim * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * in_dim);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        int off = 0;
        for (int j = 0, oi = 0, ci = 0; j < d; j++) {
            if (col_type[j] == 0) {
                feat[off++] = (xi[j] - ord_mu[oi]) / ord_sd[oi];
                oi++;
            } else {
                const float *e = emb + ((long)ci * 256 + xi[j]) * EMB_DIM;
                for (int k = 0; k < EMB_DIM; k++) feat[off++] = e[k];
                ci++;
            }
        }

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < in_dim; f++) {
            float xf = feat[f];
            if (xf == 0.0f) continue;
            const float *w = W + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += xf * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```