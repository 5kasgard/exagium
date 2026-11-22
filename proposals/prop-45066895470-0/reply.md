```hypothesis
Family: label-emblinear (embedding + linear). Ordinal features z-scored; each categorical feature gets a learned 4-dim embedding (unseen=255). Concatenated embeddings + ordinal features feed a linear layer. Fewer params than one-hot linear, captures category similarities, faster inference than MLP. Should beat label-linear (0.8524) on accuracy at lower Ir/item than label-mlp16. Fails if embeddings don't converge or unseen handling hurts.
```
```python
"""label-emblinear: embedding + linear for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Ordinal: z-scored (1 dim each). Categorical: learned 4-dim embeddings (255=unseen).
Linear output layer. Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Deterministic seed=1. Exports: magic 0x454D424C, d, n_ord, n_cat, C, E=4;
ord_mean[d_ord], ord_std[d_ord]; cat_vocab_size[n_cat]; emb_tables[sum(vocab+1)*E];
lin_W[(n_ord + n_cat*E) x C], lin_b[C]. All float32 little-endian."""
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
    d = X.shape[1]

    # Identify ordinal/categorical columns
    ord_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]
    n_ord, n_cat = len(ord_idx), len(cat_idx)

    rng = np.random.default_rng(SEED)

    # Ordinal stats
    ord_mean = np.zeros(n_ord, np.float64)
    ord_std = np.ones(n_ord, np.float64)
    for k, j in enumerate(ord_idx):
        v = X[:, j].astype(np.float64)
        ord_mean[k] = v.mean()
        ord_std[k] = v.std() or 1.0

    # Categorical vocab sizes (from training data, +1 for unseen=255)
    cat_vocab = []
    cat_offsets = [0]
    for j in cat_idx:
        vocab = int(X[:, j].max()) + 1  # 0..max_seen
        vocab = max(vocab, 255) + 1     # include index 255 for unseen
        cat_vocab.append(vocab)
        cat_offsets.append(cat_offsets[-1] + vocab)
    cat_vocab = np.array(cat_vocab, np.int32)
    total_cat_emb = cat_offsets[-1] * E

    # Build input matrix: [ordinal_z | cat_embeddings]
    # We'll train embeddings directly, so input is just indices for cat, z-scored for ord
    Z_ord = np.zeros((X.shape[0], n_ord), np.float32)
    for k, j in enumerate(ord_idx):
        Z_ord[:, k] = (X[:, j].astype(np.float32) - ord_mean[k]) / ord_std[k]

    # Categorical indices (clamp 255 to vocab-1 which is the unseen slot)
    Z_cat_idx = X[:, cat_idx].astype(np.int32)
    for k in range(n_cat):
        Z_cat_idx[:, k] = np.minimum(Z_cat_idx[:, k], cat_vocab[k] - 1)

    n = X.shape[0]
    in_dim = n_ord + n_cat * E

    # Parameters
    emb = (rng.uniform(-1, 1, (cat_offsets[-1], E)) * math.sqrt(6.0 / E)).astype(np.float32)
    W = (rng.uniform(-1, 1, (in_dim, C)) * math.sqrt(6.0 / in_dim)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [emb, W, b]

    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z_ord = Z_ord[idx]
            z_cat = Z_cat_idx[idx]
            yy = y[idx]
            bs = len(idx)

            # Forward
            # Embedding lookup: sum over categorical features
            emb_part = np.zeros((bs, n_cat * E), np.float32)
            for k in range(n_cat):
                off = cat_offsets[k] * E
                e = emb[cat_offsets[k]:cat_offsets[k+1]]
                emb_part[:, k*E:(k+1)*E] = e[z_cat[:, k]]
            z = np.concatenate([z_ord, emb_part], axis=1)
            logits = z @ W + b

            # Backward
            g = softmax_grad(logits, yy)  # [bs, C]
            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(0)
            grad_z = g @ W.T  # [bs, in_dim]

            # Split grad_z to ordinal and embedding parts
            grad_ord = grad_z[:, :n_ord]
            grad_emb_part = grad_z[:, n_ord:]

            # Embedding gradients
            grad_emb = np.zeros_like(emb)
            for k in range(n_cat):
                off = cat_offsets[k]
                ge = grad_emb_part[:, k*E:(k+1)*E]
                idx_k = z_cat[:, k]
                np.add.at(grad_emb, idx_k + off, ge)
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Export weights
    # magic, d, n_ord, n_cat, C, E
    out = struct.pack("<6i", 0x454D424C, d, n_ord, n_cat, C, E)
    # ord_mean, ord_std (float32)
    out += ord_mean.astype("<f4").tobytes()
    out += ord_std.astype("<f4").tobytes()
    # cat_vocab (int32)
    out += cat_vocab.astype("<i4").tobytes()
    # embeddings (float32)
    out += emb.astype("<f4").tobytes()
    # linear W, b
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-emblinear: embedding + linear for tabular label tasks.
 * ./infer weights.bin in.rec N
 * Weights format: magic 0x454D424C, d, n_ord, n_cat, C, E;
 * ord_mean[n_ord], ord_std[n_ord] (float32);
 * cat_vocab[n_cat] (int32);
 * emb[sum(cat_vocab) * E] (float32);
 * lin_W[(n_ord + n_cat*E) * C], lin_b[C] (float32). */
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 24 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], n_ord = hd[2], n_cat = hd[3], C = hd[4], E = hd[5];
    if (C < 1 || C > 256 || E < 1 || E > 16 || n_ord < 0 || n_cat < 0 || n_ord + n_cat != d) {
        fprintf(stderr, "bad header\n"); return 3;
    }

    const float *ord_mean = (const float *)(hd + 6);
    const float *ord_std = ord_mean + n_ord;
    const int32_t *cat_vocab = (const int32_t *)(ord_std + n_ord);
    const float *emb = (const float *)(cat_vocab + n_cat);

    long emb_total = 0;
    for (int k = 0; k < n_cat; k++) emb_total += cat_vocab[k];
    const float *W = emb + emb_total * E;
    const float *b = W + (n_ord + n_cat * E) * C;

    long expected = 24 + 4L * (n_ord + n_ord + n_cat + emb_total * E + (n_ord + n_cat * E) * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * (n_ord + n_cat * E));
    float *s = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Ordinal part
        for (int j = 0; j < n_ord; j++) {
            z[j] = (xi[j] - ord_mean[j]) / ord_std[j];
        }

        // Categorical embeddings
        int off = n_ord;
        for (int c = 0; c < n_cat; c++) {
            int idx = xi[n_ord + c];
            if (idx >= cat_vocab[c]) idx = cat_vocab[c] - 1; // unseen -> last slot
            const float *e = emb + (cat_vocab[c] * c + idx) * E; // WRONG: need cumulative offset
        }

        // Fix: need cumulative offsets for embeddings
    }

    // Recompute with correct embedding offsets
    int32_t *cat_offset = malloc(sizeof(int32_t) * (n_cat + 1));
    cat_offset[0] = 0;
    for (int c = 0; c < n_cat; c++) cat_offset[c+1] = cat_offset[c] + cat_vocab[c];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < n_ord; j++) {
            z[j] = (xi[j] - ord_mean[j]) / ord_std[j];
        }

        int off = n_ord;
        for (int c = 0; c < n_cat; c++) {
            int idx = xi[n_ord + c];
            if (idx >= cat_vocab[c]) idx = cat_vocab[c] - 1;
            const float *e = emb + (cat_offset[c] + idx) * E;
            for (int e_i = 0; e_i < E; e_i++) z[off++] = e[e_i];
        }

        // Linear layer
        for (int c = 0; c < C; c++) s[c] = b[c];
        int in_dim = n_ord + n_cat * E;
        for (int j = 0; j < in_dim; j++) {
            float zj = z[j];
            const float *wj = W + (long)j * C;
            for (int c = 0; c < C; c++) s[c] += zj * wj[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(cat_offset);
    free(z);
    free(s);
    return 0;
}
```