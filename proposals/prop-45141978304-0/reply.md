```hypothesis
An embedding-based MLP (label-embmlp) replaces one-hot encoding with learned embeddings (dim=4) for the 8 categorical features, reducing input dimension from ~108 to 38 (6 ordinal + 8×4). A 32-unit ReLU hidden layer then captures interactions more parameter-efficiently than label-mlp16's one-hot+16-hidden. Targets >0.855 dev accuracy at <1500 Ir/item. Failure: accuracy ≤0.852 (label-linear) or cost ≥2000 Ir/item.
```
```python
"""Embedding MLP for tabular label tasks: ordinal z-scored, categorical via learned embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic: seed 1, single thread)
Exports: int32 magic 0x454D424C, d, H, C, E, n_ord, n_cat; float32 ord_mean[n_ord], ord_std[n_ord],
emb[n_cat*256*E], W1[(n_ord+n_cat*E)*H], b1[H], W2[H*C], b2[C]."""
import json, math, struct, sys

import numpy as np

E, H = 4, 32
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
    d = len(types)

    ord_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]
    n_ord, n_cat = len(ord_idx), len(cat_idx)
    in_dim = n_ord + n_cat * E

    rng = np.random.default_rng(SEED)

    # Ordinal normalization stats
    ord_mean = np.zeros(n_ord, np.float32)
    ord_std = np.ones(n_ord, np.float32)
    for i, j in enumerate(ord_idx):
        v = X[:, j].astype(np.float64)
        ord_mean[i] = v.mean()
        ord_std[i] = v.std() or 1.0

    # Categorical embeddings: (n_cat, 256, E)
    emb = (rng.uniform(-1, 1, (n_cat, 256, E)) * math.sqrt(6.0 / (256 * E))).astype(np.float32)

    # MLP weights
    W1 = (rng.uniform(-1, 1, (in_dim, H)) * math.sqrt(6.0 / in_dim)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Build input batch: [ordinal_norm | embeddings]
            z = np.empty((bs, in_dim), np.float32)
            # Ordinal
            for i, j in enumerate(ord_idx):
                z[:, i] = (xb[:, j].astype(np.float32) - ord_mean[i]) / ord_std[i]
            # Categorical embeddings
            off = n_ord
            for ci, j in enumerate(cat_idx):
                z[:, off:off+E] = emb[ci, xb[:, j]]
                off += E

            # Forward
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            g = probs / bs  # [bs, C]

            # Backprop
            gW2 = h.T @ g + L2 * W2
            gb2 = g.sum(0)
            gh = g @ W2.T * (a > 0)
            gW1 = z.T @ gh + L2 * W1
            gb1 = gh.sum(0)
            gemb = np.zeros_like(emb)
            off = n_ord
            for ci, j in enumerate(cat_idx):
                # scatter-add: for each sample, add gh[sample] to gemb[ci, xb[sample, j]]
                np.add.at(gemb[ci], xb[:, j], gh[:, off:off+E])
                off += E

            grads = [gemb, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    magic = 0x454D424C  # 'EMBL'
    hdr = struct.pack("<7i", magic, d, H, C, E, n_ord, n_cat)
    out = hdr
    out += ord_mean.astype("<f4").tobytes()
    out += ord_std.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Embedding MLP for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (ordinal + categorical). Output: 1 byte class index. */
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
    if (nw < 28 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], E = hd[4], n_ord = hd[5], n_cat = hd[6];
    int in_dim = n_ord + n_cat * E;
    long expected = 28L
        + 4L * n_ord + 4L * n_ord
        + 4L * (long)n_cat * 256 * E
        + 4L * (long)in_dim * H + 4L * H
        + 4L * (long)H * C + 4L * C;
    if (C < 1 || C > 256 || nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 7);
    const float *ord_mean = p; p += n_ord;
    const float *ord_std  = p; p += n_ord;
    const float *emb      = p; p += (long)n_cat * 256 * E;
    const float *W1       = p; p += (long)in_dim * H;
    const float *b1       = p; p += H;
    const float *W2       = p; p += (long)H * C;
    const float *b2       = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * H);
    float *s = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        int off = 0;
        // Ordinal features
        for (int j = 0; j < n_ord; j++) {
            h[off++] = (xi[j] - ord_mean[j]) / ord_std[j];
        }
        // Categorical embeddings
        for (int ci = 0; ci < n_cat; ci++) {
            int col = n_ord + ci;  // column index in input
            const float *e = emb + ((long)ci * 256 + xi[col]) * E;
            for (int k = 0; k < E; k++) h[off++] = e[k];
        }

        // Hidden layer: ReLU(W1^T h + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < in_dim; v++) sum += h[v] * W1[(long)v * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
            s[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```