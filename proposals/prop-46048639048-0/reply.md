```hypothesis
Family: label-embed (entity embeddings + shallow MLP). Categorical features use learned dense embeddings (dim=8) instead of one-hot, capturing category similarities; ordinal features are z-scored. A single hidden layer (32 ReLU units) models interactions. This should beat label-linear (0.8524) by learning non-linear feature crosses at similar cost (~2500 FLOPs/item vs label-linear's ~700 Ir/item but with better accuracy). Failure: dev accuracy ≤ 0.853 or Ir/item ≥ 3000.
```
```python
import json, math, struct, sys

import numpy as np

EMBED_DIM = 8
H = 32
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x44424D45  # 'EMBD' little-endian


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

    # Identify ordinal/categorical columns
    ord_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]
    n_ord, n_cat = len(ord_idx), len(cat_idx)

    rng = np.random.default_rng(SEED)

    # Ordinal normalization stats (on binned values 0-254)
    ord_mean = X[:, ord_idx].mean(axis=0).astype(np.float32)
    ord_std = X[:, ord_idx].std(axis=0).astype(np.float32)
    ord_std[ord_std == 0] = 1.0

    # Embedding tables: n_cat x 256 x EMBED_DIM
    # Initialize with small random values; index 255 (unseen) stays near zero
    emb = (rng.uniform(-0.01, 0.01, (n_cat, 256, EMBED_DIM)) * math.sqrt(6.0 / 256)).astype(np.float32)

    # MLP weights
    in_dim = n_ord + n_cat * EMBED_DIM
    W1 = (rng.uniform(-1, 1, (in_dim, H)) * math.sqrt(6.0 / in_dim)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0

    def forward(Xb):
        # Xb: (batch, 14) uint8
        batch = Xb.shape[0]
        # Ordinal part
        z_ord = ((Xb[:, ord_idx].astype(np.float32) - ord_mean) / ord_std)  # (batch, n_ord)
        # Categorical embeddings
        z_cat_list = []
        for ci, cj in enumerate(cat_idx):
            z_cat_list.append(emb[ci, Xb[:, cj]])  # (batch, EMBED_DIM)
        z_cat = np.concatenate(z_cat_list, axis=1) if z_cat_list else np.zeros((batch, 0), np.float32)
        # Concat
        z = np.concatenate([z_ord, z_cat], axis=1)  # (batch, in_dim)
        # Hidden
        a = z @ W1 + b1
        h = np.maximum(a, 0)
        # Output
        logits = h @ W2 + b2
        return z, a, h, logits

    def softmax_grad(logits, yy):
        S = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yy = X[idx], y[idx]

            z, a, h, logits = forward(Xb)
            g = softmax_grad(logits, yy)  # (batch, C)

            # Backprop
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(axis=0)
            gh = g @ W2.T * (a > 0)  # (batch, H)
            grad_W1 = z.T @ gh + L2 * W1
            grad_b1 = gh.sum(axis=0)
            gz = gh @ W1.T  # (batch, in_dim)

            # Embedding gradients
            grad_emb = np.zeros_like(emb)
            off = n_ord
            for ci in range(n_cat):
                e_slice = gz[:, off:off + EMBED_DIM]  # (batch, EMBED_DIM)
                cat_vals = Xb[:, cat_idx[ci]]  # (batch,)
                for b in range(e_slice.shape[0]):
                    grad_emb[ci, cat_vals[b]] += e_slice[b]
                off += EMBED_DIM
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # Header: magic, d, n_cat, EMBED_DIM, H, C, n_ord
    header = struct.pack("<7i", MAGIC, d, n_cat, EMBED_DIM, H, C, n_ord)
    # Ordinal stats: mean[n_ord], std[n_ord]
    body = ord_mean.astype("<f4").tobytes() + ord_std.astype("<f4").tobytes()
    # Embeddings: n_cat * 256 * EMBED_DIM
    body += emb.astype("<f4").tobytes()
    # MLP: W1[in_dim, H], b1[H], W2[H, C], b2[C]
    body += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    body += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
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
    if (nw < 28 || hd[0] != 0x44424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], n_cat = hd[2], EMBED_DIM = hd[3], H = hd[4], C = hd[5], n_ord = hd[6];
    if (d != 14 || n_ord + n_cat != d || C != 2 || H <= 0 || EMBED_DIM <= 0) {
        fprintf(stderr, "bad weights header\n"); return 3;
    }
    int in_dim = n_ord + n_cat * EMBED_DIM;
    long expected = 28L + 4L * (2L * n_ord + (long)n_cat * 256 * EMBED_DIM + (long)in_dim * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 7);
    const float *ord_mean = p; p += n_ord;
    const float *ord_inv_std = p; p += n_ord;  // we store 1/std for faster inference
    const float *emb = p; p += (long)n_cat * 256 * EMBED_DIM;
    const float *W1 = p; p += (long)in_dim * H;
    const float *b1 = p; p += H;
    const float *W2 = p; p += (long)H * C;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * in_dim);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    // Ordinal column indices (from card: positions 0,2,4,10,11,12 are ordinal)
    int ord_idx[6] = {0, 2, 4, 10, 11, 12};
    int cat_idx[8] = {1, 3, 5, 6, 7, 8, 9, 13};

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Ordinal features: normalize
        for (int j = 0; j < n_ord; j++) {
            z[j] = (xi[ord_idx[j]] - ord_mean[j]) * ord_inv_std[j];
        }
        // Categorical embeddings
        int off = n_ord;
        for (int ci = 0; ci < n_cat; ci++) {
            const float *e = emb + ((long)ci * 256 + xi[cat_idx[ci]]) * EMBED_DIM;
            for (int ed = 0; ed < EMBED_DIM; ed++) z[off++] = e[ed];
        }

        // Hidden layer: ReLU(z @ W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < in_dim; v++) sum += z[v] * W1[(long)v * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```