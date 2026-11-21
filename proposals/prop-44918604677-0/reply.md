```hypothesis
Family: label-mlp2s (two-layer MLP with summed per-feature embeddings).
Why: The Adult dataset has mixed ordinal/categorical features; a tiny MLP (H1=8, H2=4) can capture pairwise interactions that linear models miss, while staying far cheaper than GBDT or 16-unit MLP. The summed-embedding first layer (like label-linear) keeps inference at ~200 FLOPs/item.
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or Ir/item ≥ 1500 (half of label-mlp16), the family does not meet its goal.
```

```python
"""label-mlp2s: two-hidden-layer MLP with per-feature embedding tables.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Architecture: 14 features -> per-byte lookup tables (256 x H1) summed -> ReLU -> H1=8 -> ReLU -> H2=4 -> linear -> 2 classes.
Exported weights: magic 0x4D4C5032, d, H1, H2, C; float32 T[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C]."""
import json, math, struct, sys

import numpy as np

H1, H2 = 8, 4
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
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # Build per-feature encoding matrices (same as label-linear)
    enc = []
    enc_dim = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None]          # (256, 1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float64)
            E[vals, np.arange(len(vals))] = 1.0                # (256, n_cat)
        enc.append(E.astype(np.float32))
        enc_dim.append(E.shape[1])

    # Concatenated design matrix Z (n, total_enc_dim)
    total_dim = sum(enc_dim)
    Z = np.empty((X.shape[0], total_dim), np.float32)
    off = 0
    for j, E in enumerate(enc):
        Z[:, off:off + enc_dim[j]] = E[X[:, j]]
        off += enc_dim[j]

    n, k = Z.shape
    # Initialize weights
    W1 = (rng.uniform(-1, 1, (k, H1)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
    b3 = np.zeros(C, np.float32)

    params = [W1, b1, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # forward
            a1 = z @ W1 + b1
            h1 = np.maximum(a1, 0)
            a2 = h1 @ W2 + b2
            h2 = np.maximum(a2, 0)
            logits = h2 @ W3 + b3
            g3 = softmax_grad(logits, yy)                     # (B, C)
            # backward
            g_h2 = g3 @ W3.T
            g_h2[a2 <= 0] = 0
            g2 = g_h2
            g_h1 = g2 @ W2.T
            g_h1[a1 <= 0] = 0
            g1 = g_h1
            grads = [
                z.T @ g1 + L2 * W1, g1.sum(0),
                h1.T @ g2 + L2 * W2, g2.sum(0),
                h2.T @ g3 + L2 * W3, g3.sum(0)
            ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Factor first layer into per-feature tables T[j][256][H1]
    T = []
    off = 0
    for j, dim in enumerate(enc_dim):
        W1_j = W1[off:off + dim]                              # (dim, H1)
        T_j = enc[j] @ W1_j                                   # (256, H1)
        T.append(T_j.astype(np.float32))
        off += dim

    # Write weights file
    magic = 0x4D4C5032  # 'MLP2'
    out = struct.pack("<5i", magic, d, H1, H2, C)
    for Tj in T:
        out += Tj.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    out += W3.astype("<f4").tobytes()
    out += b3.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-mlp2s inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte (class index).
 * Weights format (little-endian):
 *   int32 magic=0x4D4C5032, d, H1, H2, C
 *   float T[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C] */
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
    if (nw < 20 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H1 + H1 + (long)H1 * H2 + H2 + (long)H2 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * H1;
    const float *W2 = b1 + H1;
    const float *b2 = W2 + (long)H1 * H2;
    const float *W3 = b2 + H2;
    const float *b3 = W3 + (long)H2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h1 = malloc(sizeof(float) * H1);
    float *h2 = malloc(sizeof(float) * H2);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* first layer: sum per-feature tables */
        for (int u = 0; u < H1; u++) h1[u] = b1[u];
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((long)j * 256 + xi[j]) * H1;
            for (int u = 0; u < H1; u++) h1[u] += tj[u];
        }
        /* ReLU */
        for (int u = 0; u < H1; u++) if (h1[u] < 0.0f) h1[u] = 0.0f;

        /* second layer */
        for (int v = 0; v < H2; v++) h2[v] = b2[v];
        for (int u = 0; u < H1; u++) {
            float hu = h1[u];
            const float *w2u = W2 + (long)u * H2;
            for (int v = 0; v < H2; v++) h2[v] += hu * w2u[v];
        }
        /* ReLU */
        for (int v = 0; v < H2; v++) if (h2[v] < 0.0f) h2[v] = 0.0f;

        /* output layer */
        for (int c = 0; c < C; c++) logits[c] = b3[c];
        for (int v = 0; v < H2; v++) {
            float hv = h2[v];
            const float *w3v = W3 + (long)v * C;
            for (int c = 0; c < C; c++) logits[c] += hv * w3v[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```