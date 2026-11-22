```hypothesis
Family: label-embed — linear model with learned embeddings for categorical features (ordinal features z-scored).
Why: The adult task has 8 categorical columns with up to 256 categories each; label-linear uses one-hot (2048 dim) which cannot share strength across rare categories. Embeddings (dim=8) reduce dimensionality to 64, learn dense representations, and improve generalization while keeping inference identical (same lookup-table format as label-linear).
Failure: If dev accuracy ≤ 0.8524 (label-linear) or training does not converge within limits, the hypothesis is invalid.
```
```python
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen on tuning split, then frozen)
E = 8                # embedding dimension per categorical feature
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
    d = X.shape[1]

    # Split features into ordinal and categorical
    ord_idx = [i for i, t in enumerate(types) if t == "ordinal"]
    cat_idx = [i for i, t in enumerate(types) if t == "categorical"]
    n_ord, n_cat = len(ord_idx), len(cat_idx)

    rng = np.random.default_rng(SEED)

    # Ordinal: compute mean/std on training data
    X_ord = X[:, ord_idx].astype(np.float64)
    mu = X_ord.mean(axis=0)
    sd = X_ord.std(axis=0)
    sd[sd == 0] = 1.0

    # Categorical: embeddings (n_cat, 256, E)
    Emb = (rng.uniform(-1, 1, (n_cat, 256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)

    # Linear layer: W_ord (n_ord, C), W_cat (n_cat, E, C), bias (C)
    W_ord = (rng.uniform(-1, 1, (n_ord, C)) * math.sqrt(6.0 / n_ord)).astype(np.float32)
    W_cat = (rng.uniform(-1, 1, (n_cat, E, C)) * math.sqrt(6.0 / (E * n_cat))).astype(np.float32)
    bias = np.zeros(C, np.float32)

    params = [Emb, W_ord, W_cat, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits, dtype=np.float32)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]

            # Forward
            # Ordinal part
            x_ord = xb[:, ord_idx].astype(np.float32)
            x_ord_norm = (x_ord - mu.astype(np.float32)) / sd.astype(np.float32)  # (B, n_ord)
            logits = x_ord_norm @ W_ord  # (B, C)

            # Categorical part
            for j, c in enumerate(cat_idx):
                emb = Emb[j][xb[:, c]]  # (B, E)
                logits += emb @ W_cat[j]  # (B, C)

            logits += bias

            # Backward
            g = softmax_grad(logits, yb)  # (B, C)

            # Gradients
            g_Emb = np.zeros_like(Emb)
            g_W_ord = x_ord_norm.T @ g + L2 * W_ord
            g_W_cat = np.zeros_like(W_cat)
            g_bias = g.sum(axis=0)

            for j, c in enumerate(cat_idx):
                emb = Emb[j][xb[:, c]]  # (B, E)
                g_W_cat[j] = emb.T @ g + L2 * W_cat[j]
                # scatter-add into g_Emb
                # For each sample, add g[sample] @ W_cat[j].T to g_Emb[j, category]
                # We can do a loop over batch (small) or use np.add.at
                for b in range(len(idx)):
                    cat_val = xb[idx[b], c]
                    g_Emb[j, cat_val] += g[b] @ W_cat[j].T

            grads = [g_Emb, g_W_ord, g_W_cat, g_bias]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build lookup tables T[d][256][C] and bias b1[C] for infer.c (H=0, W=C)
    T = np.zeros((d, 256, C), dtype=np.float32)
    # Ordinal features
    for i, j in enumerate(ord_idx):
        w = W_ord[i]  # (C,)
        for v in range(256):
            T[j, v] = ((v - mu[i]) / sd[i]) * w
    # Categorical features
    for i, j in enumerate(cat_idx):
        for v in range(256):
            T[j, v] = Emb[i, v] @ W_cat[i]

    b1 = bias.astype(np.float32)

    # Pack weights: magic, d, H=0, C
    out = struct.pack("<4i", 0x54424C31, d, 0, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Table family (linear with embeddings) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Format: magic 0x54424C31, int32 d, H=0, C; float32 T[d][256][C]; float32 b1[C]. */
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