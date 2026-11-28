```hypothesis
Family: label-emb-mlp. Learned embeddings for categorical features (size 4) plus z-scored ordinal features fed into a one-hidden-layer ReLU MLP (16 units). Embeddings capture category similarity better than one-hot linear, boosting accuracy over label-linear (0.8524) while keeping inference cheap (target <2000 Ir/item). Failure: dev accuracy ≤ 0.8524.
```

```python
"""label-emb-mlp: embeddings for categoricals + small MLP.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
E = 4          # embedding size per categorical feature
H = 16         # hidden units
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
SAMPLES = 600000  # max samples per epoch for epoch count

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
    types = card["input"]["types"]

    rng = np.random.default_rng(SEED)

    # Identify ordinal and categorical columns
    ord_idx = [i for i, t in enumerate(types) if t == "ordinal"]
    cat_idx = [i for i, t in enumerate(types) if t == "categorical"]
    n_ord = len(ord_idx)
    n_cat = len(cat_idx)
    input_dim = n_ord + n_cat * E

    # Normalization for ordinal features (compute on full train)
    ord_mean = X[:, ord_idx].mean(axis=0).astype(np.float32)
    ord_std = X[:, ord_idx].std(axis=0).astype(np.float32)
    ord_std[ord_std == 0] = 1.0

    # Embedding matrices for categorical features: (n_cat, 256, E)
    emb = (rng.uniform(-1, 1, (n_cat, 256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)

    # MLP weights
    W1 = (rng.uniform(-1, 1, (input_dim, H)) * math.sqrt(6.0 / input_dim)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    def forward(Xb):
        # Xb: (batch, d) uint8
        batch = Xb.shape[0]
        # ordinal part
        z_ord = ((Xb[:, ord_idx].astype(np.float32) - ord_mean) / ord_std)  # (batch, n_ord)
        # categorical embeddings
        z_cat = np.zeros((batch, n_cat * E), dtype=np.float32)
        for ci, col in enumerate(cat_idx):
            z_cat[:, ci*E:(ci+1)*E] = emb[ci, Xb[:, col]]
        z = np.concatenate([z_ord, z_cat], axis=1)  # (batch, input_dim)
        a = z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
        return z, a, h, logits

    def softmax_grad(logits, yb):
        # logits: (batch, C), yb: (batch,)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(yb)), yb] -= 1.0
        return (probs / len(yb)).astype(np.float32)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            z, a, h, logits = forward(Xb)
            g = softmax_grad(logits, yb)  # (batch, C)

            # gradients
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(axis=0)
            gh = g @ W2.T * (a > 0)  # (batch, H)
            grad_W1 = z.T @ gh + L2 * W1
            grad_b1 = gh.sum(axis=0)
            gz = gh @ W1.T  # (batch, input_dim)

            # embedding gradients
            grad_emb = np.zeros_like(emb)
            for ci, col in enumerate(cat_idx):
                np.add.at(grad_emb[ci], Xb[:, col], gz[:, ci*E:(ci+1)*E])
            grad_emb += L2 * emb  # L2 on embeddings

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
    # Magic: 0x454D4231 ('EMB1')
    # int32: magic, d, E, H, C
    # For each feature (d total):
    #   if ordinal: float32 mean, float32 std
    #   if categorical: float32[256*E] embedding matrix (row-major)
    # Then MLP: W1[input_dim*H], b1[H], W2[H*C], b2[C]
    out = struct.pack("<5i", 0x454D4231, d, E, H, C)
    for j in range(d):
        if types[j] == "ordinal":
            idx = ord_idx.index(j)
            out += struct.pack("<2f", ord_mean[idx], ord_std[idx])
        else:
            ci = cat_idx.index(j)
            out += emb[ci].astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index (0 or 1). */
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
    if (nw < 20 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C != 2 || d != 14 || E != 4 || H != 16) { fprintf(stderr, "unexpected dims\n"); return 3; }

    const float *ptr = (const float *)(hd + 5);
    float ord_mean[6], ord_std[6];
    float emb[8][256][4];
    int ord_idx[6] = {0,2,4,10,11,12};
    int cat_idx[8] = {1,3,5,6,7,8,9,13};
    int o = 0, c = 0;
    for (int j = 0; j < d; j++) {
        if (j == ord_idx[o]) {
            ord_mean[o] = *ptr++; ord_std[o] = *ptr++; o++;
        } else {
            for (int v = 0; v < 256; v++)
                for (int e = 0; e < E; e++)
                    emb[c][v][e] = *ptr++;
            c++;
        }
    }
    int input_dim = 6 + 8 * E;  // 6 + 32 = 38
    const float *W1 = ptr; ptr += input_dim * H;
    const float *b1 = ptr; ptr += H;
    const float *W2 = ptr; ptr += H * C;
    const float *b2 = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * input_dim);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build input vector z
        int zi = 0;
        for (int oi = 0; oi < 6; oi++) {
            float val = (float)xi[ord_idx[oi]];
            z[zi++] = (val - ord_mean[oi]) / ord_std[oi];
        }
        for (int ci = 0; ci < 8; ci++) {
            const float *e = emb[ci][xi[cat_idx[ci]]];
            for (int e_i = 0; e_i < E; e_i++) z[zi++] = e[e_i];
        }

        // Hidden layer: h = ReLU(z @ W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < input_dim; v++) sum += z[v] * W1[v * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output logits
        for (int cl = 0; cl < C; cl++) {
            float sum = b2[cl];
            for (int u = 0; u < H; u++) sum += h[u] * W2[u * C + cl];
            logits[cl] = sum;
        }

        // Argmax
        unsigned char best = logits[1] > logits[0] ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```