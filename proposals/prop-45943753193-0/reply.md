```hypothesis
Family: label-embed. A shallow neural network with learned embeddings for categorical features (size 8) and z-scored ordinal features, followed by a 32-unit ReLU hidden layer. The embeddings capture non-linear interactions among categorical variables better than one-hot linear models, while the fixed per-value lookup tables (same export format as label-linear) keep inference fast. If dev accuracy does not exceed label-linear (0.8524) by ≥0.5 pp, the approach fails.
```

```python
"""label-embed: entity embeddings for categorical, z-score for ordinal, one ReLU layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports per-byte lookup tables in label-linear format (magic 0x54424C31)."""
import json, math, struct, sys

import numpy as np

E = 8          # embedding size for each categorical feature
H = 32         # hidden layer size
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

    rng = np.random.default_rng(SEED)

    # Compute ordinal statistics on full training set
    ordinal_mean = {}
    ordinal_std = {}
    cat_features = []
    ord_features = []
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            ordinal_mean[j] = mu
            ordinal_std[j] = sd
            ord_features.append(j)
        else:
            cat_features.append(j)

    # Build fixed encoding matrices for ordinal (256x1) and learnable embeddings for categorical (256xE)
    enc_ord = {}
    for j in ord_features:
        mu, sd = ordinal_mean[j], ordinal_std[j]
        enc_ord[j] = ((np.arange(256) - mu) / sd).astype(np.float32)[:, None]  # (256,1)

    # Learnable embeddings for categorical
    emb = {}
    for j in cat_features:
        emb[j] = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / E)).astype(np.float32)

    # First layer weights: input_dim = len(ord_features)*1 + len(cat_features)*E
    input_dim = len(ord_features) + len(cat_features) * E
    W1 = (rng.uniform(-1, 1, (input_dim, H)) * math.sqrt(6.0 / input_dim)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [W1, b1, W2, b2]
    # embeddings are separate; we'll update them via gradient
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    emb_state = {j: (np.zeros_like(emb[j]), np.zeros_like(emb[j])) for j in cat_features}

    n = X.shape[0]
    t = 0

    def forward(batch_idx):
        """Return hidden activations (batch, H) and cache for backward."""
        batch = X[batch_idx]
        # Build concatenated input
        parts = []
        for j in range(d):
            if j in ord_features:
                parts.append(enc_ord[j][batch[:, j]])          # (batch,1)
            else:
                parts.append(emb[j][batch[:, j]])              # (batch,E)
        Z = np.concatenate(parts, axis=1).astype(np.float32)   # (batch, input_dim)
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        return Z, a, h

    def softmax_grad(h, yy):
        S = h @ W2 + b2
        S -= S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)   # (batch, C)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            yy = y[idx]
            Z, a, h = forward(idx)
            g = softmax_grad(h, yy)                 # (batch, C)

            # Gradients for output layer
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(0)

            # Backprop to hidden
            gh = g @ W2.T * (a > 0)                 # (batch, H)

            # Gradients for first layer
            grad_W1 = Z.T @ gh + L2 * W1
            grad_b1 = gh.sum(0)

            # Gradients for embeddings (categorical features)
            emb_grads = {}
            offset = 0
            for j in range(d):
                if j in ord_features:
                    offset += 1
                else:
                    # slice of W1 corresponding to this embedding
                    w_slice = W1[offset:offset + E]          # (E, H)
                    # gradient for each example: gh_example (H,) @ w_slice.T (H,E) -> (E,)
                    # accumulate over batch
                    # We can do: for each example, emb[batch[i, j]] += gh[i] @ w_slice.T
                    # Vectorized: use np.add.at
                    grad_emb = np.zeros_like(emb[j])
                    np.add.at(grad_emb, batch[:, j], gh @ w_slice.T)
                    grad_emb += L2 * emb[j]
                    emb_grads[j] = grad_emb
                    offset += E

            # Adam updates
            t += 1
            grads = [grad_W1, grad_b1, grad_W2, grad_b2]
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

            for j in cat_features:
                m, v = emb_state[j]
                gr = emb_grads[j]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                emb[j] -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # After training, compute per-byte lookup tables T for export (same format as label-linear)
    W = H
    T_list = []
    offset = 0
    for j in range(d):
        if j in ord_features:
            # contribution: enc_ord[j] (256,1) @ W1[offset:offset+1] (1,H) -> (256,H)
            Tj = enc_ord[j] @ W1[offset:offset + 1]
            offset += 1
        else:
            # emb[j] (256,E) @ W1[offset:offset+E] (E,H) -> (256,H)
            Tj = emb[j] @ W1[offset:offset + E]
            offset += E
        T_list.append(Tj.astype(np.float32))

    T_stack = np.stack(T_list)  # (d, 256, H)
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += T_stack.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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