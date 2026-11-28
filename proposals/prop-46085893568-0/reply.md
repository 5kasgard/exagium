```hypothesis
Family: label-embmlp (embedding MLP with folded weights).
Why: Learned embeddings for the 8 categorical features capture category similarities, and a single ReLU layer adds non-linear interactions. At inference the first layer is folded into per-feature 256×H tables, so cost matches label-mlp16 while using fewer parameters (embedding dim 8 vs one-hot). Expects to beat label-linear (0.8524) on dev accuracy.
Failure signal: dev accuracy ≤ 0.8524 or inference instructions/item > label-mlp16 (≈2300).
```
```python
"""label-embmlp: embedding MLP for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Embeds each categorical feature into E=8 dimensions, ordinal features are z-scored.
Hidden layer H=16 ReLU. Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Exports folded weights in the same format as label-linear (magic 0x54424C31)."""
import json, math, struct, sys

import numpy as np

E = 8          # embedding dimension for categorical features
H = 16         # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
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

    # Build embedding matrices and compute total input dimension D
    emb_mats = []          # list of (256, E) for categorical, (256, 1) for ordinal
    cat_indices = []       # indices of categorical features
    ord_indices = []       # indices of ordinal features
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            mat = ((np.arange(256, dtype=np.float32) - mu) / sd).reshape(256, 1)
            emb_mats.append(mat)
            ord_indices.append(j)
        else:
            # categorical: learnable embedding, initialize with Xavier
            mat = rng.uniform(-1, 1, (256, E)).astype(np.float32) * math.sqrt(6.0 / (256 + E))
            emb_mats.append(mat)
            cat_indices.append(j)

    n_cat = len(cat_indices)
    n_ord = len(ord_indices)
    D = n_ord * 1 + n_cat * E

    # Initialize MLP weights
    W1 = rng.uniform(-1, 1, (D, H)).astype(np.float32) * math.sqrt(6.0 / D)
    b1 = np.zeros(H, np.float32)
    W2 = rng.uniform(-1, 1, (H, C)).astype(np.float32) * math.sqrt(6.0 / H)
    b2 = np.zeros(C, np.float32)

    # Adam state for all parameters: [emb_mats (cat only), W1, b1, W2, b2]
    params = []
    grads = []
    states = []   # each state is (m, v)
    # embeddings for categorical features
    for j in cat_indices:
        params.append(emb_mats[j])
        grads.append(np.zeros_like(emb_mats[j]))
        states.append((np.zeros_like(emb_mats[j]), np.zeros_like(emb_mats[j])))
    # W1, b1, W2, b2
    for p in (W1, b1, W2, b2):
        params.append(p)
        grads.append(np.zeros_like(p))
        states.append((np.zeros_like(p), np.zeros_like(p)))

    t = 0
    n = X.shape[0]
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bs = len(yb)

            # Forward: build concatenated embeddings Z (bs, D)
            Z_parts = []
            for j in range(d):
                if j in cat_indices:
                    Z_parts.append(emb_mats[j][xb[:, j]])
                else:
                    Z_parts.append(emb_mats[j][xb[:, j]])
            Z = np.concatenate(Z_parts, axis=1)   # (bs, D)

            a = Z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            dlogits = probs / bs   # (bs, C)

            # Backward
            dW2 = h.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)
            dh = dlogits @ W2.T
            dh[a <= 0] = 0.0
            dW1 = Z.T @ dh + L2 * W1
            db1 = dh.sum(axis=0)
            dZ = dh @ W1.T   # (bs, D)

            # Split dZ into per-feature gradients
            dZ_parts = np.split(dZ, np.cumsum([E if j in cat_indices else 1 for j in range(d)])[:-1], axis=1)

            # Accumulate gradients
            param_idx = 0
            for j in range(d):
                if j in cat_indices:
                    # sparse update: add gradient for the rows that were used
                    np.add.at(grads[param_idx], xb[:, j], dZ_parts[j])
                    param_idx += 1
            # W1, b1, W2, b2
            grads[param_idx] += dW1; param_idx += 1
            grads[param_idx] += db1; param_idx += 1
            grads[param_idx] += dW2; param_idx += 1
            grads[param_idx] += db2; param_idx += 1

            # Adam update
            t += 1
            for i, (p, g, (m, v)) in enumerate(zip(params, grads, states)):
                m *= 0.9; m += 0.1 * g
                v *= 0.999; v += 0.001 * g * g
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
                g.fill(0.0)

    # After training, compute folded tables T_j = emb_mats[j] @ W1_slice
    T_list = []
    off = 0
    for j in range(d):
        if j in cat_indices:
            W1_slice = W1[off:off + E]
            off += E
        else:
            W1_slice = W1[off:off + 1]
            off += 1
        T_j = emb_mats[j] @ W1_slice   # (256, H)
        T_list.append(T_j.astype(np.float32))

    # Write weights file
    magic = 0x54424C31
    out = struct.pack("<4i", magic, d, H, C)
    out += np.stack(T_list).astype("<f4").tobytes()
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