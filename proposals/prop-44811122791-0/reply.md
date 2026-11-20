```hypothesis
Family: label-emb (embedding-MLP for tabular). Categorical features (8 of 14 columns) have high cardinality; one-hot in label-linear creates a wide sparse bottleneck. Learned embeddings compress categories densely, capture similarity, and feed a small ReLU MLP (H=64). This should beat label-linear accuracy (0.8524) by modeling non-linear interactions, and beat label-mlp16 speed (2325 Ir/item) by avoiding the huge one-hot input layer. Failure: dev accuracy ≤ 0.8524 or Ir/item ≥ 2325.
```
```python
"""label-emb: Embedding + 1-hidden-layer MLP for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, numpy+stdlib only.
Input: 14 bytes (6 ordinal z-scored, 8 categorical embedded).
Architecture: concat(embeddings, ordinal_z) -> Linear(H=64) -> ReLU -> Linear(C=2).
Export: magic 0x454D4231, d, n_cat, n_ord, E, H, C, cat_idx[n_cat], ord_idx[n_ord],
        emb[n_cat,256,E], ord_z[n_ord,256], W1[D,H], b1[H], W2[H,C], b2[C] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

E = 8          # embedding dim per categorical feature
H = 64         # hidden units
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 128
SAMPLES = 600000
MAX_EPOCHS = 50


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
    d = card["input"]["bytes"]

    # Identify categorical vs ordinal column indices
    cat_idx = np.array([j for j, t in enumerate(types) if t == "categorical"], dtype=np.uint8)
    ord_idx = np.array([j for j, t in enumerate(types) if t == "ordinal"], dtype=np.uint8)
    n_cat, n_ord = len(cat_idx), len(ord_idx)
    D = n_cat * E + n_ord  # input dim to MLP

    rng = np.random.default_rng(SEED)

    # Ordinal: fixed z-score lookup tables (not learned)
    ord_z = np.zeros((n_ord, 256), dtype=np.float32)
    for i, col in enumerate(ord_idx):
        v = X[:, col].astype(np.float64)
        mu, sd = v.mean(), v.std() or 1.0
        ord_z[i] = ((np.arange(256) - mu) / sd).astype(np.float32)

    # Categorical: learned embeddings [n_cat, 256, E]
    emb = (rng.uniform(-0.01, 0.01, (n_cat, 256, E))).astype(np.float32)

    # MLP weights
    W1 = (rng.uniform(-1, 1, (D, H)) * math.sqrt(6.0 / D)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Build input batch: [cat_embeddings..., ord_z...]
            z = np.empty((bs, D), dtype=np.float32)
            off = 0
            for i, col in enumerate(cat_idx):
                z[:, off:off+E] = emb[i, xb[:, col]]
                off += E
            for i, col in enumerate(ord_idx):
                z[:, off] = ord_z[i, xb[:, col]]
                off += 1

            # Forward
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs  # dL/dlogits

            # Backward
            dW2 = h.T @ probs + L2 * W2
            db2 = probs.sum(0)
            dh = probs @ W2.T
            da = dh * (a > 0)
            dW1 = z.T @ da + L2 * W1
            db1 = da.sum(0)
            dz = da @ W1.T  # (bs, D)

            # Gradient for embeddings (scatter-add)
            demb = np.zeros_like(emb)
            off = 0
            for i, col in enumerate(cat_idx):
                cat_vals = xb[:, col]  # (bs,)
                grad_slice = dz[:, off:off+E]  # (bs, E)
                # Accumulate: for each sample, add grad_slice[s] to demb[i, cat_vals[s]]
                np.add.at(demb[i], cat_vals, grad_slice)
                off += E
            # Ordinal tables are fixed, no gradient

            grads = [demb, dW1, db1, dW2, db2]
            t = 0  # Adam step counter (approximate per-batch)
            # We'll use a simple Adam with global step
            # Note: label-linear uses per-batch t; we do same
            # But we need persistent t across batches. Let's track globally.
            # Actually, label-linear increments t per batch. We'll do the same.
            # We'll move t outside the loop.
            pass  # placeholder, will restructure

    # The above loop structure needs t to persist. Let's rewrite training loop properly.
    # We'll redo the training section below.

    # --- REWRITTEN TRAINING LOOP ---
    rng = np.random.default_rng(SEED)
    emb = (rng.uniform(-0.01, 0.01, (n_cat, 256, E))).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (D, H)) * math.sqrt(6.0 / D)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            z = np.empty((bs, D), dtype=np.float32)
            off = 0
            for i, col in enumerate(cat_idx):
                z[:, off:off+E] = emb[i, xb[:, col]]
                off += E
            for i, col in enumerate(ord_idx):
                z[:, off] = ord_z[i, xb[:, col]]
                off += 1

            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2

            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            dW2 = h.T @ probs + L2 * W2
            db2 = probs.sum(0)
            dh = probs @ W2.T
            da = dh * (a > 0)
            dW1 = z.T @ da + L2 * W1
            db1 = da.sum(0)
            dz = da @ W1.T

            demb = np.zeros_like(emb)
            off = 0
            for i, col in enumerate(cat_idx):
                cat_vals = xb[:, col]
                grad_slice = dz[:, off:off+E]
                np.add.at(demb[i], cat_vals, grad_slice)
                off += E

            grads = [demb, dW1, db1, dW2, db2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights
    magic = 0x454D4231  # 'EMB1'
    header = struct.pack("<7I", magic, d, n_cat, n_ord, E, H, C)
    idx_bytes = cat_idx.tobytes() + ord_idx.tobytes()
    # All float32 little-endian
    out = header + idx_bytes
    out += emb.astype("<f4").tobytes()
    out += ord_z.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (14). Output: 1 byte class index.
 * Weight format: see train.py. */
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
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 28) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t *hd = (uint32_t *)w;
    if (hd[0] != 0x454D4231) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], n_cat = hd[2], n_ord = hd[3], E = hd[4], H = hd[5], C = hd[6];
    if (d != 14 || n_cat != 8 || n_ord != 6 || E != 8 || H != 64 || C != 2) {
        fprintf(stderr, "unexpected dims\n"); return 3;
    }
    int D = n_cat * E + n_ord;  // 70
    size_t idx_off = 28;
    if (nw < idx_off + n_cat + n_ord) { fprintf(stderr, "truncated indices\n"); return 3; }
    unsigned char *cat_idx = w + idx_off;
    unsigned char *ord_idx = w + idx_off + n_cat;
    size_t ptr = idx_off + n_cat + n_ord;

    float *emb = (float *)(w + ptr); ptr += (size_t)n_cat * 256 * E * sizeof(float);
    float *ord_z = (float *)(w + ptr); ptr += (size_t)n_ord * 256 * sizeof(float);
    float *W1  = (float *)(w + ptr); ptr += (size_t)D * H * sizeof(float);
    float *b1  = (float *)(w + ptr); ptr += (size_t)H * sizeof(float);
    float *W2  = (float *)(w + ptr); ptr += (size_t)H * C * sizeof(float);
    float *b2  = (float *)(w + ptr); ptr += (size_t)C * sizeof(float);
    if (ptr != (size_t)nw) { fprintf(stderr, "weight size mismatch\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * D);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build input vector z
        int off = 0;
        for (int ci = 0; ci < n_cat; ci++) {
            int col = cat_idx[ci];
            int cat_val = xi[col];
            const float *e = emb + ((size_t)ci * 256 + cat_val) * E;
            memcpy(z + off, e, E * sizeof(float));
            off += E;
        }
        for (int oi = 0; oi < n_ord; oi++) {
            int col = ord_idx[oi];
            int bin = xi[col];
            z[off++] = ord_z[oi * 256 + bin];
        }

        // Hidden layer: h = ReLU(z @ W1 + b1)
        for (int j = 0; j < H; j++) {
            float sum = b1[j];
            for (int k = 0; k < D; k++) sum += z[k] * W1[(size_t)k * H + j];
            h[j] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: logits = h @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int j = 0; j < H; j++) sum += h[j] * W2[(size_t)j * C + c];
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(z); free(h); free(logits);
    return 0;
}
```