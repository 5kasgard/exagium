```hypothesis
We propose "label-embedding-linear": learn dense embeddings for the 6 ordinal (binned numeric) features while keeping 8 categorical features as one-hot, followed by a linear classifier. The ordinal bins likely have non-linear relationships with income; embeddings capture this better than z-scoring. Inference remains a single linear layer with precomputed per-feature lookup tables (same format as label-linear), so cost per item stays low. If dev accuracy does not exceed label-linear's 0.8524, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (same as label-linear)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
CANDIDATE_EMB_DIMS = [2, 4, 8, 16]  # tried on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sg(scores, y):
    """Softmax cross-entropy gradient."""
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return (probs / len(y)).astype(np.float32)


def train_model(X, y, emb_dim, ordinal_indices, cat_onehots, seed, epochs, batch_size):
    n, d = X.shape
    C = 2
    rng = np.random.default_rng(seed)

    # Build representation matrices: learned embeddings for ordinal, fixed one-hot for categorical
    embeddings = []          # only for ordinal features, in order of ordinal_indices
    rep_dims = []            # dimension of each feature's representation
    for j in range(d):
        if j in ordinal_indices:
            scale = math.sqrt(6.0 / emb_dim)
            Emb = rng.uniform(-scale, scale, (256, emb_dim)).astype(np.float32)
            embeddings.append(Emb)
            rep_dims.append(emb_dim)
        else:
            embeddings.append(None)
            rep_dims.append(cat_onehots[j].shape[1])

    total_dim = sum(rep_dims)
    scale = math.sqrt(6.0 / total_dim)
    W_final = rng.uniform(-scale, scale, (total_dim, C)).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state for [W_final, b] + each ordinal embedding
    params = [W_final, b] + [Emb for Emb in embeddings if Emb is not None]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Offsets of each feature in the concatenated vector
    offsets = []
    off = 0
    for dim in rep_dims:
        offsets.append(off)
        off += dim

    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            X_batch = X[idx]
            y_batch = y[idx]
            B = len(idx)

            # Forward: build concatenated representation
            Z_parts = []
            for j in range(d):
                if j in ordinal_indices:
                    Emb = embeddings[ordinal_indices.index(j)]
                    Z_parts.append(Emb[X_batch[:, j]])
                else:
                    Z_parts.append(cat_onehots[j][X_batch[:, j]])
            Z_batch = np.concatenate(Z_parts, axis=1)  # (B, total_dim)

            scores = Z_batch @ W_final + b
            g = sg(scores, y_batch)                     # (B, C)

            dW_final = Z_batch.T @ g + L2 * W_final
            db = g.sum(axis=0)
            dZ_batch = g @ W_final.T                    # (B, total_dim)

            # Gradients for ordinal embeddings
            grad_embeddings = []
            for j in range(d):
                if j in ordinal_indices:
                    dim_j = rep_dims[j]
                    off_j = offsets[j]
                    grad_slice = dZ_batch[:, off_j:off_j + dim_j]   # (B, dim_j)
                    grad_Emb = np.zeros_like(embeddings[ordinal_indices.index(j)])
                    np.add.at(grad_Emb, X_batch[:, j], grad_slice)
                    grad_Emb += L2 * embeddings[ordinal_indices.index(j)]
                    grad_embeddings.append(grad_Emb)

            grads = [dW_final, db] + grad_embeddings

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    return W_final, b, embeddings


def evaluate(X, y, W_final, b, embeddings, ordinal_indices, cat_onehots):
    n = X.shape[0]
    correct = 0
    for s in range(0, n, 1024):
        idx = slice(s, min(s + 1024, n))
        X_batch = X[idx]
        y_batch = y[idx]
        Z_parts = []
        for j in range(X.shape[1]):
            if j in ordinal_indices:
                Emb = embeddings[ordinal_indices.index(j)]
                Z_parts.append(Emb[X_batch[:, j]])
            else:
                Z_parts.append(cat_onehots[j][X_batch[:, j]])
        Z_batch = np.concatenate(Z_parts, axis=1)
        scores = Z_batch @ W_final + b
        preds = scores.argmax(axis=1)
        correct += (preds == y_batch).sum()
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    C = card["classes"]

    ordinal_indices = [j for j, t in enumerate(types) if t == "ordinal"]
    cat_indices = [j for j, t in enumerate(types) if t == "categorical"]

    # One-hot matrices for categorical features (using full training data)
    cat_onehots = {}
    for j in cat_indices:
        vals = np.unique(X[:, j])
        E = np.zeros((256, len(vals)), np.float32)
        E[vals, np.arange(len(vals))] = 1.0
        cat_onehots[j] = E

    # Tuning split: last 20% in file order
    n = X.shape[0]
    split = int(0.8 * n)
    X_train_tune, X_tune = X[:split], X[split:]
    y_train_tune, y_tune = y[:split], y[split:]

    epochs_tune = min(MAX_EPOCHS, math.ceil(SAMPLES / split))

    best_emb_dim = CANDIDATE_EMB_DIMS[0]
    best_acc = -1.0
    for emb_dim in CANDIDATE_EMB_DIMS:
        W_final, b, embeddings = train_model(
            X_train_tune, y_train_tune, emb_dim, ordinal_indices, cat_onehots,
            seed=1000 + emb_dim, epochs=epochs_tune, batch_size=BATCH
        )
        acc = evaluate(X_tune, y_tune, W_final, b, embeddings, ordinal_indices, cat_onehots)
        if acc > best_acc:
            best_acc = acc
            best_emb_dim = emb_dim

    # Final training on all data
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    W_final, b, embeddings = train_model(
        X, y, best_emb_dim, ordinal_indices, cat_onehots,
        seed=SEED, epochs=epochs_full, batch_size=BATCH
    )

    # Build per-feature contribution tables T_j (256 x C) for the linear model
    rep_dims = []
    for j in range(d):
        if j in ordinal_indices:
            rep_dims.append(best_emb_dim)
        else:
            rep_dims.append(cat_onehots[j].shape[1])
    offsets = []
    off = 0
    for dim in rep_dims:
        offsets.append(off)
        off += dim

    T = []
    for j in range(d):
        if j in ordinal_indices:
            Emb = embeddings[ordinal_indices.index(j)]
            off_j = offsets[j]
            dim_j = rep_dims[j]
            T_j = Emb @ W_final[off_j:off_j + dim_j]
        else:
            off_j = offsets[j]
            dim_j = rep_dims[j]
            T_j = cat_onehots[j] @ W_final[off_j:off_j + dim_j]
        T.append(T_j.astype(np.float32))

    # Write weights.bin in label-linear format (H=0)
    out = struct.pack("<4i", 0x54424C31, d, 0, C)
    out += np.stack(T).astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
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
                for (int