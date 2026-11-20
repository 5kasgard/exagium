```hypothesis
We propose "label-emb-linear", a factorized linear model that learns a 16-dimensional embedding for each of the 16 context positions, followed by a linear output layer. The embeddings capture similarity between characters (e.g., vowels cluster together), providing regularization that should improve generalization over one-hot linear regression. The model is trained with Adam on the first 80% of data, hyperparameters (embedding dimension, learning rate) are selected on the last 20% tuning split, then retrained on all data. The final weights are collapsed into a standard linear lookup table (per-position per-byte logits) so inference matches label-linear's speed. If dev accuracy does not exceed 0.34, the factorization offers no benefit for this task.
```
```python
"""label-emb-linear: factorized linear model with per-position byte embeddings.
Train: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports weights in label-linear format (magic 0x54424C31, d, H=0, C, T, b)."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid for tuning
EMB_DIMS = [8, 16, 32]
LRS = [1e-3, 3e-3]
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
SEED = 1


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, emb_dim, lr, epochs, rng):
    """Train factorized model on (X, y). Returns embeddings list, W, b."""
    n, d = X.shape
    C = 65
    # Initialize parameters
    embeddings = [rng.uniform(-0.1, 0.1, (256, emb_dim)).astype(np.float32) for _ in range(d)]
    W = rng.uniform(-0.1, 0.1, (d * emb_dim, C)).astype(np.float32)
    b = np.zeros(C, np.float32)
    # Adam state
    params = embeddings + [W, b]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward: gather embeddings
            e = np.empty((B, d * emb_dim), np.float32)
            for j in range(d):
                e[:, j*emb_dim:(j+1)*emb_dim] = embeddings[j][xb[:, j]]

            logits = e @ W + b                     # (B, C)
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # Loss gradient
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            # Gradients for W and b
            grad_W = e.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)

            # Gradient for embeddings
            grad_e = grad_logits @ W.T             # (B, d*emb_dim)
            grad_embeddings = []
            for j in range(d):
                gj = np.zeros((256, emb_dim), np.float32)
                np.add.at(gj, xb[:, j], grad_e[:, j*emb_dim:(j+1)*emb_dim])
                gj += L2 * embeddings[j]
                grad_embeddings.append(gj)

            # Adam update
            t += 1
            all_grads = grad_embeddings + [grad_W, grad_b]
            for i, (p, gr) in enumerate(zip(params, all_grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    return embeddings, W, b


def evaluate(X, y, embeddings, W, b):
    n, d = X.shape
    emb_dim = embeddings[0].shape[1]
    e = np.empty((n, d * emb_dim), np.float32)
    for j in range(d):
        e[:, j*emb_dim:(j+1)*emb_dim] = embeddings[j][X[:, j]]
    logits = e @ W + b
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]

    n = len(X)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    # If tuning split is too small, skip tuning and use default config
    if len(X_tune) < 100:
        best_emb, best_lr = 16, 3e-3
    else:
        best_acc = -1.0
        best_emb, best_lr = EMB_DIMS[0], LRS[0]
        for emb in EMB_DIMS:
            for lr in LRS:
                rng = np.random.default_rng(SEED + emb * 100 + int(lr * 10000))
                epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
                embs, W, b = train_model(X_train, y_train, emb, lr, epochs, rng)
                acc = evaluate(X_tune, y_tune, embs, W, b)
                if acc > best_acc:
                    best_acc, best_emb, best_lr = acc, emb, lr

    # Retrain on full data with best config
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    embeddings, W, b = train_model(X, y, best_emb, best_lr, epochs, rng)

    # Collapse to linear lookup table T[d][256][C]
    T = []
    for j in range(d):
        T.append(embeddings[j] @ W[j*best_emb:(j+1)*best_emb])
    T = np.stack(T).astype("<f4")   # (d, 256, C)
    b = b.astype("<f4")

    # Write weights in label-linear format (H=0)
    out = struct.pack("<4i", 0x54424C31, d, 0, C) + T.tobytes() + b.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Table family (linear) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). */
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