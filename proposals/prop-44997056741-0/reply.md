```hypothesis
Family: label-emb-linear. Learned embeddings for categorical features (size E) plus z-scored ordinal features, fed to a linear classifier. This reduces dimensionality vs one-hot, captures categorical similarities, and keeps inference extremely fast (single matvec). Hyperparameters (E ∈ {4,8,16}) chosen on the last 20% tuning split. If dev accuracy ≤ 0.8524 (label-linear) the embedding advantage does not materialize for this task.
```

```python
import json, math, struct, sys

import numpy as np

# hyperparameter grid searched on tuning split (last 20% of train.in)
EMBED_SIZES = [4, 8, 16]
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x454D424C  # "L B M E" little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_model(card, X, y, E, epochs):
    """Train embeddings + linear layer, return (T, W, b, D)."""
    d = card["input"]["bytes"]
    C = card["classes"]
    types = card["input"]["types"]
    rng = np.random.default_rng(SEED)

    # identify ordinal / categorical features and their output dimensions
    ordinal_info = []   # (col, mu, sd, dim_index)
    cat_info = []       # (col, num_cats, emb_matrix, offset)
    dim = 0
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            ordinal_info.append((j, mu, sd, dim))
            dim += 1
        else:
            # category indices 0..max_val, 255 = unseen
            max_val = int(X[:, j].max())
            num_cats = max_val + 1          # 0..max_val inclusive
            emb = (rng.uniform(-1, 1, (num_cats + 1, E)) * math.sqrt(6.0 / (num_cats + 1))).astype(np.float32)
            cat_info.append((j, num_cats, emb, dim))
            dim += E
    D = dim

    # linear layer
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [ci[2] for ci in cat_info] + [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    steps_per_epoch = (n + BATCH - 1) // BATCH
    total_steps = min(MAX_EPOCHS * steps_per_epoch, math.ceil(SAMPLES / BATCH))

    def softmax_grad(logits, yy):
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits, dtype=np.float32)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    t = 0
    for _ in range(MAX_EPOCHS):
        if t >= total_steps:
            break
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            if t >= total_steps:
                break
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = xb.shape[0]

            # build batch feature matrix z (bs, D)
            z = np.zeros((bs, D), np.float32)
            for col, mu, sd, d_idx in ordinal_info:
                z[:, d_idx] = (xb[:, col].astype(np.float32) - mu) / sd
            for col, num_cats, emb, offset in cat_info:
                # clip 255 to unseen index (num_cats)
                cat_idx = xb[:, col].astype(np.int32)
                cat_idx = np.where(cat_idx > num_cats, num_cats, cat_idx)
                z[:, offset:offset + E] = emb[cat_idx]

            logits = z @ W + b
            g = softmax_grad(logits, yb)                 # (bs, C)
            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(axis=0)
            grad_z = g @ W.T                             # (bs, D)

            # gradients for embeddings
            grad_embs = []
            for col, num_cats, emb, offset in cat_info:
                ge = np.zeros_like(emb)
                cat_idx = xb[:, col].astype(np.int32)
                cat_idx = np.where(cat_idx > num_cats, num_cats, cat_idx)
                np.add.at(ge, cat_idx, grad_z[:, offset:offset + E])
                ge += L2 * emb
                grad_embs.append(ge)

            # Adam update
            all_grads = grad_embs + [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, all_grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # build lookup tables T[j][v] -> D-vector
    T = np.zeros((d, 256, D), np.float32)
    for col, mu, sd, d_idx in ordinal_info:
        for v in range(256):
            T[col, v, d_idx] = (v - mu) / sd
    for col, num_cats, emb, offset in cat_info:
        for v in range(256):
            idx = v if v <= num_cats else num_cats
            T[col, v, offset:offset + E] = emb[idx]

    return T, W, b, D


def evaluate(card, X, y, T, W, b, D):
    n = X.shape[0]
    C = card["classes"]
    z = np.zeros((n, D), np.float32)
    for j in range(card["input"]["bytes"]):
        z += T[j, X[:, j]]
    logits = z @ W + b
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)

    # split: last 20% for tuning
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    best_acc, best_E, best_model = -1, None, None
    for E in EMBED_SIZES:
        T, W, b, D = build_model(card, X_tr, y_tr, E, epochs=10)  # quick eval
        acc = evaluate(card, X_val, y_val, T, W, b, D)
        if acc > best_acc:
            best_acc, best_E, best_model = acc, E, (T, W, b, D)
        print(f"E={E} val_acc={acc:.4f}", file=sys.stderr)

    # retrain on full data with best_E
    print(f"best E={best_E} val_acc={best_acc:.4f}", file=sys.stderr)
    T, W, b, D = build_model(card, X, y, best_E, epochs=MAX_EPOCHS)

    # write weights.bin
    d = card["input"]["bytes"]
    C = card["classes"]
    out = struct.pack("<4i", MAGIC, d, D, C)
    out += T.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 16L + 4L * d * 256 * D + 4L * D * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *W = T + (long)d * 256 * D;
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(D * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(acc, 0, D * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * D;
            for (int k = 0; k < D; k++) acc[k] += t[k];
        }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int k = 0; k < D; k++) sum += acc[k] * W[(long)k * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```