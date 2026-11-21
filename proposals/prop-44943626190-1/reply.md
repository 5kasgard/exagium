```hypothesis
Propose a two-hidden-layer MLP family "label-mlp2" with per-feature ordinal embeddings (like label-linear) followed by two ReLU layers. The 8x8 optdigits patterns benefit from hierarchical feature extraction; the extra layer adds nonlinear capacity while inference remains fast (two small dense layers after summed embeddings). If dev accuracy does not exceed label-linear (0.9613) or per-item cost increases substantially, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters for grid search (tuned on last 20% of train)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 200
SAMPLES = 600000
MAX_EPOCHS = 100

# Candidate architectures: (H1, H2)
CANDIDATES = [(32, 16), (64, 32), (16, 8), (48, 24)]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_features(X, types):
    """Return list of encoding matrices E_j (256 x dim_j) and total dim."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])  # (256,1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float64)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def build_T(enc, W1):
    """Fuse encodings and first-layer weights into per-byte lookup tables T (d,256,H1)."""
    d = len(enc)
    H1 = W1.shape[1]
    T = np.zeros((d, 256, H1), np.float32)
    off = 0
    for j, E in enumerate(enc):
        dim = E.shape[1]
        T[j] = (E.astype(np.float32) @ W1[off:off+dim]).astype(np.float32)
        off += dim
    return T


def train_model(X, y, enc, H1, H2, C, rng, epochs, batch, lr, l2, verbose=False):
    """Train on (X,y) with given architecture, return parameters and final validation accuracy if provided."""
    n, d = X.shape
    # Build encoded design matrix Z (n, total_dim)
    Z_list = [E[X[:, j]] for j, E in enumerate(enc)]
    Z = np.concatenate(Z_list, axis=1).astype(np.float32)
    k = Z.shape[1]

    # Initialize weights
    W1 = (rng.uniform(-1, 1, (k, H1)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
    b3 = np.zeros(C, np.float32)

    params = [W1, b1, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            z, yy = Z[idx], y[idx]

            # Forward
            a1 = z @ W1 + b1
            h1 = np.maximum(a1, 0)
            a2 = h1 @ W2 + b2
            h2 = np.maximum(a2, 0)
            logits = h2 @ W3 + b3

            # Backward
            g3 = softmax_grad(logits, yy)
            dW3 = h2.T @ g3 + l2 * W3
            db3 = g3.sum(0)

            g2 = g3 @ W3.T
            g2[a2 <= 0] = 0
            dW2 = h1.T @ g2 + l2 * W2
            db2 = g2.sum(0)

            g1 = g2 @ W2.T
            g1[a1 <= 0] = 0
            dW1 = z.T @ g1 + l2 * W1
            db1 = g1.sum(0)

            grads = [dW1, db1, dW2, db2, dW3, db3]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    return params


def evaluate(X, y, enc, params):
    W1, b1, W2, b2, W3, b3 = params
    Z_list = [E[X[:, j]] for j, E in enumerate(enc)]
    Z = np.concatenate(Z_list, axis=1).astype(np.float32)
    a1 = Z @ W1 + b1
    h1 = np.maximum(a1, 0)
    a2 = h1 @ W2 + b2
    h2 = np.maximum(a2, 0)
    logits = h2 @ W3 + b3
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    n = X.shape[0]
    split = int(0.8 * n)

    # Tuning split (last 20%)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # Compute encodings on the training portion (for grid search)
    enc_train = encode_features(X_train, types)

    best_acc = -1
    best_params = None
    best_H1 = best_H2 = None
    base_rng = np.random.default_rng(SEED)

    for H1, H2 in CANDIDATES:
        rng = np.random.default_rng(base_rng.integers(2**32))
        params = train_model(X_train, y_train, enc_train, H1, H2, C, rng,
                             epochs=min(MAX_EPOCHS, math.ceil(SAMPLES / split)),
                             batch=BATCH, lr=LR, l2=L2)
        # Need encodings for validation based on training statistics
        # Recompute encodings on full training set? Actually we should use the same encodings (from X_train) for validation.
        # But validation set should be encoded with training statistics. We already have enc_train.
        acc = evaluate(X_val, y_val, enc_train, params)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_H1, best_H2 = H1, H2

    # Retrain on full dataset with best architecture
    print(f"Selected H1={best_H1}, H2={best_H2}, val_acc={best_acc:.4f}", file=sys.stderr)
    enc_full = encode_features(X, types)
    rng = np.random.default_rng(SEED)
    final_params = train_model(X, y, enc_full, best_H1, best_H2, C, rng,
                               epochs=min(MAX_EPOCHS, math.ceil(SAMPLES / n)),
                               batch=BATCH, lr=LR, l2=L2)

    W1, b1, W2, b2, W3, b3 = final_params
    T = build_T(enc_full, W1)

    # Write weights.bin
    magic = 0x4D4C5032  # 'MLP2'
    d = X.shape[1]
    H1, H2 = best_H1, best_H2
    header = struct.pack("<4i", magic, d, H1, H2, C)
    # Order: T (d*256*H1), b1 (H1), W2 (H1*H2), b2 (H2), W3 (H2*C), b3 (C)
    body = (T.astype("<f4").tobytes() +
            b1.astype("<f4").tobytes() +
            W2.astype("<f4").tobytes() +
            b2.astype("<f4").tobytes() +
            W3.astype("<f4").tobytes() +
            b3.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)


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
    if (nw < 20 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256 || H1 < 1 || H2 < 1) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20L
        + 4L * d * 256 * H1
        + 4L * H1
        + 4L * H1 * H2
        + 4L * H2
        + 4L * H2 * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld != %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * H1;
    const float *W2 = b1 + H1;
    const float *b2 = W2 + (long)H1 * H2;
    const float *W3 = b2 + H2;
    const float *b3 = W3 + (long)H2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc1 = malloc(sizeof(float) * H1);
    float *h1 = malloc(sizeof(float) * H1);
    float *acc2 = malloc(sizeof(float) * H2);
    float *h2 = malloc(sizeof(float) * H2);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Layer 1: acc1 = b1 + sum_j T[j, xi[j], :]
        memcpy(acc1, b1, H1 * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H1;
            for (int u = 0; u < H1; u++) acc1[u] += t[u];
        }
        // ReLU
        for (int u = 0; u < H1; u++) h1[u] = acc1[u] > 0.0f ? acc1[u] : 0.0f;

        // Layer 2: acc2 = b2 + h1 @ W2
        memcpy(acc2, b2, H2 * sizeof(float));
        for (int u = 0; u < H1; u++) {
            float hu = h1[u];
            if (hu == 0.0f) continue;
            const float *w = W2 + (long)u * H2;
            for (int v = 0; v < H2; v++) acc2[v] += hu * w[v];
        }
        // ReLU
        for (int v = 0; v < H2; v++) h2[v] = acc2[v] > 0.0f ? acc2[v] : 0.0f;

        // Output layer: scores = b3 + h2 @ W3
        memcpy(scores, b3, C * sizeof(float));
        for (int v = 0; v < H2; v++) {
            float hv = h2[v];
            if (hv == 0.0f) continue;
            const float *w = W3 + (long)v * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```