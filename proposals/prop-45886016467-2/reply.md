```hypothesis
We propose a two-layer MLP with hidden size 64 (label-mlp64) using the table-family architecture. The linear model (H=0) has 266k parameters but no non-linearity; adding a ReLU layer of equal width provides non-linear interactions at similar parameter count. Early stopping on the tuning split (last 20% of train) prevents overfitting and selects the optimal epoch count before final training on all data. If dev accuracy does not exceed 0.34, the ReLU bottleneck may be insufficient for this sequence task.
```

```python
"""Table family with one ReLU hidden layer (H=64) and early stopping on the tuning split.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Uses the last 20% of train.in (file order) as tuning set
to choose the number of epochs, then retrains on the full training set."""
import json, math, struct, sys

import numpy as np

H = 64
SEED, LR, L2, BATCH, MAX_EPOCHS, PATIENCE = 1, 3e-3, 1e-4, 64, 50, 5


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoding(X, types):
    """Return list of encoding matrices E_j (256 x dim_j) for each position."""
    enc = []
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def encode(X, enc):
    """Apply encoding matrices to X (n x d) -> Z (n x k)."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def forward(Z, params, H):
    """Return logits (n x C) and hidden pre-activations (for backward)."""
    if H:
        W1, b1, W2, b2 = params
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
        return logits, (a, h)
    else:
        W1, b1 = params
        logits = Z @ W1 + b1
        return logits, (logits,)


def loss_and_grad(Z, y, params, H, L2):
    """Cross-entropy loss and gradients for all params."""
    n = Z.shape[0]
    if H:
        W1, b1, W2, b2 = params
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
        # softmax + cross-entropy gradient
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(n), y] -= 1.0
        probs /= n
        # gradients
        gW2 = h.T @ probs + L2 * W2
        gb2 = probs.sum(axis=0)
        gh = probs @ W2.T
        gh[a <= 0] = 0.0
        gW1 = Z.T @ gh + L2 * W1
        gb1 = gh.sum(axis=0)
        return (gW1, gb1, gW2, gb2)
    else:
        W1, b1 = params
        logits = Z @ W1 + b1
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(n), y] -= 1.0
        probs /= n
        gW1 = Z.T @ probs + L2 * W1
        gb1 = probs.sum(axis=0)
        return (gW1, gb1)


def accuracy(Z, y, params, H):
    logits, _ = forward(Z, params, H)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def train_loop(Z, y, params, H, rng, epochs, batch_size, lr, l2):
    """Train for given epochs, return params and list of validation accuracies (if provided)."""
    n = Z.shape[0]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            z, yy = Z[idx], y[idx]
            grads = loss_and_grad(z, yy, params, H, l2)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    d = X_all.shape[1]

    # Split: last 20% for tuning
    n_all = len(X_all)
    n_tune = n_all // 5
    n_train = n_all - n_tune
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_tune, y_tune = X_all[n_train:], y_all[n_train:]

    rng = np.random.default_rng(SEED)

    # Build encoding on train split only
    enc = build_encoding(X_train, types)
    Z_train = encode(X_train, enc)
    Z_tune = encode(X_tune, enc)
    k = Z_train.shape[1]

    # Initialize parameters
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), np.float32)
        b1 = np.zeros(C, np.float32)
        params = [W1, b1]

    # Phase 1: train on train split with early stopping on tune split
    best_acc = -1.0
    best_params = None
    best_epoch = 0
    wait = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        params = train_loop(Z_train, y_train, params, H, rng, 1, BATCH, LR, L2)
        acc = accuracy(Z_tune, y_tune, params, H)
        if acc > best_acc:
            best_acc = acc
            best_params = [p.copy() for p in params]
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    # Phase 2: retrain on full data for best_epoch epochs
    enc_full = build_encoding(X_all, types)
    Z_all = encode(X_all, enc_full)
    # Re-initialize with same seed
    rng = np.random.default_rng(SEED)
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), np.float32)
        b1 = np.zeros(C, np.float32)
        params = [W1, b1]

    params = train_loop(Z_all, y_all, params, H, rng, best_epoch, BATCH, LR, L2)

    # Export weights in table-family format (magic 0x54424C31)
    off, T = 0, []
    for E in enc_full:
        T.append((E.astype(np.float32) @ params[0][off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    W = H if H else C
    out = struct.pack("<4i", 0x54424C31, d, H, C) + np.stack(T).astype("<f4").tobytes() + params[1].astype("<f4").tobytes()
    if H:
        out += params[2].astype("<f4").tobytes() + params[3].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
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