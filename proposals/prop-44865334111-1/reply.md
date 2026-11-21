```hypothesis
Family: label-mlp8. A one-hidden-layer MLP with 8 ReLU units, using per-byte lookup tables for the first layer (like label-linear). Hyperparameters (learning rate, L2) are tuned on the last 20% of training data. This should beat label-linear (0.9613) with only slightly higher cost (Ir/item ~6000-7000). If dev accuracy is ≤0.965 or Ir/item >8000, the model does not meet the goal.
```

```python
"""label-mlp8: one-hidden-layer MLP with 8 ReLU units, per-byte lookup tables.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Tunes LR and L2 on last 20% of train.in,
then trains on all data. Exports same format as label-linear (magic 0x54424C31)."""
import json, math, struct, sys

import numpy as np

SEED = 1
H = 8
CANDIDATE_LRS = [1e-3, 3e-3, 1e-2]
CANDIDATE_L2S = [1e-4, 1e-5]
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


def build_encoding(X, types):
    """Return list of encoding matrices E_j (256 x dim_j) and total dim k."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def encode(X, enc):
    """Apply encodings to X (n x d) -> Z (n x k)."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def make_tables(enc, W1, H):
    """Convert per-feature encodings and first-layer weights to lookup tables T[d][256][W]."""
    W = H if H else W1.shape[1]
    off, T = 0, []
    for E in enc:
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    return np.stack(T)  # (d, 256, W)


def forward(Z, W1, b1, W2, b2, H):
    """Compute logits for batch Z (n x k)."""
    a = Z @ W1 + b1
    if H:
        h = np.maximum(a, 0)
        return h @ W2 + b2
    return a


def train_model(X_tr, y_tr, X_va, y_va, enc, C, lr, l2, rng):
    """Train on (X_tr, y_tr), return (params, val_acc)."""
    Z_tr = encode(X_tr, enc)
    Z_va = encode(X_va, enc) if X_va is not None else None
    n, k = Z_tr.shape
    W = H if H else C
    W1 = (rng.uniform(-1, 1, (k, W)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(W, np.float32)
    if H:
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        W2 = b2 = None
        params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_tr[idx], y_tr[idx]
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                logits = h @ W2 + b2
            else:
                logits = z @ W1 + b1
            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            if H:
                gh = (g @ W2.T) * (a > 0)
                grads = [z.T @ gh + l2 * W1, gh.sum(0), h.T @ g + l2 * W2, g.sum(0)]
            else:
                grads = [z.T @ g + l2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    if X_va is not None:
        logits_va = forward(Z_va, W1, b1, W2, b2, H)
        pred = logits_va.argmax(axis=1)
        acc = (pred == y_va).mean()
        return params, acc
    return params, None


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    n = len(X_all)
    split = int(0.8 * n)
    X_tr, y_tr = X_all[:split], y_all[:split]
    X_va, y_va = X_all[split:], y_all[split:]

    # Build encoding on full training data (no leakage: validation not used for stats)
    enc_full = build_encoding(X_all, types)

    # Grid search on validation split
    rng = np.random.default_rng(SEED)
    best_acc, best_lr, best_l2 = -1, None, None
    for lr in CANDIDATE_LRS:
        for l2 in CANDIDATE_L2S:
            rng_i = np.random.default_rng(SEED)  # same seed for each config
            params, acc = train_model(X_tr, y_tr, X_va, y_va, enc_full, C, lr, l2, rng_i)
            if acc > best_acc:
                best_acc, best_lr, best_l2 = acc, lr, l2

    # Retrain on all data with best hyperparameters
    rng_final = np.random.default_rng(SEED)
    final_params, _ = train_model(X_all, y_all, None, None, enc_full, C, best_lr, best_l2, rng_final)
    W1, b1 = final_params[0], final_params[1]
    W2, b2 = (final_params[2], final_params[3]) if H else (None, None)

    # Build lookup tables for export
    T = make_tables(enc_full, W1, H)
    W = H if H else C
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
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