```hypothesis
Family: label-mlp-tuned (one-hidden-layer ReLU MLP with hidden size chosen by validation on the tuning split).
Why: The adult dataset has non-linear feature interactions; a small hidden layer can capture them while keeping inference fast (cost scales with H). Tuning H on the last 20% of train.in selects a good accuracy/cost trade-off.
Failure: If no H in {2,4,8,16} beats label-linear on dev accuracy, or if the best H's inference cost exceeds label-linear by >2x without accuracy gain.
```

```python
"""Table family with tuned hidden size for label tasks: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy. Chooses H from {2,4,8,16} on the last 20% of train.in, then trains on all data.
Exported as per-byte lookup tables: int32 magic 0x54424C31, d, H, C; float32 T[d][256][W], b1[W]; if H: float32 W2[H][C], b2[C] (W = H, or C for linear)."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
H_OPTIONS = [2, 4, 8, 16]  # hidden sizes to try on the tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoders(X, types):
    """Return list of encoding matrices (256 x n_vals) for each column."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def transform(X, enc):
    """Apply encoders to X (n x d uint8) -> Z (n x k float32)."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def train_val_split(X, y, val_frac=0.2):
    """Split last val_frac for validation (file order)."""
    n = X.shape[0]
    val_size = n // 5
    train_size = n - val_size
    return X[:train_size], y[:train_size], X[train_size:], y[train_size:]


def init_params(k, H, C, rng):
    """Initialize weights for given architecture."""
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        return [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), np.float32)
        b1 = np.zeros(C, np.float32)
        return [W1, b1]


def sgd_step(z, yy, params, H, C, L2, state, t, LR):
    """One minibatch Adam update, returns updated params and state."""
    def softmax_grad(S):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    if H:
        W1, b1, W2, b2 = params
        a = z @ W1 + b1
        h = np.maximum(a, 0)
        g = softmax_grad(h @ W2 + b2)
        gh = (g @ W2.T) * (a > 0)
        grads = [z.T @ gh + L2 * W1, gh.sum(0), h.T @ g + L2 * W2, g.sum(0)]
    else:
        W1, b1 = params
        g = softmax_grad(z @ W1 + b1)
        grads = [z.T @ g + L2 * W1, g.sum(0)]

    t += 1
    new_params = []
    new_state = []
    for p, gr, (m, v) in zip(params, grads, state):
        m = m * 0.9 + 0.1 * gr
        v = v * 0.999 + 0.001 * gr * gr
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        p = p - LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        new_params.append(p)
        new_state.append((m, v))
    return new_params, new_state, t


def evaluate(Z, y, params, H, C):
    """Return accuracy on given data."""
    if H:
        W1, b1, W2, b2 = params
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
    else:
        W1, b1 = params
        logits = Z @ W1 + b1
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def train_model(X, y, enc, H, C, rng, epochs, batch):
    """Train on full dataset, return final params."""
    Z = transform(X, enc)
    n, k = Z.shape
    params = init_params(k, H, C, rng)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            params, state, t = sgd_step(Z[idx], y[idx], params, H, C, L2, state, t, LR)
    return params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]

    # Tuning split: last 20% of train.in in file order
    X_tr, y_tr, X_val, y_val = train_val_split(X, y)

    best_H = None
    best_acc = -1.0

    for H in H_OPTIONS:
        # Use a deterministic seed per H
        rng = np.random.default_rng(SEED + H)
        enc = build_encoders(X_tr, types)
        Z_tr = transform(X_tr, enc)
        Z_val = transform(X_val, enc)
        n_tr, k = Z_tr.shape
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
        params = init_params(k, H, C, rng)
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0
        for _ in range(epochs):
            order = rng.permutation(n_tr)
            for s in range(0, n_tr, BATCH):
                idx = order[s:s + BATCH]
                params, state, t = sgd_step(Z_tr[idx], y_tr[idx], params, H, C, L2, state, t, LR)
        acc = evaluate(Z_val, y_val, params, H, C)
        if acc > best_acc:
            best_acc = acc
            best_H = H

    # Retrain on all data with best_H
    rng = np.random.default_rng(SEED + best_H)
    enc = build_encoders(X, types)
    Z = transform(X, enc)
    n, k = Z.shape
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    params = train_model(X, y, enc, best_H, C, rng, epochs, BATCH)

    # Export weights in the same format as label-linear
    W = best_H if best_H else C
    # Build T: for each column j, T_j = E_j @ W1[:, offset:offset+E_j.shape[1]]
    if best_H:
        W1, b1, W2, b2 = params
    else:
        W1, b1 = params
        W2 = b2 = None

    T = []
    off = 0
    for E in enc:
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]

    out = struct.pack("<4i", 0x54424C31, d, best_H, C) + np.stack(T).astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    if best_H:
        out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()

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