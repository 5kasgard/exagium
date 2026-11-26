```hypothesis
Family: label-mlphs (one-hidden-layer ReLU MLP with hidden size chosen on the tuning split).
Why: The optdigits pixels are ordinal 0..16; a small non-linear model can capture stroke interactions that linear models miss. The example label-mlp16 (H=16) already reaches 97.7% dev accuracy, so searching H in {16,32,64,128} on the held-out 20% tuning split should match or exceed that while staying fast. The same per-byte lookup-table encoding as label-linear keeps inference at ~10k Ir/item.
Failure signal: If dev accuracy stays ≤96.2% (label-linear) the extra capacity does not help or the validation choice overfits.
```

```python
"""label-mlphs: one-hidden-layer ReLU MLP with hidden size selected on the tuning split.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes each input byte (ordinal→z-score, categorical→one-hot) into a lookup table.
Trains with Adam (lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n))).
Exports the same binary format as label-linear (magic 0x54424C31, d, H, C, T, b1, [W2, b2])."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
CANDIDATE_H = [16, 32, 64, 128]   # hidden sizes to try on the tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encodings(X, types):
    """Return list of encoding matrices E_j (256 x dim_j) and total dim k."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def encode(X, enc):
    """Apply encodings to X (n x d) → Z (n x k)."""
    parts = [E[X[:, j]] for j, E in enumerate(enc)]
    return np.concatenate(parts, axis=1).astype(np.float32)


def train_model(Z, y, H, C, seed):
    """Train on (Z, y) with given hidden size H, return params and rng state for reproducibility."""
    n, k = Z.shape
    rng = np.random.default_rng(seed)
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
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            def softmax_grad(S):
                S = S - S.max(axis=1, keepdims=True)
                P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                return (P / len(yy)).astype(np.float32)
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                g = softmax_grad(h @ W2 + b2)
                gh = (g @ W2.T) * (a > 0)
                grads = [z.T @ gh + L2 * W1, gh.sum(0), h.T @ g + L2 * W2, g.sum(0)]
            else:
                g = softmax_grad(z @ W1 + b1)
                grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params, enc


def predict(Z, params, H, C):
    """Return predicted class indices for Z."""
    if H:
        W1, b1, W2, b2 = params
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        scores = h @ W2 + b2
    else:
        W1, b1 = params
        scores = Z @ W1 + b1
    return scores.argmax(axis=1).astype(np.int64)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]

    # Split: last 20% is tuning split (validation), first 80% for candidate training
    n = X.shape[0]
    val_size = n // 5
    train_size = n - val_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    best_H, best_acc = 0, -1.0
    for H in CANDIDATE_H:
        enc = build_encodings(X_train, card["input"]["types"])
        Z_train = encode(X_train, enc)
        Z_val = encode(X_val, enc)
        # Use a deterministic seed per H to make the search reproducible
        params, _ = train_model(Z_train, y_train, H, C, SEED + H)
        preds = predict(Z_val, params, H, C)
        acc = (preds == y_val).mean()
        if acc > best_acc or (acc == best_acc and H < best_H):
            best_acc, best_H = acc, H

    # Retrain on full data with best_H
    enc = build_encodings(X, card["input"]["types"])
    Z = encode(X, enc)
    params, _ = train_model(Z, y, best_H, C, SEED)

    # Build per-byte lookup tables T[j][byte][w] = E_j[byte] @ W1[:, w]
    k = Z.shape[1]
    W = best_H if best_H else C
    W1 = params[0]
    T = []
    off = 0
    for E in enc:
        dim = E.shape[1]
        T.append((E.astype(np.float32) @ W1[off:off + dim]).astype(np.float32))
        off += dim
    T = np.stack(T)  # (d, 256, W)
    b1 = params[1].astype(np.float32)

    # Pack weights: magic, d, H, C; T (d*256*W float32); b1 (W); if H: W2 (H*C), b2 (C)
    out = struct.pack("<4i", 0x54424C31, d, best_H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if best_H:
        W2 = params[2].astype("<f4").tobytes()
        b2 = params[3].astype("<f4").tobytes()
        out += W2 + b2
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