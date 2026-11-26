```hypothesis
Family: label-tiny-mlp (one-hidden-layer MLP with ReLU, hidden size tuned on validation split).
Why: The optdigits dataset has non-linear structure; a small MLP can capture interactions missed by linear models while staying fast. The label-mlp16 achieves 97.7% but with 16 hidden units; tuning a smaller size (e.g., 4-16) should beat linear (96.1%) at lower cost than label-mlp16.
Failure: If dev accuracy <= 0.9613 (linear) or inference cost exceeds label-linear by >50%.
```

```python
"""label-tiny-mlp: one-hidden-layer MLP with ReLU, hidden size chosen on tuning split.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (fixed seeds), single thread, stdlib + numpy only.
Exports same table format as label-linear: magic 0x54424C31, d, H, C; float32 T[d][256][W], b1[W]; if H: W2[H][C], b2[C] (W=H)."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (same as label-linear)
SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
CANDIDATE_H = [4, 8, 12, 16]  # hidden sizes to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoding(X, types):
    """Return list of encoding matrices E_j (256, dim_j) and total dim k."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])  # (256,1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def encode(X, enc):
    """Apply encodings to X (n, d) -> Z (n, k)."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def init_params(k, H, C, rng):
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


def softmax_grad(logits, y):
    """Return gradient of cross-entropy w.r.t logits."""
    S = logits - logits.max(axis=1, keepdims=True)
    P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
    P[np.arange(len(y)), y] -= 1.0
    return (P / len(y)).astype(np.float32)


def train_mlp(Z, y, H, C, rng, epochs, batch_size, lr, l2):
    k = Z.shape[1]
    params = init_params(k, H, C, rng)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = Z.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            z, yy = Z[idx], y[idx]
            if H:
                a = z @ params[0] + params[1]          # (batch, H)
                h = np.maximum(a, 0)
                logits = h @ params[2] + params[3]     # (batch, C)
                g = softmax_grad(logits, yy)           # (batch, C)
                gh = (g @ params[2].T) * (a > 0)       # (batch, H)
                grads = [
                    z.T @ gh + l2 * params[0],         # W1
                    gh.sum(0),                         # b1
                    h.T @ g + l2 * params[2],          # W2
                    g.sum(0),                          # b2
                ]
            else:
                logits = z @ params[0] + params[1]
                g = softmax_grad(logits, yy)
                grads = [
                    z.T @ g + l2 * params[0],
                    g.sum(0),
                ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params


def predict(Z, params, H, C):
    if H:
        a = Z @ params[0] + params[1]
        h = np.maximum(a, 0)
        logits = h @ params[2] + params[3]
    else:
        logits = Z @ params[0] + params[1]
    return logits.argmax(axis=1).astype(np.int64)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    d = X.shape[1]

    # Split: last 20% for tuning (file order)
    n = X.shape[0]
    tune_size = int(0.2 * n)
    train_size = n - tune_size
    X_tr, y_tr = X[:train_size], y[:train_size]
    X_tune, y_tune = X[train_size:], y[train_size:]

    # Hyperparameter search on tuning split
    best_H, best_acc = None, -1.0
    for H in CANDIDATE_H:
        rng = np.random.default_rng(SEED + H)
        enc = build_encoding(X_tr, types)
        Z_tr = encode(X_tr, enc)
        Z_tune = encode(X_tune, enc)
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / train_size))
        params = train_mlp(Z_tr, y_tr, H, C, rng, epochs, BATCH, LR, L2)
        pred = predict(Z_tune, params, H, C)
        acc = (pred == y_tune).mean()
        if acc > best_acc:
            best_acc, best_H = acc, H

    # Retrain on full data with best_H
    rng = np.random.default_rng(SEED + best_H)
    enc_full = build_encoding(X, types)
    Z_full = encode(X, enc_full)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    params = train_mlp(Z_full, y, best_H, C, rng, epochs, BATCH, LR, L2)

    # Build lookup tables T (d, 256, W) where W = best_H
    W = best_H
    off, T_list = 0, []
    for E in enc_full:
        # E is (256, 1) for ordinal
        T_list.append((E.astype(np.float32) @ params[0][off:off + E.shape[1], :]).astype(np.float32))
        off += E.shape[1]
    T = np.stack(T_list)  # (d, 256, W)

    # Write weights.bin
    magic, d, H, C = 0x54424C31, d, best_H, C
    out = struct.pack("<4i", magic, d, H, C)
    out += T.astype("<f4").tobytes()
    out += params[1].astype("<f4").tobytes()  # b1 (W,)
    if H:
        out += params[2].astype("<f4").tobytes()  # W2 (H, C)
        out += params[3].astype("<f4").tobytes()  # b2 (C,)
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