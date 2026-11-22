```hypothesis
Family: label-mlp (one hidden layer MLP with validation-chosen width). The optdigits pixels have non-linear interactions; a small ReLU network captures this better than linear models. We select hidden size H ∈ {16,32,64,128} on the last 20% tuning split, then retrain on all data. If dev accuracy ≤ 0.9613 (label-linear) or inference cost exceeds label-mlp16 significantly, the hypothesis fails.
```

```python
"""label-mlp: one-hidden-layer ReLU MLP with per-byte first-layer tables.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameter H chosen on last 20% of train.in (file order), then retrained on all."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
H_CANDIDATES = [16, 32, 64, 128]  # searched on tuning split
MAGIC = 0x4D4C5031  # 'MLP1'

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def encode_ordinal(X, types):
    """Return list of per-feature encoding matrices (256 x dim_j) and total dim."""
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

def train_mlp(Z, y, H, C, rng):
    n, k = Z.shape
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            P = np.exp(logits); P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            gh = (g @ W2.T) * (a > 0)
            grads = [z.T @ gh + L2 * W1, gh.sum(0), h.T @ g + L2 * W2, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params

def accuracy(Z, y, params):
    W1, b1, W2, b2 = params
    a = Z @ W1 + b1
    h = np.maximum(a, 0)
    logits = h @ W2 + b2
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    n = X.shape[0]
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    enc = encode_ordinal(X, types)
    Z_all = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    Z_train = Z_all[:split]
    Z_val = Z_all[split:]

    best_H, best_acc = H_CANDIDATES[0], -1.0
    for H in H_CANDIDATES:
        rng = np.random.default_rng(SEED)
        params = train_mlp(Z_train, y_train, H, C, rng)
        acc = accuracy(Z_val, y_val, params)
        if acc > best_acc:
            best_acc, best_H = acc, H

    # Retrain on all data with best_H
    rng = np.random.default_rng(SEED)
    W1, b1, W2, b2 = train_mlp(Z_all, y, best_H, C, rng)

    # Build per-byte tables for first layer
    off, T = 0, []
    for E in enc:
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T = np.stack(T)  # d x 256 x H

    d = card["input"]["bytes"]
    H = best_H
    expected_size = 16 + 4 * (d * 256 * H + H + H * C + C)
    out = struct.pack("<4i", MAGIC, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    assert len(out) == expected_size
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from weights). Output: one byte class index.
 * Weights format: magic 0x4D4C5031, int32 d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 16 || hd[0] != 0x4D4C5031) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || H < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *sc = malloc(sizeof(float) * C);
    if (!acc || !sc) { fprintf(stderr, "oom\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int h = 0; h < H; h++) acc[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += t[h];
        }
        for (int c = 0; c < C; c++) sc[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float ah = acc[h] > 0.0f ? acc[h] : 0.0f;
            for (int c = 0; c < C; c++) sc[c] += ah * W2[(long)h * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```