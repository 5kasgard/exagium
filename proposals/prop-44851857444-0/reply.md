```hypothesis
Family: label-mlp-tuned. A one-hidden-layer MLP (ReLU) whose hidden size is chosen from {8,16,32,64} on the last 20% of train.in (validation split), then retrained on all data. The non-linearity should capture feature interactions that logistic regression misses, while validation-based sizing prevents overfitting. If dev accuracy does not exceed label-linear (0.8524), the hypothesis fails.
```

```python
"""label-mlp-tuned: one-hidden-layer MLP with validation-tuned hidden size.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (fixed seeds), single thread, stdlib+numpy only.
Weight format identical to label-linear (magic 0x54424C31, d, H, C, T, b1, W2, b2)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
HIDDEN_CANDIDATES = [8, 16, 32, 64]   # tried on validation split (last 20%)


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode(X, types):
    """Return list of encoding matrices E_j (256 x k_j) and total k."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def forward(Z, W1, b1, W2, b2, H):
    """Return logits (n x C)."""
    if H:
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        return h @ W2 + b2
    else:
        return Z @ W1 + b1


def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


def train_split(Z, y, H, rng, n_samples):
    """Train on given split, return (W1,b1,W2,b2)."""
    n, k = Z.shape
    C = 2
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
    epochs = min(MAX_EPOCHS, math.ceil(n_samples / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # forward
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                logits = h @ W2 + b2
            else:
                logits = z @ W1 + b1
            # softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            P = np.exp(logits)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            # backward
            if H:
                gh = (g @ W2.T) * (a > 0)
                grads = [z.T @ gh + L2 * W1, gh.sum(0),
                         h.T @ g + L2 * W2, g.sum(0)]
            else:
                grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    types = card["input"]["types"]
    C = card["classes"]

    # encode once on full training data (statistics from full train)
    enc = encode(X, types)
    Z_full = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

    # validation split: last 20% in file order
    n = Z_full.shape[0]
    split = int(0.8 * n)
    Z_tr, y_tr = Z_full[:split], y[:split]
    Z_val, y_val = Z_full[split:], y[split:]

    best_H, best_acc = 0, -1.0
    for H in HIDDEN_CANDIDATES:
        rng = np.random.default_rng(SEED + H)
        params = train_split(Z_tr, y_tr, H, rng, SAMPLES)
        W1, b1 = params[0], params[1]
        if H:
            W2, b2 = params[2], params[3]
        else:
            W2, b2 = None, None
        logits = forward(Z_val, W1, b1, W2, b2, H)
        acc = accuracy(logits, y_val)
        if acc > best_acc:
            best_acc, best_H = acc, H

    # retrain on all data with best_H
    rng = np.random.default_rng(SEED)
    params = train_split(Z_full, y, best_H, rng, SAMPLES)
    W1, b1 = params[0], params[1]
    if best_H:
        W2, b2 = params[2], params[3]
    else:
        W2, b2 = np.zeros((C, C), np.float32), np.zeros(C, np.float32)  # dummy, not used

    # build per-byte lookup tables T[j][256][W] where W = best_H (or C if linear)
    W = best_H if best_H else C
    off, T = 0, []
    for E in enc:
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]

    # pack weights.bin
    out = struct.pack("<4i", 0x54424C31, X.shape[1], best_H, C)
    out += np.stack(T).astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if best_H:
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