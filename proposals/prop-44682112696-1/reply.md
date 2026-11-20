```hypothesis
Family: label-mlptune (small MLP with hidden size tuned on last 20% of train).
Optdigits pixels have local correlations; a single ReLU layer (4-16 units) captures non-linearities
that linear models miss, while staying cheaper than label-mlp16.
If dev accuracy ≤ 0.9613 (label-linear) or instructions/item ≥ 6000, the family fails.
```

```python
"""Table family with tuned hidden size for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes each byte: ordinal z-scored, categorical one-hot.
Trains a one-hidden-layer ReLU MLP (or linear if H=0) with Adam.
Chooses H on the last 20% of train.in (file order), then retrains on all data.
Exported format: magic 0x54424C31, d, H, C; float32 T[d][256][W], b1[W];
if H>0: W2[H][C], b2[C] (W = H if H>0 else C)."""
import json
import math
import struct
import sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
CANDIDATE_H = [0, 4, 8, 12, 16]  # 0 = linear baseline


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def compute_encodings(X, types):
    """Return list of encoding matrices E_j (256, dim_j) for each byte."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)
            enc.append(E)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def apply_encodings(X, enc):
    """X (n, d) uint8 -> Z (n, total_dim) float32."""
    parts = [enc[j][X[:, j]] for j in range(X.shape[1])]
    return np.concatenate(parts, axis=1).astype(np.float32)


def train_model(Z, y, H, C, n_epochs, rng):
    """Train on Z (n, k), return W1, b1, W2, b2 (W2,b2 None if H=0)."""
    n, k = Z.shape
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

    def sg(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                g = sg(h @ W2 + b2, yy)
                gh = (g @ W2.T) * (a > 0)
                grads = [
                    z.T @ gh + L2 * W1,
                    gh.sum(0),
                    h.T @ g + L2 * W2,
                    g.sum(0),
                ]
            else:
                g = sg(z @ W1 + b1, yy)
                grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    if H:
        return W1, b1, W2, b2
    else:
        return W1, b1, None, None


def evaluate(Z, y, H, W1, b1, W2, b2):
    """Return accuracy on Z, y."""
    if H:
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        scores = h @ W2 + b2
    else:
        scores = Z @ W1 + b1
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    d = X.shape[1]
    n = X.shape[0]

    # Split: last 20% for tuning
    n_tune = n // 5
    train_idx = n - n_tune
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_tune, y_tune = X[train_idx:], y[train_idx:]

    best_H = 0
    best_acc = -1.0

    for H in CANDIDATE_H:
        # Encodings on training portion only
        enc = compute_encodings(X_train, types)
        Z_train = apply_encodings(X_train, enc)
        Z_tune = apply_encodings(X_tune, enc)

        n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(y_train)))
        rng = np.random.default_rng(SEED + H * 1000)
        W1, b1, W2, b2 = train_model(Z_train, y_train, H, C, n_epochs, rng)
        acc = evaluate(Z_tune, y_tune, H, W1, b1, W2, b2)
        if acc > best_acc:
            best_acc = acc
            best_H = H

    # Final training on all data with best_H
    enc = compute_encodings(X, types)
    Z = apply_encodings(X, enc)
    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng = np.random.default_rng(SEED)
    W1, b1, W2, b2 = train_model(Z, y, best_H, C, n_epochs, rng)

    # Export weights
    W = best_H if best_H else C
    off = 0
    T_list = []
    for E in enc:
        dim = E.shape[1]
        T_list.append((E @ W1[off:off + dim]).astype(np.float32))
        off += dim
    T = np.stack(T_list)  # (d, 256, W)

    out = struct.pack("<4i", 0x54424C31, d, best_H, C)
    out += T.astype("<f4").tobytes()
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