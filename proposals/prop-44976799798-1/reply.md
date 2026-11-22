```hypothesis
Family: label-mlp-tuned (single hidden-layer MLP with hidden size chosen on tuning split).
Why: The optdigits pixels are ordinal (0..16) and likely have non-linear interactions; a small ReLU layer adds capacity over linear while staying cheap. Tuning H on the last 20% of train.in picks a sweet spot (e.g., 8–32 units) without hand-tuning.
Failure signal: If dev accuracy ≤ label-linear (0.9613) or cost per item exceeds label-mlp16, the extra complexity isn't justified.
```

```python
"""label-mlp-tuned: single-hidden-layer MLP with hidden size selected on tuning split.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes ordinal features as z-scored per byte (0..255 lookup tables).
Trains candidates H in {0,4,8,16,32} on first 80% of train.in (file order),
picks best accuracy on last 20%, then retrains on all data with that H.
Exports same table format as label-linear (magic 0x54424C31)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
H_CANDIDATES = [0, 4, 8, 16, 32]  # 0 = linear baseline


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_features(X, types, fit=True, encoders=None):
    """Return encoded matrix Z (n, k) and list of per-feature encoder matrices (256, dim_j)."""
    n, d = X.shape
    if fit:
        encoders = []
    Z_parts = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            if fit:
                mu, sd = v.mean(), v.std() or 1.0
                E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)  # (256,1)
                encoders.append(E)
            else:
                E = encoders[j]
            Z_parts.append(E[X[:, j]])
        else:
            if fit:
                vals = np.unique(X[:, j])
                E = np.zeros((256, len(vals)), np.float32)
                E[vals, np.arange(len(vals))] = 1.0
                encoders.append(E)
            else:
                E = encoders[j]
            Z_parts.append(E[X[:, j]])
    Z = np.concatenate(Z_parts, axis=1).astype(np.float32)
    return Z, encoders


def train_mlp(Z, y, H, C, rng, epochs):
    n, k = Z.shape
    W = H if H else C
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
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            def sg(S):
                S = S - S.max(axis=1, keepdims=True)
                P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                return (P / len(yy)).astype(np.float32)
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                g = sg(h @ W2 + b2)
                gh = (g @ W2.T) * (a > 0)
                grads = [z.T @ gh + L2 * W1, gh.sum(0), h.T @ g + L2 * W2, g.sum(0)]
            else:
                g = sg(z @ W1 + b1)
                grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params


def evaluate(Z, y, params, H, C):
    if H:
        W1, b1, W2, b2 = params
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        scores = h @ W2 + b2
    else:
        W1, b1 = params
        scores = Z @ W1 + b1
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def build_tables(encoders, params, H, C, d):
    W = H if H else C
    off = 0
    T_list = []
    for E in encoders:
        dim = E.shape[1]
        if H:
            T_list.append((E @ params[0][off:off + dim]).astype(np.float32))
        else:
            T_list.append((E @ params[0][off:off + dim]).astype(np.float32))
        off += dim
    return T_list


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    d = X_all.shape[1]

    # Split: last 20% for tuning (file order)
    n = len(X_all)
    split = int(0.8 * n)
    X_train, y_train = X_all[:split], y_all[:split]
    X_tune, y_tune = X_all[split:], y_all[split:]

    # Encode on training split
    Z_train, encoders = encode_features(X_train, types, fit=True)
    Z_tune, _ = encode_features(X_tune, types, fit=False, encoders=encoders)

    # Epochs based on training split size
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
    rng = np.random.default_rng(SEED)

    best_H, best_acc = 0, -1.0
    for H in H_CANDIDATES:
        # Deterministic per-H seed
        r = np.random.default_rng(SEED + H * 1000)
        params = train_mlp(Z_train, y_train, H, C, r, epochs)
        acc = evaluate(Z_tune, y_tune, params, H, C)
        if acc > best_acc:
            best_acc, best_H = acc, H

    # Retrain on all data with best_H
    Z_all, encoders_full = encode_features(X_all, types, fit=True)
    rng_final = np.random.default_rng(SEED + best_H * 1000)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    params_final = train_mlp(Z_all, y_all, best_H, C, rng_final, epochs_full)

    # Build lookup tables
    T_list = build_tables(encoders_full, params_final, best_H, C, d)
    W = best_H if best_H else C
    b1 = params_final[1].astype("<f4")
    out = struct.pack("<4i", 0x54424C31, d, best_H, C) + np.stack(T_list).astype("<f4").tobytes() + b1.tobytes()
    if best_H:
        W2 = params_final[2].astype("<f4")
        b2 = params_final[3].astype("<f4")
        out += W2.tobytes() + b2.tobytes()

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