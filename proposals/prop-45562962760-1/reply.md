```hypothesis
Family: label-mlp2tuned (two-hidden-layer MLP with per-byte first-layer embedding, hidden sizes tuned on validation split).
Why: The optdigits pixels have local spatial structure; two ReLU layers can capture interactions better than one. Tuning H1,H2 on the last 20% of train.in adapts capacity to the task. The per-byte embedding keeps inference fast (table lookups + two dense layers). Expected to beat label-linear (0.9613) and approach label-mlp16 (0.9770) with moderate cost.
Failure: If dev accuracy <= 0.9613 or cost > label-mlp16, the extra layer is not justified.
```

```python
"""Two-hidden-layer MLP with per-byte first-layer embedding, hidden sizes tuned on validation split.
python3 -s -P train.py card.json train.in train.key weights.bin   (deterministic: seed 1, single thread)
Encoding by the card's per-byte types: ordinal z-scored. First layer: per-byte lookup table to H1, then ReLU.
Second layer: dense H1->H2 + ReLU. Third layer: dense H2->C.
Adam lr 3e-3, L2 1e-4, minibatch 64, epochs min(100, ceil(600000 / n)).
Exported as: int32 magic 0x4D4C5032, d, H1, H2, C; float32 T[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
# Candidate (H1, H2) pairs to try on the tuning split
CANDIDATES = [(32, 16), (64, 32), (16, 8), (64, 16), (32, 32)]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_mlp(Z, y, H1, H2, C, rng, epochs):
    """Train MLP on given data, return parameters (W1, b1, W2, b2, W3, b3)."""
    n, d = Z.shape
    # Initialize
    W1 = (rng.uniform(-1, 1, (d, H1)) * math.sqrt(6.0 / d)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
    b3 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def sg(S):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # Forward
            a1 = z @ W1 + b1
            h1 = np.maximum(a1, 0)
            a2 = h1 @ W2 + b2
            h2 = np.maximum(a2, 0)
            a3 = h2 @ W3 + b3
            # Backward
            g3 = sg(a3)  # (batch, C)
            g2 = (g3 @ W3.T) * (a2 > 0)  # (batch, H2)
            g1 = (g2 @ W2.T) * (a1 > 0)  # (batch, H1)
            grads = [
                z.T @ g1 + L2 * W1,  # dW1
                g1.sum(0),            # db1
                h1.T @ g2 + L2 * W2,  # dW2
                g2.sum(0),            # db2
                h2.T @ g3 + L2 * W3,  # dW3
                g3.sum(0),            # db3
            ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params


def evaluate(Z, y, params):
    W1, b1, W2, b2, W3, b3 = params
    a1 = Z @ W1 + b1
    h1 = np.maximum(a1, 0)
    a2 = h1 @ W2 + b2
    h2 = np.maximum(a2, 0)
    a3 = h2 @ W3 + b3
    pred = a3.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]

    # Normalize each column (all ordinal)
    mu = X.mean(axis=0).astype(np.float64)
    sd = X.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)

    # Split: last 20% for validation (tuning split)
    n = len(Z)
    val_size = n // 5
    train_idx = np.arange(n - val_size)
    val_idx = np.arange(n - val_size, n)
    Z_train, y_train = Z[train_idx], y[train_idx]
    Z_val, y_val = Z[val_idx], y[val_idx]

    # Tune H1, H2
    base_rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_H1, best_H2 = CANDIDATES[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(Z_train)))

    for idx, (H1, H2) in enumerate(CANDIDATES):
        # Deterministic seed per candidate
        rng = np.random.default_rng(SEED + idx + 1)
        params = train_mlp(Z_train, y_train, H1, H2, C, rng, epochs)
        acc = evaluate(Z_val, y_val, params)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_H1, best_H2 = H1, H2

    # Retrain on full data with best config
    rng = np.random.default_rng(SEED)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    W1, b1, W2, b2, W3, b3 = train_mlp(Z, y, best_H1, best_H2, C, rng, epochs_full)

    # Build per-byte lookup table T for first layer: T[j, v, :] = (v - mu_j)/sd_j * W1[j, :]
    T = np.zeros((d, 256, best_H1), dtype=np.float32)
    for j in range(d):
        scale = (np.arange(256, dtype=np.float32) - mu[j]) / sd[j]
        T[j] = scale[:, None] * W1[j, None, :]

    # Pack weights: magic, d, H1, H2, C; then T, b1, W2, b2, W3, b3 (all float32 little-endian)
    magic = 0x4D4C5032  # 'MLP2'
    header = struct.pack("<4i", magic, d, best_H1, best_H2, C)
    # Note: header is 5 int32? Actually 4i packs 4 ints. We need 5 ints: magic, d, H1, H2, C.
    # Fix: use 5i.
    header = struct.pack("<5i", magic, d, best_H1, best_H2, C)
    out = header
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    out += W3.astype("<f4").tobytes()
    out += b3.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Two-hidden-layer MLP with per-byte first-layer embedding: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class (argmax). Format: see train.py. */
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
    if (nw < 20 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256 || H1 < 1 || H2 < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H1 + H1 + (long)H1 * H2 + H2 + (long)H2 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

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
    float *acc3 = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // First layer: per-byte lookup + bias
        for (int u = 0; u < H1; u++) acc1[u] = b1[u];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H1;
            for (int u = 0; u < H1; u++) acc1[u] += t[u];
        }
        // ReLU
        for (int u = 0; u < H1; u++) h1[u] = acc1[u] > 0.0f ? acc1[u] : 0.0f;

        // Second layer
        for (int v = 0; v < H2; v++) acc2[v] = b2[v];
        for (int u = 0; u < H1; u++) {
            float hu = h1[u];
            if (hu == 0.0f) continue;
            const float *w = W2 + (long)u * H2;
            for (int v = 0; v < H2; v++) acc2[v] += hu * w[v];
        }
        // ReLU
        for (int v = 0; v < H2; v++) h2[v] = acc2[v] > 0.0f ? acc2[v] : 0.0f;

        // Third layer (output)
        for (int c = 0; c < C; c++) acc3[c] = b3[c];
        for (int v = 0; v < H2; v++) {
            float hv = h2[v];
            if (hv == 0.0f) continue;
            const float *w = W3 + (long)v * C;
            for (int c = 0; c < C; c++) acc3[c] += hv * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc3[c] > acc3[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(acc1); free(h1); free(acc2); free(h2); free(acc3);
    return 0;
}
```