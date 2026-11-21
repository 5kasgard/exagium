```hypothesis
We propose a two-hidden-layer MLP (label-mlp2h) with ReLU activations. The 8x8 pixel input benefits from hierarchical feature extraction; two layers can capture interactions between pixels better than a single layer. We tune hidden sizes (H1, H2) on the last 20% of train.in, then retrain on all data. If dev accuracy does not exceed 0.965 (beating label-linear) or cost per item exceeds label-mlp16, the hypothesis fails.
```

```python
"""Two-hidden-layer MLP for label tasks with fixed-size byte items (ordinal z-scored).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic: fixed seed, single thread. Chooses architecture on tuning split (last 20%).
Exports: magic 0x4D4C5032, d, H1, H2, C; float32 T1[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C]."""
import json, math, struct, sys

import numpy as np

# Tuning grid
GRID = [
    (16, 8), (32, 16), (64, 32),
]
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 80
SAMPLES = 600000


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def zscore_encode(X, means, stds):
    """X: (n, d) uint8 -> (n, d) float32 z-scored."""
    return ((X.astype(np.float32) - means) / stds).astype(np.float32)


def softmax_cross_entropy_grad(logits, y):
    """logits: (n, C), y: (n,) int. Returns gradient (n, C) float32."""
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return (probs / len(y)).astype(np.float32)


def evaluate(Z, y, params):
    W1, b1, W2, b2, W3, b3 = params
    h1 = np.maximum(Z @ W1 + b1, 0)
    h2 = np.maximum(h1 @ W2 + b2, 0)
    logits = h2 @ W3 + b3
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def train_model(Z_train, y_train, Z_val, y_val, H1, H2, rng):
    n, d = Z_train.shape
    C = 10
    # Xavier init
    W1 = (rng.uniform(-1, 1, (d, H1)) * math.sqrt(6.0 / d)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
    b3 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_train[idx], y_train[idx]
            # forward
            h1_pre = z @ W1 + b1
            h1 = np.maximum(h1_pre, 0)
            h2_pre = h1 @ W2 + b2
            h2 = np.maximum(h2_pre, 0)
            logits = h2 @ W3 + b3
            # loss grad
            g = softmax_cross_entropy_grad(logits, yy)
            # backward
            g_W3 = h2.T @ g + L2 * W3
            g_b3 = g.sum(0)
            gh2 = (g @ W3.T) * (h2_pre > 0)
            g_W2 = h1.T @ gh2 + L2 * W2
            g_b2 = gh2.sum(0)
            gh1 = (gh2 @ W2.T) * (h1_pre > 0)
            g_W1 = z.T @ gh1 + L2 * W1
            g_b1 = gh1.sum(0)
            grads = [g_W1, g_b1, g_W2, g_b2, g_W3, g_b3]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    val_acc = evaluate(Z_val, y_val, params)
    return val_acc, params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    n = X.shape[0]
    n_val = int(0.2 * n)
    idx_train = np.arange(n - n_val)
    idx_val = np.arange(n - n_val, n)

    # Compute stats on full train for final encoding, but for tuning we use train portion stats.
    # We'll do tuning with stats from train portion, then final with full stats.
    best_acc = -1
    best_params = None
    best_H1 = best_H2 = None
    best_means = best_stds = None

    for H1, H2 in GRID:
        # stats on train portion
        means = X[idx_train].mean(axis=0).astype(np.float32)
        stds = X[idx_train].std(axis=0).astype(np.float32)
        stds[stds == 0] = 1.0
        Z_train = zscore_encode(X[idx_train], means, stds)
        Z_val = zscore_encode(X[idx_val], means, stds)
        y_train = y[idx_train]
        y_val = y[idx_val]

        rng = np.random.default_rng(SEED + H1 * 100 + H2)
        val_acc, params = train_model(Z_train, y_train, Z_val, y_val, H1, H2, rng)
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params
            best_H1, best_H2 = H1, H2
            best_means, best_stds = means, stds

    # Retrain on full data with best architecture
    print(f"Best architecture: H1={best_H1}, H2={best_H2}, val_acc={best_acc:.4f}", file=sys.stderr)
    means = X.mean(axis=0).astype(np.float32)
    stds = X.std(axis=0).astype(np.float32)
    stds[stds == 0] = 1.0
    Z_full = zscore_encode(X, means, stds)
    rng = np.random.default_rng(SEED + best_H1 * 100 + best_H2 + 1000)
    _, final_params = train_model(Z_full, y, Z_full, y, best_H1, best_H2, rng)  # no val needed
    W1, b1, W2, b2, W3, b3 = final_params

    # Build embedding tables T1: for each feature j, value v -> (v - mean)/std * W1[j]
    T1 = np.empty((d, 256, best_H1), dtype=np.float32)
    for j in range(d):
        scale = (np.arange(256, dtype=np.float32) - means[j]) / stds[j]
        T1[j] = scale[:, None] * W1[j]  # (256, H1)

    # Write weights
    magic = 0x4D4C5032  # "MLP2"
    out = struct.pack("<5i", magic, d, best_H1, best_H2, C)
    out += T1.astype("<f4").tobytes()
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
/* Two-hidden-layer MLP (label-mlp2h) for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class index. */
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
    if (nw < 20 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H1 + H1 + (long)H1 * H2 + H2 + (long)H2 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }
    const float *T1 = (const float *)(hd + 5);
    const float *b1 = T1 + (long)d * 256 * H1;
    const float *W2 = b1 + H1;
    const float *b2 = W2 + (long)H1 * H2;
    const float *W3 = b2 + H2;
    const float *b3 = W3 + (long)H2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc1 = malloc(sizeof(float) * H1);
    float *acc2 = malloc(sizeof(float) * H2);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Layer 1: acc1 = b1 + sum_j T1[j][xi[j]]
        for (int h = 0; h < H1; h++) acc1[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *t = T1 + ((long)j * 256 + xi[j]) * H1;
            for (int h = 0; h < H1; h++) acc1[h] += t[h];
        }
        // ReLU1
        for (int h = 0; h < H1; h++) if (acc1[h] < 0.0f) acc1[h] = 0.0f;
        // Layer 2: acc2 = b2 + acc1 @ W2
        for (int h = 0; h < H2; h++) acc2[h] = b2[h];
        for (int u = 0; u < H1; u++) {
            float a = acc1[u];
            if (a == 0.0f) continue;
            const float *w = W2 + (long)u * H2;
            for (int h = 0; h < H2; h++) acc2[h] += a * w[h];
        }
        // ReLU2
        for (int h = 0; h < H2; h++) if (acc2[h] < 0.0f) acc2[h] = 0.0f;
        // Output layer: scores = b3 + acc2 @ W3
        for (int c = 0; c < C; c++) scores[c] = b3[c];
        for (int u = 0; u < H2; u++) {
            float a = acc2[u];
            if (a == 0.0f) continue;
            const float *w = W3 + (long)u * C;
            for (int c = 0; c < C; c++) scores[c] += a * w[c];
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```