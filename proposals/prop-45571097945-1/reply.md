```hypothesis
Family: label-rff (Random Fourier Features for RBF kernel approximation).
Why: Optdigits pixels have spatial structure; RBF kernel captures locality non-linearly. RFF approximates RBF with a fixed random projection + linear layer, adding non-linearity with only one extra matrix multiply and cosines. It should beat linear (0.961) and approach gbdt (0.979) while being faster than gbdt/knn.
Failure: If dev accuracy ≤ 0.961 or instructions/item > 2×label-linear, the family does not improve the trade-off.
```

```python
"""label-rff: Random Fourier Features + linear layer for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes ordinal bytes as z-scored. Generates fixed random projection (W,b) for RBF kernel approximation.
Trains linear layer on RFF features with Adam. Hyperparameters (D, gamma) chosen on last 20% tuning split."""
import json, math, struct, sys

import numpy as np

# Fixed training hyperparameters
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
# RFF hyperparameter grid
D_GRID = [64, 128, 256, 512]
GAMMA_GRID = [0.1, 0.5, 1.0, 2.0]
MAGIC = 0x5246464C  # 'RFFL'

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]

    # Per-byte z-scoring (all ordinal)
    mu = X.mean(axis=0).astype(np.float64)
    sd = X.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)  # (n, d)

    # Tuning split: last 20%
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)
    Z_train, y_train = Z[train_idx], y[train_idx]
    Z_tune, y_tune = Z[tune_idx], y[tune_idx]

    # Base RNG for reproducible random projections
    base_rng = np.random.default_rng(SEED)

    best_acc = -1.0
    best_D = D_GRID[0]
    best_gamma = GAMMA_GRID[0]
    best_W = None
    best_b = None
    best_W2 = None
    best_b2 = None

    for D in D_GRID:
        for gamma in GAMMA_GRID:
            # Deterministic seed per (D, gamma)
            seed = (D * 10000 + int(gamma * 1000)) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            # Random projection: W ~ N(0, gamma), b ~ U(0, 2π)
            W = (rng.normal(0, math.sqrt(gamma), (D, d))).astype(np.float32)
            b = (rng.uniform(0, 2 * math.pi, D)).astype(np.float32)

            # Compute RFF features: sqrt(2/D) * cos(Z @ W.T + b)
            scale = math.sqrt(2.0 / D)
            Z_train_rff = scale * np.cos(Z_train @ W.T + b)  # (n_train, D)
            Z_tune_rff = scale * np.cos(Z_tune @ W.T + b)    # (n_tune, D)

            # Train linear layer on RFF features
            n_train, D_feat = Z_train_rff.shape
            rng_train = np.random.default_rng(SEED)  # fixed seed for training
            # Xavier init for W2
            W2 = (rng_train.uniform(-1, 1, (D_feat, C)) * math.sqrt(6.0 / (D_feat + C))).astype(np.float32)
            b2 = np.zeros(C, np.float32)
            params = [W2, b2]
            state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
            t = 0

            for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))):
                order = rng_train.permutation(n_train)
                for s in range(0, n_train, BATCH):
                    idx = order[s:s + BATCH]
                    z, yy = Z_train_rff[idx], y_train[idx]
                    # Softmax gradient
                    logits = z @ W2 + b2
                    logits -= logits.max(axis=1, keepdims=True)
                    exp_logits = np.exp(logits)
                    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                    probs[np.arange(len(yy)), yy] -= 1.0
                    g = (probs / len(yy)).astype(np.float32)
                    grads = [z.T @ g + L2 * W2, g.sum(0)]
                    t += 1
                    for i, (p, gr) in enumerate(zip(params, grads)):
                        m, v = state[i]
                        m *= 0.9; m += 0.1 * gr
                        v *= 0.999; v += 0.001 * gr * gr
                        p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

            # Evaluate on tuning split
            logits_tune = Z_tune_rff @ W2 + b2
            preds = logits_tune.argmax(axis=1)
            acc = (preds == y_tune).mean()
            if acc > best_acc:
                best_acc = acc
                best_D = D
                best_gamma = gamma
                best_W = W.copy()
                best_b = b.copy()
                best_W2 = W2.copy()
                best_b2 = b2.copy()

    # Retrain on full dataset with best hyperparameters
    rng = np.random.default_rng((best_D * 10000 + int(best_gamma * 1000)) & 0xFFFFFFFF)
    W = (rng.normal(0, math.sqrt(best_gamma), (best_D, d))).astype(np.float32)
    b = (rng.uniform(0, 2 * math.pi, best_D)).astype(np.float32)
    scale = math.sqrt(2.0 / best_D)
    Z_rff = scale * np.cos(Z @ W.T + b)

    n_full = Z_rff.shape[0]
    rng_train = np.random.default_rng(SEED)
    W2 = (rng_train.uniform(-1, 1, (best_D, C)) * math.sqrt(6.0 / (best_D + C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n_full))):
        order = rng_train.permutation(n_full)
        for s in range(0, n_full, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_rff[idx], y[idx]
            logits = z @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W2, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Save weights: magic, d, D, C; then float32 arrays: mu, sd, W, b, W2, b2
    mu_f32 = mu.astype(np.float32)
    sd_f32 = sd.astype(np.float32)
    header = struct.pack("<4i", MAGIC, d, best_D, C)
    body = (mu_f32.tobytes() + sd_f32.tobytes() +
            best_W.astype(np.float32).tobytes() +
            best_b.astype(np.float32).tobytes() +
            best_W2.astype(np.float32).tobytes() +
            best_b2.astype(np.float32).tobytes())
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-rff inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output record: 1 byte (class 0..9).
 * Weights format: magic(0x5246464C), d, D, C; float32 mu[d], sd[d], W[D*d], b[D], W2[D*C], b2[C]. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    if (nw < 16 || hd[0] != 0x5246464C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 16L + 4L * (2L * d + (long)D * d + D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *mu = (const float *)(hd + 4);
    const float *sd = mu + d;
    const float *W = sd + d;
    const float *b = W + (long)D * d;
    const float *W2 = b + D;
    const float *b2 = W2 + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *rff = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);
    float scale = sqrtf(2.0f / D);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize and compute RFF features
        for (int u = 0; u < D; u++) {
            float sum = b[u];
            const float *wu = W + (long)u * d;
            for (int j = 0; j < d; j++) {
                float val = (xi[j] - mu[j]) / sd[j];
                sum += val * wu[j];
            }
            rff[u] = scale * cosf(sum);
        }

        // Linear layer
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int u = 0; u < D; u++) {
            float fu = rff[u];
            const float *w2u = W2 + (long)u * C;
            for (int c = 0; c < C; c++) scores[c] += fu * w2u[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```