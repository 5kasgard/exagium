```hypothesis
Family: label-rff (Random Fourier Features approximating an RBF kernel). The 8x8 digit pixels exhibit local spatial correlations that a linear model misses; an RBF kernel captures these non-linear interactions. RFF maps inputs to explicit cosine features (cos(Wx+b)) enabling fast linear inference. With 64–256 features tuned on the last 20% of train, it should surpass label-linear's 96.1% accuracy while staying near 5k–10k Ir/item. Failure if dev accuracy ≤ 96.1% or Ir/item > 12000.
```

```python
"""label-rff: Random Fourier Features (RBF kernel approx) + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Encoding: per-byte z-scoring (ordinal). RFF: D features, W~N(0,gamma), b~U[0,2pi].
Linear head trained with Adam (lr=3e-3, L2=1e-4, batch=64, epochs min(100, ceil(600000/n))).
Hyperparameter D chosen on tuning split (last 20% of train), then retrained on all train.
Weights: magic 0x52464631, d, D, C; float32 mean[d], inv_std[d], W[d][D], b[D], W2[D][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
D_CANDIDATES = [32, 64, 128, 256]  # RFF feature counts to try


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def zscore_encode(X, types):
    """Return (Z, mean, inv_std) where Z is float32 [n, d] normalized."""
    n, d = X.shape
    mean = np.zeros(d, np.float64)
    inv_std = np.ones(d, np.float64)
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std()
            mean[j], inv_std[j] = mu, 1.0 / (sd if sd > 0 else 1.0)
            X[:, j] = (v - mu) * inv_std[j]
        else:
            # Not used here, but keep interface
            pass
    return X.astype(np.float32), mean.astype(np.float32), inv_std.astype(np.float32)


def rff_features(Z, W, b):
    """Z: [n, d], W: [d, D], b: [D] -> [n, D] cos features scaled by sqrt(2/D)."""
    # Z @ W + b  (broadcast b)
    proj = Z @ W + b  # [n, D]
    np.cos(proj, out=proj)
    proj *= math.sqrt(2.0 / W.shape[1])
    return proj


def train_linear(Phi, y, C, rng):
    """Train linear softmax with Adam. Returns W2 [D, C], b2 [C]."""
    n, D = Phi.shape
    W2 = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            phi, yy = Phi[idx], y[idx]
            logits = phi @ W2 + b2
            g = softmax_grad(logits, yy)
            grads = [phi.T @ g + L2 * W2, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W2, b2


def evaluate(Phi, y, W2, b2):
    logits = Phi @ W2 + b2
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]

    # Z-score encoding on full data (stats from full train, but we'll recompute per split)
    # Split: last 20% tuning, first 80% train-proper
    n = X.shape[0]
    split = int(n * 0.8)
    idx_train, idx_tune = np.arange(split), np.arange(split, n)
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_te, y_te = X[idx_tune], y[idx_tune]

    # Encode train-proper to get stats
    X_tr_enc, mean, inv_std = zscore_encode(X_tr.astype(np.float64), types)
    # Apply same stats to tuning
    X_te_enc = (X_te.astype(np.float64) - mean) * inv_std
    X_te_enc = X_te_enc.astype(np.float32)

    best_D, best_acc = D_CANDIDATES[0], -1.0
    best_W = best_b = best_W2 = best_b2 = None

    for D in D_CANDIDATES:
        rng = np.random.default_rng(SEED + D * 1000)  # deterministic per D
        # RFF params: W ~ N(0, gamma), gamma = 1/(2*sigma^2). Heuristic: median distance?
        # Use gamma = 1.0 / d (default RBF scale)
        gamma = 1.0 / X_tr_enc.shape[1]
        W = (rng.normal(0, math.sqrt(2 * gamma), (X_tr_enc.shape[1], D))).astype(np.float32)
        b = (rng.uniform(0, 2 * math.pi, D)).astype(np.float32)

        Phi_tr = rff_features(X_tr_enc, W, b)
        Phi_te = rff_features(X_te_enc, W, b)

        rng2 = np.random.default_rng(SEED + D * 1000 + 1)
        W2, b2 = train_linear(Phi_tr, y_tr, C, rng2)
        acc = evaluate(Phi_te, y_te, W2, b2)

        if acc > best_acc:
            best_acc, best_D = acc, D
            best_W, best_b, best_W2, best_b2 = W, b, W2, b2

    # Retrain on ALL data with best_D
    rng = np.random.default_rng(SEED + best_D * 1000)
    gamma = 1.0 / X.shape[1]
    W = (rng.normal(0, math.sqrt(2 * gamma), (X.shape[1], best_D))).astype(np.float32)
    b = (rng.uniform(0, 2 * math.pi, best_D)).astype(np.float32)

    X_all_enc, mean, inv_std = zscore_encode(X.astype(np.float64), types)
    Phi_all = rff_features(X_all_enc, W, b)

    rng2 = np.random.default_rng(SEED + best_D * 1000 + 1)
    W2, b2 = train_linear(Phi_all, y, C, rng2)

    # Pack weights
    # magic, d, D, C
    hdr = struct.pack("<4i", 0x52464631, X.shape[1], best_D, C)
    out = hdr + mean.tobytes() + inv_std.tobytes() + W.tobytes() + b.tobytes() + W2.tobytes() + b2.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-rff inference: ./infer weights.bin in.rec N
 * Input record: d bytes (0..255). Output: 1 byte class index.
 * Weights format (little-endian):
 *   int32 magic=0x52464631, d, D, C
 *   float mean[d], inv_std[d], W[d][D], b[D], W2[D][C], b2[C]  (all row-major)
 * Computation: x_norm = (x - mean) * inv_std; phi = sqrt(2/D)*cos(x_norm @ W + b); logits = phi @ W2 + b2; argmax. */
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
    if (nw < 16 || hd[0] != 0x52464631) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d < 1 || D < 1 || C < 1 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * (d + d + (long)d * D + D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *inv_std = mean + d;
    const float *W = inv_std + d;
    const float *b = W + (long)d * D;
    const float *W2 = b + D;
    const float *b2 = W2 + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *xnorm = malloc(sizeof(float) * d);
    float *phi = malloc(sizeof(float) * D);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize: xnorm[j] = (xi[j] - mean[j]) * inv_std[j]
        for (int j = 0; j < d; j++) {
            xnorm[j] = (xi[j] - mean[j]) * inv_std[j];
        }

        // RFF projection: phi = sqrt(2/D) * cos(xnorm @ W + b)
        float scale = sqrtf(2.0f / D);
        for (int u = 0; u < D; u++) {
            float sum = b[u];
            for (int j = 0; j < d; j++) {
                sum += xnorm[j] * W[(long)j * D + u];
            }
            phi[u] = scale * cosf(sum);
        }

        // Linear head: logits = phi @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < D; u++) {
                sum += phi[u] * W2[(long)u * C + c];
            }
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```