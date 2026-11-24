```hypothesis
Family: label-rff (Random Fourier Features). Maps 64-d pixel vectors to D random Fourier features (cosine projections) then linear classification. RBF-like non-linearity should capture digit shapes better than linear. Hyperparameters D and kernel bandwidth gamma chosen on tuning split. Expected to beat label-linear (0.961) and approach label-mlp16 (0.977) with lower cost than label-gbdt. Failure if dev accuracy <= 0.961 or instructions/item > label-mlp16.
```

```python
"""label-rff: Random Fourier Features + linear classifier for optdigits.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encoding: per-feature z-score (ordinal). RFF: sqrt(2/D)*cos(W*x + b), W~N(0,gamma), b~U(0,2π).
Linear layer trained with Adam (lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n))).
Hyperparameters D, gamma chosen on last 20% of train.in (tuning split), then retrain on all.
Weights format (little-endian): magic 0x52464631, d, D, C; float32 mean[d], std[d], W[D*d], b[D], W2[D*C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
D_CANDIDATES = [64, 128, 256, 512]
GAMMA_CANDIDATES = [0.1, 0.5, 1.0, 2.0]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_linear(Z, y, C, rng, n_epochs):
    """Train linear softmax with Adam on features Z (n, D). Returns W2 (D, C), b2 (C)."""
    n, D = Z.shape
    W2 = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            S = z @ W2 + b2
            S = S - S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W2, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params[0], params[1]


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X_bytes.shape[1]
    C = card["classes"]
    n = X_bytes.shape[0]
    tune_size = int(0.2 * n)
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)

    # per-feature z-score on full train (for final) but for tuning we use train split only
    def compute_stats(X):
        mu = X.mean(axis=0).astype(np.float32)
        sd = X.std(axis=0).astype(np.float32)
        sd[sd == 0] = 1.0
        return mu, sd

    best_acc = -1.0
    best_D = D_CANDIDATES[0]
    best_gamma = GAMMA_CANDIDATES[0]
    best_mu = None
    best_sd = None

    for D in D_CANDIDATES:
        for gamma in GAMMA_CANDIDATES:
            # deterministic seed per hyperparameter combo
            rng = np.random.default_rng(1000 + D * 10 + int(gamma * 10))
            mu, sd = compute_stats(X_bytes[train_idx])
            X_train_norm = (X_bytes[train_idx].astype(np.float32) - mu) / sd
            X_tune_norm = (X_bytes[tune_idx].astype(np.float32) - mu) / sd
            # generate RFF parameters
            W_rff = (rng.normal(0, math.sqrt(gamma), (D, d))).astype(np.float32)
            b_rff = (rng.uniform(0, 2 * math.pi, D)).astype(np.float32)
            # compute features
            Z_train = np.cos(X_train_norm @ W_rff.T + b_rff) * math.sqrt(2.0 / D)
            Z_tune = np.cos(X_tune_norm @ W_rff.T + b_rff) * math.sqrt(2.0 / D)
            # train linear
            n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(train_idx)))
            W2, b2 = train_linear(Z_train, y[train_idx], C, rng, n_epochs)
            # evaluate
            S = Z_tune @ W2 + b2
            pred = S.argmax(axis=1)
            acc = (pred == y[tune_idx]).mean()
            if acc > best_acc:
                best_acc = acc
                best_D = D
                best_gamma = gamma
                best_mu = mu
                best_sd = sd

    # retrain on all data with best hyperparameters
    rng = np.random.default_rng(SEED)
    mu, sd = best_mu, best_sd
    X_norm = (X_bytes.astype(np.float32) - mu) / sd
    W_rff = (rng.normal(0, math.sqrt(best_gamma), (best_D, d))).astype(np.float32)
    b_rff = (rng.uniform(0, 2 * math.pi, best_D)).astype(np.float32)
    Z = np.cos(X_norm @ W_rff.T + b_rff) * math.sqrt(2.0 / best_D)
    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    W2, b2 = train_linear(Z, y, C, rng, n_epochs)

    # pack weights
    # magic, d, D, C
    out = struct.pack("<4i", 0x52464631, d, best_D, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += W_rff.astype("<f4").tobytes()
    out += b_rff.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-rff inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixel counts 0..16). Output: 1 byte class 0..9.
 * Weights format: magic 0x52464631, int32 d, D, C; float32 mean[d], std[d], W[D*d], b[D], W2[D*C], b2[C]. */
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
    if (d != 64 || C != 10 || D <= 0 || D > 1024) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * (2L * d + (long)D * d + D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std  = mean + d;
    const float *W    = std + d;
    const float *b    = W + (long)D * d;
    const float *W2   = b + D;
    const float *b2   = W2 + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * D);
    float *score = malloc(sizeof(float) * C);
    float *xnorm = malloc(sizeof(float) * d);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) xnorm[j] = (xi[j] - mean[j]) / std[j];
        for (int u = 0; u < D; u++) {
            float sum = b[u];
            const float *Wu = W + (long)u * d;
            for (int j = 0; j < d; j++) sum += Wu[j] * xnorm[j];
            feat[u] = cosf(sum) * sqrtf(2.0f / D);
        }
        for (int c = 0; c < C; c++) score[c] = b2[c];
        for (int u = 0; u < D; u++) {
            float f = feat[u];
            const float *W2u = W2 + (long)u * C;
            for (int c = 0; c < C; c++) score[c] += f * W2u[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```