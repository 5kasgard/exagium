```hypothesis
Family: label-rff (Random Fourier Features). Approximates an RBF kernel with a fixed random projection (cosine features) followed by a linear layer. The 8x8 digit pixels have local spatial structure that a linear model misses; RFF captures non-linearities with only one extra matrix multiply. We fix the random seed for determinism and tune the number of features D on the 20% tuning split. If dev accuracy does not exceed label-linear's 0.9613, the family is not worthwhile.
```
```python
"""label-rff: Random Fourier Features for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each pixel as float in [0,1] (divide by 16). Generates D random Fourier features:
  z = cos(x @ Omega + b)  where Omega[d, D] ~ N(0, 1), b[D] ~ U[0, 2π].
Trains a linear classifier on z with Adam (lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n))).
Chooses D from {128, 256, 512} on the last 20% of train.in (tuning split), then retrains on all train.
Exports weights: magic=0x52464631, d, D, C; float32 Omega[d][D], b[D], W_out[D][C], b_out[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
D_CANDIDATES = [128, 256, 512]  # tuned on the last 20% of train.in


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_linear(Z, y, C, rng, epochs):
    n, D = Z.shape
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            scores = z @ W + b
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W, b


def accuracy(Z, y, W, b):
    scores = Z @ W + b
    preds = scores.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X_bytes.shape[1]
    # normalize to [0,1]
    X = X_bytes.astype(np.float32) / 16.0

    n = X.shape[0]
    split = int(0.8 * n)  # last 20% for tuning
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    # generate a large pool of random features, we'll slice for each D
    # we fix the total max D to generate once
    max_D = max(D_CANDIDATES)
    Omega_pool = rng.normal(0, 1, (d, max_D)).astype(np.float32)
    b_pool = rng.uniform(0, 2 * math.pi, max_D).astype(np.float32)

    best_D = D_CANDIDATES[0]
    best_acc = -1.0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / max(1, split)))

    for D in D_CANDIDATES:
        Omega = Omega_pool[:, :D]
        b = b_pool[:D]
        Z_train = np.cos(X_train @ Omega + b)
        Z_tune = np.cos(X_tune @ Omega + b)
        W, b_out = train_linear(Z_train, y_train, C, np.random.default_rng(SEED), epochs)
        acc = accuracy(Z_tune, y_tune, W, b_out)
        if acc > best_acc:
            best_acc = acc
            best_D = D

    # retrain on full data with best_D
    Omega = Omega_pool[:, :best_D]
    b = b_pool[:best_D]
    Z_full = np.cos(X @ Omega + b)
    W, b_out = train_linear(Z_full, y, C, np.random.default_rng(SEED), min(MAX_EPOCHS, math.ceil(SAMPLES / n)))

    # write weights
    magic = 0x52464631  # 'RFF1'
    out = struct.pack("<4i", magic, d, best_D, C)
    out += Omega.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-rff inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels 0..16). Output record: one byte class index.
 * Weights format: magic(0x52464631), d, D, C; float32 Omega[d][D], b[D], W_out[D][C], b_out[C].
 * Computes z = cos(x/16 * Omega + b), then scores = z * W_out + b_out, argmax. */
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
    if (nw < 16 || hd[0] != 0x52464631) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * ((long)d * D + D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *Omega = (const float *)(hd + 4);
    const float *b = Omega + (long)d * D;
    const float *W_out = b + D;
    const float *b_out = W_out + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // compute z = cos(x/16 * Omega + b)
        for (int j = 0; j < D; j++) z[j] = b[j];
        for (int p = 0; p < d; p++) {
            float xp = xi[p] / 16.0f;
            const float *Om = Omega + (long)p * D;
            for (int j = 0; j < D; j++) z[j] += xp * Om[j];
        }
        for (int j = 0; j < D; j++) z[j] = cosf(z[j]);
        // scores = z * W_out + b_out
        for (int c = 0; c < C; c++) scores[c] = b_out[c];
        for (int j = 0; j < D; j++) {
            float zj = z[j];
            const float *Wc = W_out + (long)j * C;
            for (int c = 0; c < C; c++) scores[c] += zj * Wc[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```