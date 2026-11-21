```hypothesis
Family: label-pcalinear. Optdigits pixels are highly correlated; PCA reduces noise and dimensionality, and a linear model on the top components often outperforms raw linear. The combined model is still a linear model at inference, so cost is similar to label-linear. Failure: if PCA components don't capture discriminative variance, accuracy may not exceed label-linear.
```

```python
"""label-pcalinear: PCA + logistic regression for ordinal byte features.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Chooses number of PCA components on last 20% of train.in, then trains on all data.
Exports a single linear model (with baked-in standardization) for fast inference."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
CANDIDATE_K = [8, 16, 32, 48, 64]   # tried on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_logistic(Z, y, C, rng):
    """Z: (n, k) float32, y: (n,) int64. Returns W (k, C), b (C)."""
    n, k = Z.shape
    W = np.zeros((k, C), np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # softmax gradient
            S = z @ W + b
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W, b


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]
    assert d == 64 and C == 10

    # split: last 20% for tuning (file order)
    split = int(n * 0.8)
    train_idx = np.arange(split)
    tune_idx = np.arange(split, n)

    rng = np.random.default_rng(SEED)

    # standardization on training portion
    X_train = X[train_idx].astype(np.float64)
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    inv_sigma = 1.0 / sigma

    def standardize(Xsub):
        return (Xsub.astype(np.float64) - mu) * inv_sigma

    X_train_std = standardize(X_train).astype(np.float32)
    X_tune_std = standardize(X[tune_idx]).astype(np.float32)
    y_train = y[train_idx]
    y_tune = y[tune_idx]

    best_K, best_acc = 64, -1.0
    for K in CANDIDATE_K:
        # PCA on training standardized data
        U, S, Vt = np.linalg.svd(X_train_std, full_matrices=False)
        Vk = Vt[:K].astype(np.float32)          # (K, d)
        Z_train = X_train_std @ Vk.T            # (n_train, K)
        Z_tune = X_tune_std @ Vk.T

        # train logistic regression
        W_log, b_log = train_logistic(Z_train, y_train, C, np.random.default_rng(SEED + K))

        # evaluate on tuning
        scores = Z_tune @ W_log + b_log
        pred = scores.argmax(axis=1)
        acc = (pred == y_tune).mean()
        if acc > best_acc:
            best_acc, best_K = acc, K

    # retrain on full data with best_K
    X_full_std = standardize(X).astype(np.float32)
    U, S, Vt = np.linalg.svd(X_full_std, full_matrices=False)
    Vk = Vt[:best_K].astype(np.float32)
    Z_full = X_full_std @ Vk.T
    W_log, b_log = train_logistic(Z_full, y, C, np.random.default_rng(SEED))

    # combine into a single linear model on original features:
    # scores = (x - mu) * inv_sigma @ Vk.T @ W_log + b_log
    #        = x @ (Vk.T @ W_log * inv_sigma) + (b_log - mu * inv_sigma @ Vk.T @ W_log)
    combined_W = (Vk.T @ W_log) * inv_sigma.astype(np.float32)[:, None]   # (d, C)
    combined_b = b_log - (mu * inv_sigma).astype(np.float32) @ combined_W  # (C,)

    # save weights: magic, d, K, C, mean[d], inv_sigma[d], combined_W[d][C], combined_b[C]
    magic = 0x5043414C  # "PCAL"
    header = struct.pack("<4i", magic, d, best_K, C)
    body = (
        mu.astype("<f4").tobytes() +
        inv_sigma.astype("<f4").tobytes() +
        combined_W.astype("<f4").tobytes() +
        combined_b.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-pcalinear inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixel counts). Output record: 1 byte (class 0..9).
 * Weights format: magic 0x5043414C, int32 d, K, C; float32 mean[d], inv_sigma[d], W[d][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x5043414C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 64 || C != 10 || K < 1 || K > 64) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + 4L * (2L * d + (long)d * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *inv_sigma = mean + d;
    const float *W = inv_sigma + d;          // row-major: d x C
    const float *b = W + (long)d * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *std = malloc(sizeof(float) * d);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // standardize
        for (int j = 0; j < d; j++) std[j] = (xi[j] - mean[j]) * inv_sigma[j];

        // linear layer
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d; j++) {
            float sj = std[j];
            if (sj == 0.0f) continue;
            const float *wj = W + (long)j * C;
            for (int c = 0; c < C; c++) scores[c] += sj * wj[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```