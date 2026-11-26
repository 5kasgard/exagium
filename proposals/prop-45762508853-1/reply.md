```hypothesis
Family: label-rbfnet (RBF network with k-means centers + linear readout).
Why: Optdigits has 64 ordinal pixel features (0–16) with local spatial structure. A small set of RBF centers captures non-linear manifolds better than a linear model, while staying far cheaper than k-NN or deep nets. We fix input scaling (÷16), pick K and γ on the 20% tuning split, then train on all data.
Failure signal: If dev accuracy ≤ 0.9613 (label-linear) or Ir/item > 5000, the family does not meet the goal.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters chosen after a quick manual tuning on the tuning split
SEED = 1
K = 64          # number of RBF centers
GAMMA = 0.5     # RBF width
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans(X, K, rng, max_iter=20):
    """Deterministic k-means (Lloyd) with k-means++ init."""
    n, d = X.shape
    cents = np.empty((K, d), dtype=np.float32)
    # k-means++ initialization
    idx = rng.integers(n)
    cents[0] = X[idx]
    for k in range(1, K):
        dist2 = ((X[:, None] - cents[:k][None]) ** 2).sum(axis=2).min(axis=1)
        probs = dist2 / dist2.sum()
        idx = rng.choice(n, p=probs)
        cents[k] = X[idx]
    # Lloyd iterations
    for _ in range(max_iter):
        dist2 = ((X[:, None] - cents[None]) ** 2).sum(axis=2)
        labels = dist2.argmin(axis=1)
        new_cents = np.zeros_like(cents)
        counts = np.zeros(K, dtype=np.int32)
        for i in range(n):
            new_cents[labels[i]] += X[i]
            counts[labels[i]] += 1
        for k in range(K):
            if counts[k] > 0:
                new_cents[k] /= counts[k]
            else:
                # re-initialize empty cluster
                idx = rng.integers(n)
                new_cents[k] = X[idx]
        if np.allclose(cents, new_cents):
            break
        cents = new_cents
    return cents.astype(np.float32)


def rbf_features(X, centers, gamma):
    """X: (n, d) float32 in [0,1], centers: (K, d) float32 -> (n, K) float32."""
    # squared Euclidean distance via broadcasting
    diff = X[:, None, :] - centers[None, :, :]   # (n, K, d)
    dist2 = (diff * diff).sum(axis=2)            # (n, K)
    return np.exp(-gamma * dist2).astype(np.float32)


def train_logreg(Z, y, C, rng):
    """Adam on softmax cross-entropy, returns (W, b). Z: (n, K) float32."""
    n, K = Z.shape
    W = (rng.uniform(-1, 1, (K, C)) * math.sqrt(6.0 / K)).astype(np.float32)
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
            # forward
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            # gradient
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = probs / len(yy)          # (batch, C)
            grad_W = z.T @ grad + L2 * W    # (K, C)
            grad_b = grad.sum(axis=0)       # (C,)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W, b


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 64

    # Scale to [0,1]
    X = (X_bytes.astype(np.float32) / 16.0)

    # Split: last 20% for tuning
    n = X.shape[0]
    tune_size = int(0.2 * n)
    train_X, tune_X = X[:-tune_size], X[-tune_size:]
    train_y, tune_y = y[:-tune_size], y[-tune_size:]

    rng = np.random.default_rng(SEED)

    # Grid search on tuning split (small grid, deterministic)
    best_acc = -1.0
    best_centers = None
    best_W = None
    best_b = None

    for K_try in (32, 64, 128):
        for GAMMA_try in (0.25, 0.5, 1.0):
            centers = kmeans(train_X, K_try, rng)
            Z_train = rbf_features(train_X, centers, GAMMA_try)
            Z_tune = rbf_features(tune_X, centers, GAMMA_try)
            W, b = train_logreg(Z_train, train_y, C, rng)
            # evaluate on tuning
            logits = Z_tune @ W + b
            preds = logits.argmax(axis=1)
            acc = (preds == tune_y).mean()
            if acc > best_acc:
                best_acc = acc
                best_centers = centers
                best_W = W
                best_b = b
                best_K = K_try
                best_GAMMA = GAMMA_try

    # Retrain on all data with best hyperparameters
    centers = kmeans(X, best_K, rng)
    Z_all = rbf_features(X, centers, best_GAMMA)
    W, b = train_logreg(Z_all, y, C, rng)

    # Pack weights: magic, d, K, C, gamma, centers[K][d], W[K][C], b[C]
    magic = 0x5242464E  # 'RBFN'
    header = struct.pack("<4if", magic, d, best_K, C, best_GAMMA)
    body = centers.astype("<f4").tobytes() + W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* RBF network inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (uint8 0..16). Output record: 1 byte class 0..9. */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }

    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x5242464E) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = wb[4] | wb[5] << 8 | wb[6] << 16 | wb[7] << 24;
    int K = wb[8] | wb[9] << 8 | wb[10] << 16 | wb[11] << 24;
    int C = wb[12] | wb[13] << 8 | wb[14] << 16 | wb[15] << 24;
    float gamma;
    memcpy(&gamma, wb + 16, 4);

    if (d != 64 || C < 1 || C > 256 || K < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20 + (long)K * d * 4 + (long)K * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *centers = (const float *)(wb + 20);
    const float *W = centers + (long)K * d;
    const float *b = W + (long)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *rbf = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);
    float *x_f = malloc(sizeof(float) * d);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // scale to [0,1]
        for (int j = 0; j < d; j++) x_f[j] = xi[j] / 16.0f;

        // RBF features
        for (int c = 0; c < K; c++) {
            float dist2 = 0.0f;
            const float *cent = centers + (long)c * d;
            for (int j = 0; j < d; j++) {
                float diff = x_f[j] - cent[j];
                dist2 += diff * diff;
            }
            rbf[c] = expf(-gamma * dist2);
        }

        // linear readout
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int k = 0; k < K; k++) {
            float rv = rbf[k];
            const float *wk = W + (long)k * C;
            for (int c = 0; c < C; c++) scores[c] += rv * wk[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```