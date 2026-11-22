```hypothesis
Family: label-rbf (radial basis function network). The 8x8 digit images have smooth spatial variation; RBF centers can capture local patterns. A small set of centers (k~30) with a linear readout should exceed linear accuracy while keeping inference cost moderate (k*d distance calculations). If dev accuracy does not surpass label-linear (0.9613) or cost exceeds label-mlp16 (8685 Ir/item), the approach fails.
```

```python
"""RBF network family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Trains k-means centers on raw pixel values (0..16), then logistic regression on RBF features.
Hyperparameters (k, gamma) chosen on last 20% of train.in (tuning split)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
K_GRID = [10, 20, 30, 40, 50]
GAMMA_GRID = [0.05, 0.1, 0.2, 0.5, 1.0]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def kmeans(X, k, seed, max_iter=100):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), k, replace=False)
    centers = X[idx].astype(np.float32)
    for _ in range(max_iter):
        diff = X[:, None, :] - centers[None, :, :]
        dist2 = (diff ** 2).sum(axis=2)
        labels = dist2.argmin(axis=1)
        new_centers = np.zeros_like(centers)
        counts = np.zeros(k, dtype=np.int32)
        for i in range(len(X)):
            c = labels[i]
            new_centers[c] += X[i]
            counts[c] += 1
        for c in range(k):
            if counts[c] > 0:
                new_centers[c] /= counts[c]
        if np.allclose(centers, new_centers):
            centers = new_centers
            break
        centers = new_centers
    return centers


def rbf_features(X, centers, gamma):
    diff = X[:, None, :] - centers[None, :, :]
    dist2 = (diff ** 2).sum(axis=2)
    return np.exp(-gamma * dist2).astype(np.float32)


def train_logistic(Z, y, C, seed):
    n, k = Z.shape
    rng = np.random.default_rng(seed)
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_logits = (probs / len(yy)).astype(np.float32)
            grad_W = z.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = len(X)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    best_acc, best_k, best_gamma = -1.0, 0, 0.0
    for k in K_GRID:
        for gamma in GAMMA_GRID:
            seed = 1000 + k * 10 + int(gamma * 100)
            centers = kmeans(X_train.astype(np.float32), k, seed)
            Z_train = rbf_features(X_train.astype(np.float32), centers, gamma)
            Z_val = rbf_features(X_val.astype(np.float32), centers, gamma)
            W, b = train_logistic(Z_train, y_train, C, seed)
            logits = Z_val @ W + b
            pred = logits.argmax(axis=1)
            acc = (pred == y_val).mean()
            if acc > best_acc:
                best_acc, best_k, best_gamma = acc, k, gamma

    # Retrain on full data with best hyperparameters
    centers = kmeans(X.astype(np.float32), best_k, SEED)
    Z = rbf_features(X.astype(np.float32), centers, best_gamma)
    W, b = train_logistic(Z, y, C, SEED)

    # Write weights.bin: magic, d, k, C, gamma, centers, W, b
    magic = 0x52424631  # 'RBF1'
    out = struct.pack("<4i f", magic, d, best_k, C, best_gamma)
    out += centers.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* RBF network inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class. */
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
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x52424631) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    float gamma = *(float *)(hd + 4);
    const float *centers = (const float *)(hd + 5);
    const float *W = centers + (long)k * d;
    const float *b = W + (long)k * C;
    long expected = 20 + 4L * ((long)k * d + (long)k * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *act = malloc(sizeof(float) * k);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        for (int c = 0; c < k; c++) {
            float sum = 0.0f;
            const float *cen = centers + (long)c * d;
            for (int j = 0; j < d; j++) {
                float diff = (float)xi[j] - cen[j];
                sum += diff * diff;
            }
            act[c] = expf(-gamma * sum);
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int u = 0; u < k; u++) {
            float a = act[u];
            const float *wu = W + (long)u * C;
            for (int c = 0; c < C; c++) scores[c] += a * wu[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```