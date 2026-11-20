```hypothesis
We propose "label-poly2": a degree-2 polynomial expansion of the 64 pixel features followed by a linear multi-class classifier. The optdigits pixels are ordinal (0-16) and likely have quadratic interactions (e.g., stroke thickness, curvature). A linear model on polynomial features can capture such nonlinearities while remaining fast at inference (2144 features, 10 classes ~21k multiply-adds per item). We tune hyperparameters on the last 20% of training data. If dev accuracy does not exceed label-linear (0.9613) or inference cost exceeds label-mlp16 (8685 Ir/item), the approach fails.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
GRID = [
    {"lr": 0.01, "l2": 1e-4, "epochs": 50},
    {"lr": 0.03, "l2": 1e-4, "epochs": 50},
    {"lr": 0.1, "l2": 1e-4, "epochs": 50},
    {"lr": 0.03, "l2": 1e-3, "epochs": 50},
    {"lr": 0.03, "l2": 1e-2, "epochs": 50},
    {"lr": 0.03, "l2": 1e-4, "epochs": 100},
]
BATCH = 64


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def expand(Z):
    n, d = Z.shape
    num_feat = d + d * (d + 1) // 2
    Phi = np.empty((n, num_feat), dtype=np.float32)
    Phi[:, :d] = Z
    idx = d
    for i in range(d):
        for j in range(i, d):
            Phi[:, idx] = Z[:, i] * Z[:, j]
            idx += 1
    return Phi


def train_linear(Phi, y, C, lr, l2, epochs, batch, rng):
    n, k = Phi.shape
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    beta1, beta2 = 0.9, 0.999
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            x, yy = Phi[idx], y[idx]
            logits = x @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_logits = (probs / len(yy)).astype(np.float32)
            gW = x.T @ grad_logits + l2 * W
            gb = grad_logits.sum(axis=0)
            t += 1
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * gW * gW
            mb = beta1 * mb + (1 - beta1) * gb
            vb = beta2 * vb + (1 - beta2) * gb * gb
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            mb_hat = mb / (1 - beta1 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            b -= lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)
    return W, b


def accuracy(Phi, y, W, b):
    logits = Phi @ W + b
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 64
    n = X.shape[0]
    tune_start = int(0.8 * n)
    X_train, X_tune = X[:tune_start], X[tune_start:]
    y_train, y_tune = y[:tune_start], y[tune_start:]

    mean = X_train.mean(axis=0).astype(np.float64)
    std = X_train.std(axis=0).astype(np.float64)
    std[std == 0] = 1.0

    def encode(X):
        return ((X.astype(np.float64) - mean) / std).astype(np.float32)

    Z_train = encode(X_train)
    Z_tune = encode(X_tune)
    Phi_train = expand(Z_train)
    Phi_tune = expand(Z_tune)

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_W = None
    best_b = None

    for params in GRID:
        W, b = train_linear(Phi_train, y_train, C, params["lr"], params["l2"], params["epochs"], BATCH, rng)
        acc = accuracy(Phi_tune, y_tune, W, b)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_W, best_b = W.copy(), b.copy()

    print(f"Best params: {best_params}, tune acc: {best_acc:.4f}", file=sys.stderr)

    mean_full = X.mean(axis=0).astype(np.float64)
    std_full = X.std(axis=0).astype(np.float64)
    std_full[std_full == 0] = 1.0
    Z_full = ((X.astype(np.float64) - mean_full) / std_full).astype(np.float32)
    Phi_full = expand(Z_full)

    rng = np.random.default_rng(SEED)
    W_final, b_final = train_linear(Phi_full, y, C, best_params["lr"], best_params["l2"], best_params["epochs"], BATCH, rng)

    magic = 0x504F4C59
    num_feat = d + d * (d + 1) // 2
    out = struct.pack("<4i", magic, d, C, num_feat)
    out += np.array(mean_full, dtype=np.float32).tobytes()
    out += np.array(std_full, dtype=np.float32).tobytes()
    out += b_final.astype(np.float32).tobytes()
    out += W_final.astype(np.float32).tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
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
    if (nw < 16 || hd[0] != 0x504F4C59) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], num_feat = hd[3];
    if (d != 64 || C != 10 || num_feat != 2144) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * d + 4L * d + 4L * C + 4L * (long)num_feat * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *bias = std + d;
    const float *W = bias + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *feat = malloc(sizeof(float) * num_feat);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            z[j] = (xi[j] - mean[j]) / std[j];
        }
        int idx = 0;
        for (int j = 0; j < d; j++) feat[idx++] = z[j];
        for (int j = 0; j < d; j++) {
            for (int l = j; l < d; l++) {
                feat[idx++] = z[j] * z[l];
            }
        }
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int f = 0; f < num_feat; f++) {
            const float *w = W + (long)f * C;
            float val = feat[f];
            for (int c = 0; c < C; c++) scores[c] += val * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```