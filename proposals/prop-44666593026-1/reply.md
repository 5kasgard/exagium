```hypothesis
Family: label-poly2 (quadratic polynomial logistic regression).
Why: Optdigits pixels are ordinal 0..16; pairwise interactions capture stroke crossings and curves that linear models miss.
A full quadratic expansion (2144 features) remains tiny and fast to evaluate.
Failure: If dev accuracy <= 0.9613 (label-linear) or CPU cost exceeds label-mlp16.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
BATCH = 64

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def expand(X):
    """X: (n, 64) float32 in [0,1] -> (n, 2144) float32"""
    n, d = X.shape
    num_quad = d * (d + 1) // 2
    num_feat = d + num_quad
    Z = np.empty((n, num_feat), dtype=np.float32)
    Z[:, :d] = X
    col = d
    for i in range(d):
        xi = X[:, i]
        for j in range(i, d):
            Z[:, col] = xi * X[:, j]
            col += 1
    return Z

def softmax_grad(logits, y):
    """logits: (batch, C), y: (batch,) -> gradient (batch, C)"""
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()

def train_model(X, y, C, lr, l2, epochs, rng):
    n, num_feat = X.shape
    # Xavier init
    W = (rng.uniform(-1, 1, (num_feat, C)) * math.sqrt(6.0 / num_feat)).astype(np.float32)
    b = np.zeros(C, np.float32)
    # Adam state
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            logits = xb @ W + b
            g = softmax_grad(logits, yb)  # (batch, C)
            grad_W = xb.T @ g + l2 * W
            grad_b = g.sum(axis=0)
            t += 1
            # Adam update
            mW = beta1 * mW + (1 - beta1) * grad_W
            vW = beta2 * vW + (1 - beta2) * grad_W * grad_W
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

            mb = beta1 * mb + (1 - beta1) * grad_b
            vb = beta2 * vb + (1 - beta2) * grad_b * grad_b
            mb_hat = mb / (1 - beta1 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
    return W, b

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 64

    # Normalize to [0,1] (pixel counts 0..16)
    X = X_bytes.astype(np.float32) / 16.0

    # Expand features
    Z = expand(X)
    n, num_feat = Z.shape
    assert num_feat == 2144

    # Tuning split: last 20%
    tune_size = int(0.2 * n)
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)

    Z_train, y_train = Z[train_idx], y[train_idx]
    Z_tune, y_tune = Z[tune_idx], y[tune_idx]

    # Hyperparameter grid
    lrs = [0.01, 0.03, 0.1]
    l2s = [1e-4, 1e-3, 1e-2]
    epochs_list = [10, 20, 30]

    best_acc = -1.0
    best_params = None
    best_W, best_b = None, None

    for lr in lrs:
        for l2 in l2s:
            for epochs in epochs_list:
                rng = np.random.default_rng(SEED)
                W, b = train_model(Z_train, y_train, C, lr, l2, epochs, rng)
                logits_tune = Z_tune @ W + b
                acc = accuracy(logits_tune, y_tune)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (lr, l2, epochs)
                    best_W, best_b = W.copy(), b.copy()

    print(f"Best tuning acc: {best_acc:.4f} with lr={best_params[0]}, l2={best_params[1]}, epochs={best_params[2]}", file=sys.stderr)

    # Retrain on full data
    rng = np.random.default_rng(SEED)
    W, b = train_model(Z, y, C, *best_params, rng)

    # Save weights
    # Magic: 0x504F4C32 ("POL2"), then d, num_feat, C
    header = struct.pack("<4i", 0x504F4C32, d, num_feat, C)
    # Weights: feature-major (num_feat x C) float32
    weights_bytes = best_W.astype("<f4").tobytes()
    bias_bytes = best_b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + weights_bytes + bias_bytes)

if __name__ == "__main__":
    main()
```

```c
/* Quadratic polynomial logistic regression for label tasks:
 * ./infer weights.bin in.rec N
 * Each input record is d bytes (0..16); output is one byte, the argmax class. */
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
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], num_feat = hd[2], C = hd[3];
    if (d != 64 || num_feat != 2144 || C != 10) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 16 + 4L * num_feat * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W = (const float *)(hd + 4);
    const float *b = W + num_feat * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[10];
    float xf[64];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) xf[j] = xi[j] / 16.0f;

        for (int c = 0; c < C; c++) scores[c] = b[c];

        int f = 0;
        // Linear features
        for (int j = 0; j < d; j++) {
            float v = xf[j];
            for (int c = 0; c < C; c++) scores[c] += v * W[f * C + c];
            f++;
        }
        // Quadratic features
        for (int i1 = 0; i1 < d; i1++) {
            for (int i2 = i1; i2 < d; i2++) {
                float v = xf[i1] * xf[i2];
                for (int c = 0; c < C; c++) scores[c] += v * W[f * C + c];
                f++;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```