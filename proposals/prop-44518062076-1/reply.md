```hypothesis
Family: label-embed (per-position byte embeddings + one ReLU layer).
Why: The task is character-level language modeling with 16-byte context. A linear one-hot model (label-linear) ignores interactions between positions; embeddings allow sharing statistics across positions and a hidden layer captures non-linear combinations. This should exceed label-linear's 33.75% accuracy while keeping inference fast (≈20k FLOPs/item).
Failure signal: If dev accuracy ≤ 0.3375, the task is predominantly linear in one-hot space and embeddings add no value.
```

```python
import json, math, struct, sys

import numpy as np

# Global seed for full determinism
GLOBAL_SEED = 1
# Hyperparameter grid (D, H) to search on the tuning split (last 20% of train.in)
GRID = [(32, 64), (64, 128), (128, 256)]
# Training hyperparameters
LR = 1e-3
L2 = 1e-5
BATCH = 128
EPOCHS = 30
TUNING_FRAC = 0.2

MAGIC = 0x44424D45  # 'EMBD' in little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, D, H, seed):
    n, d = X.shape
    C = 65
    rng = np.random.default_rng(seed)

    # He initialization for embeddings and weights
    E = rng.normal(0, 0.01, size=(d, 256, D)).astype(np.float32)
    W1 = rng.normal(0, math.sqrt(2.0 / D), size=(D, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, math.sqrt(2.0 / H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [E, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            bs = len(idx)

            # Forward
            v_emb = np.zeros((bs, D), np.float32)
            for pos in range(d):
                v_emb += E[pos, xb[:, pos]]
            h_pre = v_emb @ W1 + b1
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(bs), yb] -= 1.0
            grad_logits /= bs

            # Backprop
            grad_W2 = h.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)
            grad_h = grad_logits @ W2.T
            grad_h_pre = grad_h * (h_pre > 0)
            grad_W1 = v_emb.T @ grad_h_pre
            grad_b1 = grad_h_pre.sum(axis=0)
            grad_v_emb = grad_h_pre @ W1.T

            grad_E = np.zeros_like(E)
            for pos in range(d):
                np.add.at(grad_E[pos], xb[:, pos], grad_v_emb)

            grads = [grad_E, grad_W1, grad_b1, grad_W2, grad_b2]

            # L2 regularization on weight matrices (not biases)
            for i, p in enumerate(params):
                if p.ndim > 1:
                    grads[i] += L2 * p

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    return params


def evaluate(X, y, params):
    E, W1, b1, W2, b2 = params
    n, d = X.shape
    v_emb = np.zeros((n, W1.shape[0]), np.float32)
    for pos in range(d):
        v_emb += E[pos, X[:, pos]]
    h = np.maximum(v_emb @ W1 + b1, 0)
    logits = h @ W2 + b2
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]

    # Split: last 20% for tuning
    tune_size = int(n * TUNING_FRAC)
    train_size = n - tune_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_tune, y_tune = X[train_size:], y[train_size:]

    # Deterministic grid search
    main_rng = np.random.default_rng(GLOBAL_SEED)
    best_acc = -1.0
    best_D, best_H = GRID[0]

    for D, H in GRID:
        config_seed = int(main_rng.integers(2**32))
        params = train_model(X_train, y_train, D, H, config_seed)
        acc = evaluate(X_tune, y_tune, params)
        if acc > best_acc:
            best_acc = acc
            best_D, best_H = D, H

    # Final training on all data with a fixed seed
    final_seed = GLOBAL_SEED + 1
    final_params = train_model(X, y, best_D, best_H, final_seed)
    E, W1, b1, W2, b2 = final_params

    # Write weights.bin
    header = struct.pack("<5i", MAGIC, d, best_D, best_H, C)
    body = (
        E.astype("<f4").tobytes() +
        W1.astype("<f4").tobytes() +
        b1.astype("<f4").tobytes() +
        W2.astype("<f4").tobytes() +
        b2.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: per-position byte embeddings + one ReLU layer */
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
    if (nw < 20 || hd[0] != 0x44424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * D + (long)D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E = (const float *)(hd + 5);
    const float *W1 = E + (long)d * 256 * D;
    const float *b1 = W1 + (long)D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *v = (float *)malloc(sizeof(float) * D);
    float *h = (float *)malloc(sizeof(float) * H);
    float *logits = (float *)malloc(sizeof(float) * C);
    if (!v || !h || !logits) { fprintf(stderr, "oom\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(v, 0, sizeof(float) * D);
        for (int pos_i = 0; pos_i < d; pos_i++) {
            const float *emb = E + ((long)pos_i * 256 + xi[pos_i]) * D;
            for (int j = 0; j < D; j++) v[j] += emb[j];
        }

        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int j = 0; j < D; j++) sum += v[j] * W1[(long)j * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```