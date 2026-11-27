```hypothesis
Family: label-embedmlp. Position-specific byte embeddings (dim D) are summed, then passed through a ReLU hidden layer (dim H) to the output. This captures pairwise interactions between context positions that a linear model cannot, while keeping inference cost O(D*H + H*C) — far below label-gbdt. We expect dev accuracy > 0.34. If it fails to beat label-linear (0.3375), the task gains little from non-linear position interactions or the chosen D/H are too small.
```

```python
"""label-embedmlp: summed positional embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: magic 0x454D424D, d, D, H, C; then embeddings[d][256][D], W1[D][H], b1[H], W2[H][C], b2[C] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
# Hyperparameter grid searched on tuning split (last 20% of train)
GRID = [(16, 32), (16, 64), (32, 32), (32, 64)]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def split_train_tune(X, y, tune_frac=0.2):
    n = X.shape[0]
    tune_n = int(n * tune_frac)
    return X[:-tune_n], y[:-tune_n], X[-tune_n:], y[-tune_n:]

def init_params(d, D, H, C, rng):
    # embeddings: d x 256 x D
    emb = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / (256 * D))).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (D, H)) * math.sqrt(6.0 / D)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return [emb, W1, b1, W2, b2]

def forward(params, X, d, D, H, C):
    emb, W1, b1, W2, b2 = params
    n = X.shape[0]
    # sum embeddings: n x D
    summed = np.zeros((n, D), np.float32)
    for j in range(d):
        summed += emb[j, X[:, j]]
    # hidden layer
    hidden = np.maximum(summed @ W1 + b1, 0.0)
    # output
    logits = hidden @ W2 + b2
    return logits, hidden, summed

def loss_grad(logits, y, C):
    n = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n
    loss = -np.log(np.maximum(probs[np.arange(n), y] + 1.0, 1e-12)).mean()
    return loss, probs.astype(np.float32)

def backward(params, X, y, summed, hidden, grad_out, d, D, H, C, L2):
    emb, W1, b1, W2, b2 = params
    n = X.shape[0]
    # grad_out: n x C
    grad_W2 = hidden.T @ grad_out + L2 * W2
    grad_b2 = grad_out.sum(0)
    grad_hidden = grad_out @ W2.T
    grad_hidden[hidden == 0] = 0.0  # ReLU derivative
    grad_W1 = summed.T @ grad_hidden + L2 * W1
    grad_b1 = grad_hidden.sum(0)
    grad_summed = grad_hidden @ W1.T
    # embeddings gradient
    grad_emb = np.zeros((d, 256, D), np.float32)
    for j in range(d):
        idx = X[:, j]
        for i in range(n):
            grad_emb[j, idx[i]] += grad_summed[i]
    grad_emb += L2 * emb
    return [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

def adam_update(params, grads, state, t, lr):
    for i, (p, g) in enumerate(zip(params, grads)):
        m, v = state[i]
        m *= 0.9; m += 0.1 * g
        v *= 0.999; v += 0.001 * g * g
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

def accuracy(params, X, y, d, D, H, C):
    logits, _, _ = forward(params, X, d, D, H, C)
    pred = logits.argmax(1)
    return (pred == y).mean()

def train_model(X, y, d, D, H, C, rng, epochs, batch_size):
    params = init_params(d, D, H, C, rng)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    n = X.shape[0]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            xi, yi = X[idx], y[idx]
            logits, hidden, summed = forward(params, xi, d, D, H, C)
            _, grad_out = loss_grad(logits, yi, C)
            grads = backward(params, xi, yi, summed, hidden, grad_out, d, D, H, C, L2)
            t += 1
            adam_update(params, grads, state, t, LR)
    return params

def evaluate_grid(X_tr, y_tr, X_va, y_va, d, C):
    rng = np.random.default_rng(SEED)
    best_acc, best_params, best_DH = -1, None, None
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))
    for D, H in GRID:
        # Use a fresh RNG stream for each config to keep deterministic
        cfg_rng = np.random.default_rng(SEED + D * 1000 + H)
        params = train_model(X_tr, y_tr, d, D, H, C, cfg_rng, epochs, BATCH)
        acc = accuracy(params, X_va, y_va, d, D, H, C)
        if acc > best_acc:
            best_acc, best_params, best_DH = acc, params, (D, H)
    return best_params, best_DH

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)

    # Split tuning set (last 20%)
    X_tr, y_tr, X_va, y_va = split_train_tune(X, y, 0.2)

    # Grid search on tuning split
    best_params, (D, H) = evaluate_grid(X_tr, y_tr, X_va, y_va, d, C)

    # Retrain on full data with best hyperparameters
    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X)))
    final_params = train_model(X, y, d, D, H, C, rng, epochs, BATCH)

    # Pack weights
    emb, W1, b1, W2, b2 = final_params
    magic = 0x454D424D  # 'EMBM'
    header = struct.pack("<5i", magic, d, D, H, C)
    out = header + emb.astype("<f4").tobytes() + W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes() \
          + W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embedmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte (class index). */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * D + (long)D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * D;
    const float *b1 = W1 + (long)D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *summed = malloc(sizeof(float) * D);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < D; j++) summed[j] = 0.0f;
        for (int p = 0; p < d; p++) {
            const float *e = emb + ((long)p * 256 + xi[p]) * D;
            for (int j = 0; j < D; j++) summed[j] += e[j];
        }

        for (int h = 0; h < H; h++) {
            float v = b1[h];
            for (int j = 0; j < D; j++) v += summed[j] * W1[(long)j * H + h];
            hidden[h] = v > 0.0f ? v : 0.0f;
        }

        for (int c = 0; c < C; c++) {
            float v = b2[c];
            for (int h = 0; h < H; h++) v += hidden[h] * W2[(long)h * C + c];
            logits[c] = v;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```