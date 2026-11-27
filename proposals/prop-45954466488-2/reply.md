```hypothesis
Family: label-embmlp. Uses per-position character embeddings (dim E) followed by a ReLU hidden layer (size H) to capture semantic similarities and positional interactions. Beats linear by learning distributed representations; fails if embeddings cannot capture structure or if H too small.
```

```python
"""label-embmlp: per-position embeddings + one ReLU layer for next-char prediction.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Tunes E in {4,8,16} and H in {16,32,64} on last 20% of train, then retrains on all.
Exports weights in label-linear compatible format (magic 0x454D424D)."""
import json, math, struct, sys

import numpy as np

# Tuning grid
E_CHOICES = [4, 8, 16]
H_CHOICES = [16, 32, 64]
# Training hyperparameters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
TUNE_EPOCHS = 10
FULL_EPOCHS = 100
SAMPLES = 600000
MAX_EPOCHS = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


def init_params(d, C, E, H, rng):
    # emb_flat: (d*256, E)
    emb_flat = rng.normal(0, 1.0 / math.sqrt(E), size=(d * 256, E)).astype(np.float32)
    # W1: (d*E, H) He init
    W1 = rng.normal(0, math.sqrt(2.0 / (d * E)), size=(d * E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: (H, C) He init
    W2 = rng.normal(0, math.sqrt(2.0 / H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return emb_flat, W1, b1, W2, b2


def adam_step(params, grads, states, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    for i, (p, g) in enumerate(zip(params, grads)):
        m, v = states[i]
        m *= beta1; m += (1 - beta1) * g
        v *= beta2; v += (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train_epoch(X, y, params, states, d, E, H, C, l2, batch, rng, t_counter):
    emb_flat, W1, b1, W2, b2 = params
    n = X.shape[0]
    order = rng.permutation(n)
    for s in range(0, n, batch):
        idx = order[s:s + batch]
        xb = X[idx]          # (B, d)
        yb = y[idx]          # (B,)
        B = xb.shape[0]
        # flat indices for embedding lookup: (B, d)
        flat_idx = xb + np.arange(d) * 256
        emb_concat = emb_flat[flat_idx].reshape(B, d * E)   # (B, d*E)
        # forward
        hidden = emb_concat @ W1 + b1
        np.maximum(hidden, 0, out=hidden)          # ReLU in-place
        logits = hidden @ W2 + b2
        # loss gradient
        probs = softmax(logits)
        probs[np.arange(B), yb] -= 1.0
        probs /= B
        # backward
        dW2 = hidden.T @ probs
        db2 = probs.sum(axis=0)
        dhidden = probs @ W2.T
        dhidden *= (hidden > 0)                     # ReLU grad
        dW1 = emb_concat.T @ dhidden
        db1 = dhidden.sum(axis=0)
        demb_concat = dhidden @ W1.T                # (B, d*E)
        demb_concat = demb_concat.reshape(B, d, E)  # (B, d, E)
        # scatter into demb_flat
        demb_flat = np.zeros_like(emb_flat)
        np.add.at(demb_flat, flat_idx, demb_concat)
        # L2 regularization
        dW1 += l2 * W1
        dW2 += l2 * W2
        demb_flat += l2 * emb_flat
        # Adam update
        t_counter[0] += 1
        t = t_counter[0]
        grads = [demb_flat, dW1, db1, dW2, db2]
        adam_step(params, grads, states, t, LR)
    return accuracy(logits, yb)


def evaluate(X, y, params, d, E, H, C, batch):
    emb_flat, W1, b1, W2, b2 = params
    n = X.shape[0]
    correct = 0
    for s in range(0, n, batch):
        xb = X[s:s + batch]
        yb = y[s:s + batch]
        B = xb.shape[0]
        flat_idx = xb + np.arange(d) * 256
        emb_concat = emb_flat[flat_idx].reshape(B, d * E)
        hidden = emb_concat @ W1 + b1
        np.maximum(hidden, 0, out=hidden)
        logits = hidden @ W2 + b2
        correct += (logits.argmax(axis=1) == yb).sum()
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(n * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    # grid search
    best_acc = -1
    best_params = None
    best_E = best_H = None
    base_rng = np.random.default_rng(SEED)
    for E in E_CHOICES:
        for H in H_CHOICES:
            rng = np.random.default_rng(base_rng.integers(2**32))
            params = init_params(d, C, E, H, rng)
            states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
            t_counter = [0]
            for _ in range(TUNE_EPOCHS):
                train_epoch(X_train, y_train, params, states, d, E, H, C, L2, BATCH, rng, t_counter)
            acc = evaluate(X_tune, y_tune, params, d, E, H, C, BATCH)
            if acc > best_acc:
                best_acc = acc
                best_params = [p.copy() for p in params]
                best_E, best_H = E, H
            print(f"E={E} H={H} tune_acc={acc:.4f}")

    print(f"Best: E={best_E} H={best_H} acc={best_acc:.4f}")

    # retrain on full data
    rng = np.random.default_rng(base_rng.integers(2**32))
    params = init_params(d, C, best_E, best_H, rng)
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t_counter = [0]
    for _ in range(FULL_EPOCHS):
        train_epoch(X, y, params, states, d, best_E, best_H, C, L2, BATCH, rng, t_counter)

    emb_flat, W1, b1, W2, b2 = params
    # compute T = effective first-layer weights: (d, 256, H)
    T = np.zeros((d, 256, best_H), dtype=np.float32)
    for j in range(d):
        T[j] = emb_flat[j * 256:(j + 1) * 256] @ W1[j * best_E:(j + 1) * best_E]

    # write weights.bin
    magic = 0x454D424D  # 'EMBM'
    out = struct.pack("<4i", magic, d, best_H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index.
 * Weights format: magic 0x454D424D, int32 d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 16 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *sc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int h = 0; h < H; h++) acc[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += t[h];
        }
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;
        for (int c = 0; c < C; c++) sc[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            for (int c = 0; c < C; c++) sc[c] += ah * W2[(long)h * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```