```hypothesis
Family: label-embmlp (embedding + one-hidden-layer MLP). Each of the 16 context bytes gets a learned embedding (size E), concatenated into a 16*E vector, passed through a ReLU hidden layer (size H), then a linear output layer to 65 classes. This captures non-linear positional interactions with far fewer parameters than label-linear's 4096*65 one-hot weights, so it should be both more accurate and faster. Hyperparameters (E, H) are chosen on the last 20% of train.in (file order), then retrained on all data. Failure: dev accuracy ≤ 0.3375 (label-linear) or inference cost ≥ label-linear.
```

```python
"""label-embmlp: embedding + one-hidden-layer MLP for next-byte prediction.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Adam hyperparameters (same as label-linear)
LR, BETA1, BETA2, EPS, L2, BATCH = 3e-3, 0.9, 0.999, 1e-8, 1e-4, 64
SAMPLES, MAX_EPOCHS = 600000, 100
SEED = 1

# Grid searched on the tuning split (last 20% of train.in in file order)
E_GRID = [16, 32, 64]
H_GRID = [32, 64, 128]
GRID_EPOCHS = 5  # quick evaluation epochs per grid point


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(d, E, H, C, rng):
    """He init for ReLU layers, small normal for embeddings."""
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (H, d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (C, H))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return emb, W1, b1, W2, b2


def forward(emb, W1, b1, W2, b2, X):
    """X: (N, d) uint8 -> logits (N, C) float32"""
    N, d = X.shape
    E = emb.shape[2]
    # Embedding lookup per position
    x = np.empty((N, d, E), dtype=np.float32)
    for j in range(d):
        x[:, j, :] = emb[j, X[:, j], :]
    x = x.reshape(N, d * E)
    h_pre = x @ W1.T + b1
    h = np.maximum(h_pre, 0, dtype=np.float32)
    logits = h @ W2.T + b2
    return logits, h_pre, h, x


def loss_and_grad(emb, W1, b1, W2, b2, X, y):
    N = X.shape[0]
    logits, h_pre, h, x = forward(emb, W1, b1, W2, b2, X)
    # Softmax cross-entropy
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(N), y] -= 1.0
    probs /= N
    loss = -np.log(np.maximum(probs[np.arange(N), y] + 1.0/N, 1e-12)).mean()  # not used, but ok
    # Gradients
    dW2 = probs.T @ h
    db2 = probs.sum(axis=0)
    dh = probs @ W2
    dh_pre = dh * (h_pre > 0)
    dW1 = dh_pre.T @ x
    db1 = dh_pre.sum(axis=0)
    dx = dh_pre @ W1
    dx = dx.reshape(N, X.shape[1], -1)
    demb = np.zeros_like(emb)
    for j in range(X.shape[1]):
        np.add.at(demb[j], X[:, j], dx[:, j, :])
    # L2 regularization
    demb += L2 * emb
    dW1 += L2 * W1
    dW2 += L2 * W2
    return demb, dW1, db1, dW2, db2


def accuracy(emb, W1, b1, W2, b2, X, y):
    logits, _, _, _ = forward(emb, W1, b1, W2, b2, X)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def train_split(train_X, train_y, val_X, val_y, E, H, epochs, rng):
    d, C = train_X.shape[1], 65
    emb, W1, b1, W2, b2 = init_params(d, E, H, C, rng)
    params = [emb, W1, b1, W2, b2]
    shapes = [p.shape for p in params]
    # Adam state
    m = [np.zeros(s, np.float32) for s in shapes]
    v = [np.zeros(s, np.float32) for s in shapes]
    t = 0
    n = len(train_X)
    steps_per_epoch = math.ceil(n / BATCH)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = train_X[idx], train_y[idx]
            grads = loss_and_grad(emb, W1, b1, W2, b2, Xb, yb)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = BETA1 * m[i] + (1 - BETA1) * gr
                v[i] = BETA2 * v[i] + (1 - BETA2) * gr * gr
                m_hat = m[i] / (1 - BETA1 ** t)
                v_hat = v[i] / (1 - BETA2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + EPS)
    val_acc = accuracy(emb, W1, b1, W2, b2, val_X, val_y)
    return val_acc, params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = card["input"]["bytes"], card["classes"]
    assert d == 16 and C == 65

    # Tuning split: last 20% in file order
    n = len(X)
    val_size = n // 5
    train_X, train_y = X[:-val_size], y[:-val_size]
    val_X, val_y = X[-val_size:], y[-val_size:]

    rng = np.random.default_rng(SEED)
    best_acc, best_E, best_H, best_params = -1, None, None, None

    # Grid search
    for E in E_GRID:
        for H in H_GRID:
            rng_grid = np.random.default_rng(SEED + E * 1000 + H)  # deterministic per config
            acc, params = train_split(train_X, train_y, val_X, val_y, E, H, GRID_EPOCHS, rng_grid)
            if acc > best_acc:
                best_acc, best_E, best_H, best_params = acc, E, H, params

    # Retrain on full data with best config
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng_full = np.random.default_rng(SEED + 9999)
    _, final_params = train_split(X, y, val_X, val_y, best_E, best_H, max_epochs, rng_full)
    emb, W1, b1, W2, b2 = final_params

    # Pack weights: magic, d, E, H, C; then emb, W1, b1, W2, b2 (all float32 LE)
    magic = 0x454D424D  # 'EMBM'
    out = struct.pack("<5i", magic, d, best_E, best_H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format (little-endian):
 *   int32 magic=0x454D424D, d=16, E, H, C=65
 *   float32 emb[d][256][E]
 *   float32 W1[H][d*E]
 *   float32 b1[H]
 *   float32 W2[C][H]
 *   float32 b2[C] */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 20L
        + 4L * d * 256 * E
        + 4L * H * d * E
        + 4L * H
        + 4L * C * H
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + d * 256 * E;
    const float *b1 = W1 + H * d * E;
    const float *W2 = b1 + H;
    const float *b2 = W2 + C * H;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *in_vec = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* Embedding lookup: concatenate d vectors of size E */
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(in_vec + j * E, src, E * sizeof(float));
        }

        /* Hidden layer: ReLU(W1 * in_vec + b1) */
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            const float *w = W1 + (long)h * d * E;
            for (int k = 0; k < d * E; k++) sum += in_vec[k] * w[k];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        /* Output layer: W2 * hidden + b2 */
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            const float *w = W2 + (long)c * H;
            for (int h = 0; h < H; h++) sum += hidden[h] * w[h];
            logits[c] = sum;
        }

        /* Argmax (first max wins) */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```