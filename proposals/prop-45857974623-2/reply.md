```hypothesis
We propose a position-wise embedding MLP (family "label-embmlp"). Each of the 16 context bytes gets a learned embedding (dim D), summed, then passed through a ReLU hidden layer (size H) to predict 65 classes. This captures interactions between positions with fewer parameters than one-hot linear (4096*65) and should generalize better on this character-level task. If dev accuracy does not exceed 0.34, the hypothesis fails.
```

```python
"""label-embmlp: position-wise embedding + one hidden ReLU layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then trained on all train.
Exported weights: magic 0x4C424D4C, d,D,H,C; float32 embeddings[d][256][D], W1[D][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Fixed training configuration
SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
LR = 3e-3
L2 = 1e-4

# Hyperparameter grid for tuning
D_CHOICES = [16, 32, 64]
H_CHOICES = [32, 64, 128]
LR_CHOICES = [1e-3, 3e-3]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def split_data(X, y, tune_frac=0.2):
    n = X.shape[0]
    tune_n = int(n * tune_frac)
    return X[:-tune_n], y[:-tune_n], X[-tune_n:], y[-tune_n:]


def init_params(d, D, H, C, rng):
    # embeddings: [d, 256, D]
    emb = (rng.normal(0, 0.01, (d, 256, D))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / D), (D, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return emb, W1, b1, W2, b2


def forward(emb, W1, b1, W2, b2, X_idx):
    # X_idx: [batch, d] uint8
    batch = X_idx.shape[0]
    d = emb.shape[0]
    D = emb.shape[2]
    # Sum embeddings
    h = np.zeros((batch, D), np.float32)
    for j in range(d):
        h += emb[j, X_idx[:, j]]
    # Hidden layer
    h = np.maximum(h @ W1 + b1, 0, dtype=np.float32)
    # Output
    logits = h @ W2 + b2
    return logits, h


def loss_and_grad(emb, W1, b1, W2, b2, X_idx, y):
    batch = X_idx.shape[0]
    d, _, D = emb.shape
    H = W1.shape[1]
    C = W2.shape[1]

    # Forward
    h_sum = np.zeros((batch, D), np.float32)
    for j in range(d):
        h_sum += emb[j, X_idx[:, j]]
    pre_act = h_sum @ W1 + b1
    h = np.maximum(pre_act, 0, dtype=np.float32)
    logits = h @ W2 + b2

    # Softmax + cross-entropy gradient
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits, dtype=np.float32)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(batch), y] -= 1.0
    probs /= batch  # dL/dlogits

    # Gradients for output layer
    gW2 = h.T @ probs
    gb2 = probs.sum(axis=0)

    # Backprop to hidden
    gh = probs @ W2.T
    gh[pre_act <= 0] = 0.0  # ReLU gradient

    # Gradients for hidden layer
    gW1 = h_sum.T @ gh
    gb1 = gh.sum(axis=0)

    # Gradient for summed embeddings
    g_sum = gh @ W1.T  # [batch, D]

    # Gradients for each position's embeddings
    g_emb = np.zeros_like(emb)
    for j in range(d):
        np.add.at(g_emb[j], X_idx[:, j], g_sum)

    # L2 regularization gradients
    gW2 += L2 * W2
    gW1 += L2 * W1
    g_emb += L2 * emb

    # Loss for monitoring
    loss = -np.log(probs[np.arange(batch), y] * batch + 1e-12).mean()
    return loss, (g_emb, gW1, gb1, gW2, gb2)


def train_loop(emb, W1, b1, W2, b2, X_idx, y, rng, max_epochs, batch_size, total_samples):
    n = X_idx.shape[0]
    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    epochs = min(max_epochs, math.ceil(total_samples / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            xi, yi = X_idx[idx], y[idx]
            loss, grads = loss_and_grad(emb, W1, b1, W2, b2, xi, yi)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def evaluate(emb, W1, b1, W2, b2, X_idx, y):
    logits, _ = forward(emb, W1, b1, W2, b2, X_idx)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def write_weights(path, d, D, H, C, emb, W1, b1, W2, b2):
    magic = 0x4C424D4C  # 'LBML'
    expected = 16 + 4 * (d * 256 * D + D * H + H + H * C + C)
    with open(path, "wb") as f:
        f.write(struct.pack("<4i", magic, d, D, H, C))
        f.write(emb.astype("<f4").tobytes())
        f.write(W1.astype("<f4").tobytes())
        f.write(b1.astype("<f4").tobytes())
        f.write(W2.astype("<f4").tobytes())
        f.write(b2.astype("<f4").tobytes())


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 16 and C == 65

    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)

    # Split for tuning
    X_train, y_train, X_tune, y_tune = split_data(X_bytes, y)

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    best_params = None

    for D in D_CHOICES:
        for H in H_CHOICES:
            for lr in LR_CHOICES:
                global LR
                LR = lr
                emb, W1, b1, W2, b2 = init_params(d, D, H, C, rng)
                params = train_loop(emb, W1, b1, W2, b2, X_train, y_train, rng, MAX_EPOCHS, BATCH, SAMPLES)
                emb, W1, b1, W2, b2 = params
                acc = evaluate(emb, W1, b1, W2, b2, X_tune, y_tune)
                print(f"D={D} H={H} lr={lr:.0e} tune_acc={acc:.4f}", file=sys.stderr)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (D, H, lr)
                    best_params = (emb.copy(), W1.copy(), b1.copy(), W2.copy(), b2.copy())

    print(f"Best config: D={best_cfg[0]} H={best_cfg[1]} lr={best_cfg[2]:.0e} acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    D, H, lr = best_cfg
    LR = lr
    emb, W1, b1, W2, b2 = init_params(d, D, H, C, rng)
    # Use the best_params as starting point? We'll retrain from scratch for determinism.
    # But we can also continue training. Simpler: retrain from scratch on full data.
    # However, to save time and ensure determinism, we'll just use the best_params and do a few more epochs?
    # The contract says: choose settings on tuning split, then train on all of train.
    # So we should train from scratch on full data with the chosen hyperparameters.
    emb, W1, b1, W2, b2 = init_params(d, D, H, C, rng)
    params = train_loop(emb, W1, b1, W2, b2, X_bytes, y, rng, MAX_EPOCHS, BATCH, SAMPLES)
    emb, W1, b1, W2, b2 = params

    write_weights(sys.argv[4], d, D, H, C, emb, W1, b1, W2, b2)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format: magic 0x4C424D4C, int32 d,D,H,C; float32 emb[d][256][D], W1[D][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 16 || hd[0] != 0x4C424D4C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * D;
    const float *b1 = W1 + (long)D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h_sum = malloc(sizeof(float) * D);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Sum embeddings
        for (int j = 0; j < D; j++) h_sum[j] = 0.0f;
        for (int p = 0; p < d; p++) {
            const float *e = emb + ((long)p * 256 + xi[p]) * D;
            for (int j = 0; j < D; j++) h_sum[j] += e[j];
        }

        // Hidden layer: ReLU(h_sum @ W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int j = 0; j < D; j++) sum += h_sum[j] * W1[(long)j * H + u];
            hidden[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += hidden[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(h_sum); free(hidden); free(logits);
    return 0;
}
```