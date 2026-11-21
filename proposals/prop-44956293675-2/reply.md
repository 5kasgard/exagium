```hypothesis
Family: label-mlp (one-hidden-layer MLP with per-position byte embeddings, hidden size tuned on validation).
Why: The linear model (33.75%) ignores interactions between context bytes; a modest hidden layer (32–128 units) can capture non-linear n-gram patterns while keeping inference fast via the same per-byte lookup tables. The existing label-mlp16 (H=16) underfits; a tuned larger hidden layer should beat linear.
Failure signal: If validation accuracy does not exceed linear's 33.75% after tuning H, LR, and epochs, the task may be linearly separable or the data too small for hidden units to help.
```

```python
"""label-mlp: one-hidden-layer MLP with per-position byte embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters (hidden size, learning rate, epochs) chosen on last 20% of train.in,
then final model trained on all train.in.
Weights format: magic 0x4D4C5031, int32 d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning
H_CHOICES = [32, 64, 128]
LR_CHOICES = [1e-3, 3e-3, 1e-2]
L2 = 1e-4
BATCH = 64
MAX_EPOCHS_TUNE = 20
MAX_EPOCHS_FULL = 100
SAMPLES_PER_EPOCH = 600000  # same as label-linear


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_inputs(X, card):
    """Return list of one-hot encoding matrices per position (256, num_vals) and concatenated Z."""
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j]
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(v)
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    return enc, Z


def train_mlp(Z, y, H, C, lr, epochs, rng):
    """Train MLP with Adam, return (T, b1, W2, b2) where T is list of (256, H) per position."""
    n, k = Z.shape
    d = Z.shape[1] // 256  # because each position contributes 256 dims (one-hot)
    # Initialize weights
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # forward
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            # backward
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(0)
            gh = g @ W2.T
            gh[a <= 0] = 0.0
            grad_W1 = z.T @ gh + L2 * W1
            grad_b1 = gh.sum(0)
            grads = [grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    # Fold first layer into per-position tables
    off = 0
    T = []
    for j in range(d):
        T.append(W1[off:off + 256].copy())
        off += 256
    return T, b1, W2, b2


def evaluate(T, b1, W2, b2, Z, y):
    n = Z.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        z = Z[s:s + BATCH]
        yy = y[s:s + BATCH]
        a = z @ np.vstack(T) + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
        pred = logits.argmax(axis=1)
        correct += (pred == yy).sum()
    return correct / n


def write_weights(d, H, C, T, b1, W2, b2, path):
    magic = 0x4D4C5031  # "MLP1"
    header = struct.pack("<4i", magic, d, H, C)
    # T is list of (256, H) -> stack to (d, 256, H)
    T_arr = np.stack(T, axis=0).astype("<f4")
    body = T_arr.tobytes() + b1.astype("<f4").tobytes() + W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(path, "wb").write(header + body)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    # Encode once
    enc, Z = encode_inputs(X, card)

    # Split: last 20% for tuning
    split = int(0.8 * n)
    Z_train, Z_tune = Z[:split], Z[split:]
    y_train, y_tune = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_params = None
    best_model = None

    for H in H_CHOICES:
        for lr in LR_CHOICES:
            # Train on train split
            model = train_mlp(Z_train, y_train, H, C, lr, MAX_EPOCHS_TUNE, rng)
            T, b1, W2, b2 = model
            acc = evaluate(T, b1, W2, b2, Z_tune, y_tune)
            if acc > best_acc:
                best_acc = acc
                best_params = (H, lr)
                best_model = model
    print(f"Best tuning: H={best_params[0]}, lr={best_params[1]}, acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best hyperparameters, more epochs
    H_best, lr_best = best_params
    rng = np.random.default_rng(SEED)  # reset seed for reproducibility
    final_model = train_mlp(Z, y, H_best, C, lr_best, MAX_EPOCHS_FULL, rng)
    T, b1, W2, b2 = final_model
    write_weights(d, H_best, C, T, b1, W2, b2, sys.argv[4])


if __name__ == "__main__":
    main()
```

```c
/* label-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index (0..C-1).
 * Weights format: magic 0x4D4C5031, int32 d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 16 || hd[0] != 0x4D4C5031) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || H < 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

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
        for (int c = 0; c < C; c++) sc[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float ah = acc[h] > 0.0f ? acc[h] : 0.0f;
            if (ah != 0.0f) {
                const float *w = W2 + (long)h * C;
                for (int c = 0; c < C; c++) sc[c] += ah * w[c];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```