```hypothesis
Family: label-mlp (one hidden ReLU layer, hidden size tuned on last 20% of train).
Why: Optdigits has non-linear pixel interactions; a small MLP captures them better than linear.
Failure: If dev accuracy ≤ label-linear (0.9613) or CPU cost exceeds label-linear without accuracy gain.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (same as label-linear family)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
HIDDEN_CHOICES = [8, 16, 32, 64]  # tuned on validation split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_features(X, types, fit=True, stats=None):
    """Ordinal z-scoring per feature. Returns encoded matrix and stats (mean, std) for each feature."""
    n, d = X.shape
    if fit:
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds[stds == 0] = 1.0
        stats = (means, stds)
    else:
        means, stds = stats
    Z = (X - means) / stds
    return Z.astype(np.float32), stats


def train_mlp(X, y, H, C, rng, epochs):
    """Train MLP with one hidden layer of size H. Returns weights (W1, b1, W2, b2)."""
    n, d = X.shape
    # Xavier init
    W1 = (rng.uniform(-1, 1, (d, H)) * math.sqrt(6.0 / d)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            # forward
            a1 = xb @ W1 + b1
            h = np.maximum(a1, 0)
            logits = h @ W2 + b2
            g = softmax_grad(logits, yb)
            # backward
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(0)
            gh = g @ W2.T * (a1 > 0)
            grad_W1 = xb.T @ gh + L2 * W1
            grad_b1 = gh.sum(0)
            grads = [grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def evaluate(X, y, params):
    W1, b1, W2, b2 = params
    a1 = X @ W1 + b1
    h = np.maximum(a1, 0)
    logits = h @ W2 + b2
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n_all = X_all.shape[0]

    # Split: last 20% for tuning
    split = int(0.8 * n_all)
    X_train, y_train = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]

    # Encode features using training split stats
    X_train_enc, stats = encode_features(X_train.astype(np.float64), card["input"]["types"], fit=True)
    X_val_enc, _ = encode_features(X_val.astype(np.float64), card["input"]["types"], fit=False, stats=stats)

    rng = np.random.default_rng(SEED)
    best_H = HIDDEN_CHOICES[0]
    best_acc = -1.0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))

    for H in HIDDEN_CHOICES:
        # Use a deterministic sub-seed for each H to keep runs independent but reproducible
        sub_rng = np.random.default_rng(SEED + H)
        params = train_mlp(X_train_enc, y_train, H, C, sub_rng, epochs)
        acc = evaluate(X_val_enc, y_val, params)
        if acc > best_acc:
            best_acc = acc
            best_H = H

    # Retrain on all data with best_H
    X_all_enc, stats = encode_features(X_all.astype(np.float64), card["input"]["types"], fit=True)
    final_rng = np.random.default_rng(SEED + best_H)
    final_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_all))
    W1, b1, W2, b2 = train_mlp(X_all_enc, y_all, best_H, C, final_rng, final_epochs)

    # Save weights: magic, d, H, C, then W1 (d*H), b1 (H), W2 (H*C), b2 (C) all float32 little-endian
    d = X_all.shape[1]
    magic = 0x4D4C5031  # "MLP1"
    out = struct.pack("<4i", magic, d, best_H, C)
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x4D4C5031) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 16L + 4L * ((long)d * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W1 = (const float *)(hd + 4);
    const float *b1 = W1 + (long)d * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // hidden = ReLU(x @ W1 + b1)
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *w = W1 + (long)j * H;
            float xv = xi[j];
            for (int h = 0; h < H; h++) hidden[h] += xv * w[h];
        }
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        // scores = hidden @ W2 + b2
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```