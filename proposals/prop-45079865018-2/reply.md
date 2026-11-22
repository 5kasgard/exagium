```hypothesis
Family: label-embed-mlp. Position-specific byte embeddings (16×256×E) fed into a one-hidden-layer ReLU MLP (H units) with 65-class output. Embeddings capture per-position character statistics more compactly than one-hot linear, and the hidden layer models interactions between context positions. If dev accuracy does not exceed label-linear (0.3375) by ≥0.01, the added complexity is not justified.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning on the last 20% of train.in
EMB_DIM_CHOICES = [8, 16]
HIDDEN_DIM_CHOICES = [32, 64]
LR_CHOICES = [1e-3, 3e-3]
SEED = 1
BATCH = 64
MAX_EPOCHS_TUNE = 10
MAX_EPOCHS_FULL = 50
L2 = 1e-4
SAMPLES = 600000


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(params, X, y, emb_dim, hidden_dim, num_classes):
    E_flat, W1, b1, W2, b2 = params
    n = X.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        idx = slice(s, min(s + BATCH, n))
        xb = X[idx]
        yb = y[idx]
        flat_idx = xb + np.arange(16) * 256
        emb = E_flat[flat_idx].reshape(len(yb), 16 * emb_dim)
        h = np.maximum(emb @ W1 + b1, 0)
        logits = h @ W2 + b2
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def train_model(X, y, emb_dim, hidden_dim, lr, epochs, rng, num_classes=65):
    n, d = X.shape
    # Parameters
    E_flat = (rng.uniform(-1, 1, (16 * 256, emb_dim)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (16 * emb_dim, hidden_dim)) * math.sqrt(6.0 / (16 * emb_dim))).astype(np.float32)
    b1 = np.zeros(hidden_dim, np.float32)
    W2 = (rng.uniform(-1, 1, (hidden_dim, num_classes)) * math.sqrt(6.0 / hidden_dim)).astype(np.float32)
    b2 = np.zeros(num_classes, np.float32)
    params = [E_flat, W1, b1, W2, b2]
    shapes = [p.shape for p in params]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, yb):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(yb)), yb] -= 1.0
        return (probs / len(yb)).astype(np.float32)

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            flat_idx = xb + np.arange(16) * 256
            emb = E_flat[flat_idx].reshape(len(yb), 16 * emb_dim)
            # forward
            h_pre = emb @ W1 + b1
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2
            g = softmax_grad(logits, yb)
            # backward
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(0)
            gh = g @ W2.T
            gh[h_pre <= 0] = 0.0
            grad_W1 = emb.T @ gh + L2 * W1
            grad_b1 = gh.sum(0)
            # embedding gradients
            g_emb = gh @ W1.T  # (batch, 16*emb_dim)
            g_emb = g_emb.reshape(len(yb), 16, emb_dim)
            grad_E_flat = np.zeros_like(E_flat)
            for i in range(len(yb)):
                for pos in range(16):
                    token = xb[i, pos]
                    grad_E_flat[pos * 256 + token] += g_emb[i, pos]
            grad_E_flat += L2 * E_flat
            grads = [grad_E_flat, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]
    # split: last 20% for tuning
    n_val = n // 5
    X_train, y_train = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_params = None
    best_hp = None

    for emb_dim in EMB_DIM_CHOICES:
        for hidden_dim in HIDDEN_DIM_CHOICES:
            for lr in LR_CHOICES:
                hp_rng = np.random.default_rng(SEED)
                params = train_model(X_train, y_train, emb_dim, hidden_dim, lr, MAX_EPOCHS_TUNE, hp_rng, C)
                acc = accuracy(params, X_val, y_val, emb_dim, hidden_dim, C)
                if acc > best_acc:
                    best_acc = acc
                    best_params = params
                    best_hp = (emb_dim, hidden_dim, lr)
    print(f"Best hyperparams: emb_dim={best_hp[0]}, hidden_dim={best_hp[1]}, lr={best_hp[2]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data
    emb_dim, hidden_dim, lr = best_hp
    full_rng = np.random.default_rng(SEED)
    final_params = train_model(X, y, emb_dim, hidden_dim, lr, MAX_EPOCHS_FULL, full_rng, C)

    E_flat, W1, b1, W2, b2 = final_params
    # Write weights.bin
    # Header: magic, d, emb_dim, hidden_dim, C
    header = struct.pack("<5i", 0x454D4244, X.shape[1], emb_dim, hidden_dim, C)
    out = header
    out += E_flat.astype("<f4").tobytes()
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], emb_dim = hd[2], hidden_dim = hd[3], C = hd[4];
    if (d != 16 || C != 65 || emb_dim <= 0 || hidden_dim <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20 +
        4L * (16 * 256 * emb_dim) +
        4L * (16 * emb_dim * hidden_dim) +
        4L * hidden_dim +
        4L * (hidden_dim * C) +
        4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E_flat = (const float *)(hd + 5);
    const float *W1 = E_flat + 16 * 256 * emb_dim;
    const float *b1 = W1 + 16 * emb_dim * hidden_dim;
    const float *W2 = b1 + hidden_dim;
    const float *b2 = W2 + hidden_dim * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_concat = malloc(sizeof(float) * 16 * emb_dim);
    float *hidden = malloc(sizeof(float) * hidden_dim);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int pos_i = 0; pos_i < 16; pos_i++) {
            const float *e = E_flat + (pos_i * 256 + xi[pos_i]) * emb_dim;
            memcpy(emb_concat + pos_i * emb_dim, e, sizeof(float) * emb_dim);
        }

        // Hidden layer: ReLU(emb_concat @ W1 + b1)
        for (int h = 0; h < hidden_dim; h++) {
            float sum = b1[h];
            for (int j = 0; j < 16 * emb_dim; j++) {
                sum += emb_concat[j] * W1[j * hidden_dim + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < hidden_dim; h++) {
                sum += hidden[h] * W2[h * C + c];
            }
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```