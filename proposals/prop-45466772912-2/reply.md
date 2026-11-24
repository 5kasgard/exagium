```hypothesis
Family: label-embed-mlp. A small MLP with learned per-position byte embeddings (E=8) and one hidden ReLU layer (H=64). Embeddings compress the 16×256 one-hot space to 128 dense features, allowing the hidden layer to model cross-position interactions that a linear model cannot capture, while staying far cheaper than gradient-boosted trees. Hyperparameters (E, H, learning rate) are chosen on the last 20% of train.in, then the model is retrained on all data. Failure: dev accuracy ≤ 0.3375 (label-linear) or inference cost ≥ 2× label-linear.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split (last 20% of train.in)
GRID = [
    {"E": 8, "H": 64, "lr": 3e-3, "epochs_full": 50},
    {"E": 4, "H": 32, "lr": 3e-3, "epochs_full": 50},
    {"E": 16, "H": 128, "lr": 1e-3, "epochs_full": 50},
]

SEED = 1
BATCH = 64
L2 = 1e-4
VALID_FRAC = 0.2


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X_train, y_train, X_val, y_val, E, H, lr, epochs, rng):
    n, d = X_train.shape
    C = 65
    # Embeddings: (d, 256, E)
    emb = rng.normal(0, 0.01, (d, 256, E)).astype(np.float32)
    # Hidden layer: (d*E, H)
    W1 = rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer: (H, C)
    W2 = rng.normal(0, math.sqrt(2.0 / H), (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X_train[idx]
            y = y_train[idx]
            B = len(y)

            # Embedding lookup and concatenation
            embedded = np.concatenate([emb[j, x[:, j]] for j in range(d)], axis=1)  # (B, d*E)

            # Forward
            h = embedded @ W1 + b1
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2

            # Softmax cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), y] -= 1.0
            probs /= B

            # Backprop
            grad_W2 = h_relu.T @ probs + L2 * W2
            grad_b2 = probs.sum(axis=0)
            grad_h_relu = probs @ W2.T
            grad_h = grad_h_relu * (h > 0)
            grad_W1 = embedded.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)

            grad_emb = np.zeros_like(emb)
            for j in range(d):
                g = grad_h[:, j * E:(j + 1) * E]
                indices = x[:, j]
                np.add.at(grad_emb[j], indices, g)

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Validation accuracy
    def predict(X):
        embedded = np.concatenate([emb[j, X[:, j]] for j in range(d)], axis=1)
        h = embedded @ W1 + b1
        h_relu = np.maximum(h, 0)
        logits = h_relu @ W2 + b2
        return logits.argmax(axis=1)

    val_pred = predict(X_val)
    acc = (val_pred == y_val).mean()
    return acc, params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    assert d == 16 and C == 65

    n = X.shape[0]
    val_size = int(n * VALID_FRAC)
    train_size = n - val_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_config = None

    for config in GRID:
        config_seed = SEED + hash((config["E"], config["H"], config["lr"])) % 2**32
        config_rng = np.random.default_rng(config_seed)
        acc, params = train_model(X_train, y_train, X_val, y_val,
                                  config["E"], config["H"], config["lr"], 10, config_rng)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_config = config

    print(f"Best config: {best_config}, val acc: {best_acc:.4f}", file=sys.stderr)

    # Retrain on full data
    full_rng = np.random.default_rng(SEED + 12345)
    _, final_params = train_model(X, y, X_val, y_val,
                                  best_config["E"], best_config["H"], best_config["lr"],
                                  best_config["epochs_full"], full_rng)
    emb, W1, b1, W2, b2 = final_params

    # Write weights.bin
    magic = 0x454D4244  # 'EMBD'
    d, E, H, C = 16, best_config["E"], best_config["H"], 65
    emb_flat = emb.astype("<f4").ravel()
    W1_flat = W1.astype("<f4").ravel()
    b1_flat = b1.astype("<f4").ravel()
    W2_flat = W2.astype("<f4").ravel()
    b2_flat = b2.astype("<f4").ravel()

    out = struct.pack("<5i", magic, d, E, H, C)
    out += emb_flat.tobytes()
    out += W1_flat.tobytes()
    out += b1_flat.tobytes()
    out += W2_flat.tobytes()
    out += b2_flat.tobytes()

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
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }

    long total_floats = (long)d * 256 * E + (long)d * E * H + H + (long)H * C + C;
    if (nw != 20 + 4 * total_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *output = malloc(sizeof(float) * C);
    float *embedded = malloc(sizeof(float) * d * E);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(embedded + j * E, e, E * sizeof(float));
        }

        // Hidden layer: hidden = embedded * W1 + b1
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        for (int in = 0; in < d * E; in++) {
            float val = embedded[in];
            if (val != 0.0f) {
                const float *w = W1 + (long)in * H;
                for (int h = 0; h < H; h++) hidden[h] += val * w[h];
            }
        }
        // ReLU
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        // Output layer: output = hidden * W2 + b2
        for (int c = 0; c < C; c++) output[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float val = hidden[h];
            if (val != 0.0f) {
                const float *w = W2 + (long)h * C;
                for (int c = 0; c < C; c++) output[c] += val * w[c];
            }
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (output[c] > output[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```