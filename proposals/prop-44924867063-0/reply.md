```hypothesis
Family: label-embmlp (Feature Embedding MLP). Learns a dense embedding per feature value (256 per column), concatenates them, and passes through one ReLU hidden layer. Captures non-linear interactions with fewer parameters than one-hot MLP. Should beat label-linear (0.8524) and approach label-mlp16 (0.8591) at similar or lower CPU cost. Failure if dev accuracy ≤ label-linear.
```

```python
"""label-embmlp: Feature embedding + one hidden ReLU layer.
train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid searched on tuning split (last 20% of train.in)
EMB_GRID = [4, 8]
HIDDEN_GRID = [16, 32, 64]
LR_GRID = [1e-3, 3e-3]
L2_GRID = [1e-4, 1e-5]
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
SEED = 1


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return np.mean(np.argmax(logits, axis=1) == y)


def forward(Emb, W1, b1, W2, b2, X):
    """X: (n, d) uint8. Returns logits (n, C)."""
    n, d = X.shape
    E = Emb.shape[2]
    # Embedding lookup: (n, d, E)
    Z = Emb[np.arange(d), X]          # shape (n, d, E)
    Z = Z.reshape(n, d * E)           # (n, d*E)
    H = np.maximum(Z @ W1 + b1, 0)    # (n, H)
    return H @ W2 + b2                # (n, C)


def train_split(X, y, Emb, W1, b1, W2, b2, lr, l2, rng, epochs, batch):
    n = X.shape[0]
    params = [Emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            # Forward
            n_b, d = xb.shape
            E = Emb.shape[2]
            Z = Emb[np.arange(d), xb].reshape(n_b, d * E)
            A = Z @ W1 + b1
            H = np.maximum(A, 0)
            logits = H @ W2 + b2
            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / exp.sum(axis=1, keepdims=True)
            probs[np.arange(n_b), yb] -= 1.0
            grads_out = probs / n_b                       # (n_b, C)
            # Backprop
            gW2 = H.T @ grads_out + l2 * W2
            gb2 = grads_out.sum(0)
            gH = grads_out @ W2.T
            gH[A <= 0] = 0
            gW1 = Z.T @ gH + l2 * W1
            gb1 = gH.sum(0)
            # Embedding gradients: gZ = gH @ W1.T  (n_b, d*E)
            gZ = gH @ W1.T
            gZ = gZ.reshape(n_b, d, E)
            gEmb = np.zeros_like(Emb)
            np.add.at(gEmb, (np.arange(d), xb), gZ)
            gEmb += l2 * Emb
            grads = [gEmb, gW1, gb1, gW2, gb2]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    n = X.shape[0]

    # Tuning split: last 20%
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc, best_cfg, best_params = -1, None, None

    for E in EMB_GRID:
        for H in HIDDEN_GRID:
            for lr in LR_GRID:
                for l2 in L2_GRID:
                    # Initialize
                    Emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
                    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
                    b1 = np.zeros(H, np.float32)
                    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
                    b2 = np.zeros(C, np.float32)

                    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))
                    train_split(X_tr, y_tr, Emb, W1, b1, W2, b2, lr, l2, rng, epochs, BATCH)

                    logits = forward(Emb, W1, b1, W2, b2, X_val)
                    acc = accuracy(logits, y_val)
                    if acc > best_acc:
                        best_acc = acc
                        best_cfg = (E, H, lr, l2)
                        best_params = (Emb.copy(), W1.copy(), b1.copy(), W2.copy(), b2.copy())

    # Retrain on full data with best config
    E, H, lr, l2 = best_cfg
    Emb, W1, b1, W2, b2 = best_params
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    train_split(X, y, Emb, W1, b1, W2, b2, lr, l2, rng, epochs, BATCH)

    # Export weights
    # Magic: 0x454D4244 ("EMBD" little-endian), d, E, H, C
    header = struct.pack("<5i", 0x454D4244, d, E, H, C)
    # Embeddings: d * 256 * E floats (feature-major, then value, then embedding dim)
    emb_bytes = Emb.astype("<f4").tobytes()
    # MLP weights
    mlp_bytes = (W1.astype("<f4").tobytes() +
                 b1.astype("<f4").tobytes() +
                 W2.astype("<f4").tobytes() +
                 b2.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + emb_bytes + mlp_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input: d bytes per record (d from weights). Output: one byte class index. */
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
    if (C < 1 || C > 256 || E < 1 || H < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected = 20 + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *Emb = (const float *)(hd + 5);
    const float *W1 = Emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = Emb + ((long)j * 256 + xi[j]) * E;
            memcpy(emb_concat + j * E, e, sizeof(float) * E);
        }

        // Hidden layer: ReLU(emb_concat @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) sum += emb_concat[k] * W1[(long)k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```