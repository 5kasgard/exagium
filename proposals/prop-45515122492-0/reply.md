```hypothesis
Family: label-embedmlp. Learns a dense embedding (size E) for every feature value, sums them, then passes through a ReLU hidden layer (size H) to logits. This captures non-linear feature interactions and similarities between categories that linear models miss, while staying much faster than GBDT. If dev accuracy does not exceed label-linear (0.8524), it suggests the extra capacity overfits or ordinal features need explicit monotonic encoding.
```

```python
"""label-embedmlp: embedding + one-hidden-layer MLP for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Hyperparameters (E, H) chosen on last 20% of train.in (file order), then retrained on all train.
Weights format (little-endian):
  magic=0x454D4244, d, E, H, C (int32 each)
  embeddings[d][256][E] (float32)
  W1[E][H], b1[H], W2[H][C], b2[C] (float32)"""
import json, math, struct, sys

import numpy as np

# Fixed training hyperparameters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Hyperparameter grid for tuning split
E_CHOICES = [4, 8, 16]
H_CHOICES = [8, 16, 32]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(d, E, H, C, rng):
    # Embeddings: d x 256 x E, Xavier-like
    emb = (rng.normal(0, 1.0 / math.sqrt(E), size=(d, 256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / E), size=(E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), size=(H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return emb, W1, b1, W2, b2

def forward(X, emb, W1, b1, W2, b2):
    # X: (n, d) uint8
    n, d = X.shape
    E = emb.shape[2]
    # Sum embeddings
    h = np.zeros((n, E), np.float32)
    for j in range(d):
        h += emb[j, X[:, j]]
    # Hidden layer
    h = h @ W1 + b1
    np.maximum(h, 0, out=h)  # ReLU
    # Output
    logits = h @ W2 + b2
    return logits

def loss_grad(logits, y):
    # Softmax cross-entropy gradient w.r.t logits
    n = logits.shape[0]
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return probs / n  # (n, C)

def evaluate_acc(X, y, emb, W1, b1, W2, b2):
    logits = forward(X, emb, W1, b1, W2, b2)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def train_split(X, y, emb, W1, b1, W2, b2, rng, steps_per_epoch, epochs):
    n = X.shape[0]
    params = [emb, W1, b1, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            if len(idx) == 0:
                continue
            xb, yb = X[idx], y[idx]
            # Forward
            h = np.zeros((len(idx), emb.shape[2]), np.float32)
            for j in range(X.shape[1]):
                h += emb[j, xb[:, j]]
            h = h @ W1 + b1
            np.maximum(h, 0, out=h)
            logits = h @ W2 + b2
            # Loss gradient
            g = loss_grad(logits, yb)  # (b, C)
            # Backprop
            gW2 = h.T @ g + L2 * W2
            gb2 = g.sum(0)
            gh = g @ W2.T
            gh[h == 0] = 0  # ReLU grad
            gW1 = (h - gh + gh).T @ gh  # Wait, we need input to hidden layer
            # Actually, input to hidden layer is the summed embeddings (call it z)
            z = np.zeros((len(idx), emb.shape[2]), np.float32)
            for j in range(X.shape[1]):
                z += emb[j, xb[:, j]]
            gW1 = z.T @ gh + L2 * W1
            gb1 = gh.sum(0)
            # Gradient w.r.t embeddings
            g_emb = np.zeros_like(emb)
            for j in range(X.shape[1]):
                # For each feature, scatter-add gh to g_emb[j, xb[:, j]]
                np.add.at(g_emb[j], xb[:, j], gh)
            g_emb += L2 * emb
            grads = [g_emb, gW1, gb1, gW2, gb2]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    # Tuning split: last 20% in file order
    n_tune = int(0.2 * n)
    n_train = n - n_tune
    X_train, y_train = X[:n_train], y[:n_train]
    X_tune, y_tune = X[n_train:], y[n_train:]
    
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_E = best_H = None
    
    for E in E_CHOICES:
        for H in H_CHOICES:
            # Initialize
            emb, W1, b1, W2, b2 = init_params(d, E, H, C, rng)
            # Train on training split
            steps_per_epoch = math.ceil(n_train / BATCH)
            epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))
            train_split(X_train, y_train, emb, W1, b1, W2, b2, rng, steps_per_epoch, epochs)
            # Evaluate on tuning split
            acc = evaluate_acc(X_tune, y_tune, emb, W1, b1, W2, b2)
            if acc > best_acc:
                best_acc = acc
                best_E, best_H = E, H
                best_params = (emb.copy(), W1.copy(), b1.copy(), W2.copy(), b2.copy())
    
    # Retrain on full dataset with best hyperparameters
    emb, W1, b1, W2, b2 = init_params(d, best_E, best_H, C, rng)
    steps_per_epoch = math.ceil(n / BATCH)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    train_split(X, y, emb, W1, b1, W2, b2, rng, steps_per_epoch, epochs)
    
    # Write weights.bin
    # magic, d, E, H, C
    out = struct.pack("<5i", 0x454D4244, d, best_E, best_H, C)
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
/* label-embedmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from weights). Output: one byte class index (argmax). */
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
    if (C < 1 || C > 256 || d < 1 || E < 1 || H < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L
        + 4L * d * 256 * E
        + 4L * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }
    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Sum embeddings
        for (int e = 0; e < E; e++) sum_emb[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) sum_emb[e] += eptr[e];
        }
        // Hidden layer: ReLU(sum_emb * W1 + b1)
        for (int h = 0; h < H; h++) {
            float val = b1[h];
            for (int e = 0; e < E; e++) val += sum_emb[e] * W1[(long)e * H + h];
            hidden[h] = val > 0.0f ? val : 0.0f;
        }
        // Output layer
        for (int c = 0; c < C; c++) {
            float val = b2[c];
            for (int h = 0; h < H; h++) val += hidden[h] * W2[(long)h * C + c];
            logits[c] = val;
        }
        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```