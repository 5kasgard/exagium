```hypothesis
We propose "label-emb-mlp": an embedding-based MLP with one hidden layer. Each of the 16 context bytes is mapped to a learned embedding vector (dimension E), concatenated, then passed through a ReLU hidden layer (size H) and a linear output layer (65 classes). This captures positional interactions more efficiently than a linear model on one-hot encodings, reducing parameters from ~266k to ~75k (with E=16, H=32) and inference cost. We expect higher accuracy than label-linear (0.3375) at similar or lower CPU cost. If dev accuracy does not exceed 0.34, the hypothesis fails.
```

```python
"""Embedding MLP family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Chooses embedding dim E and hidden size H on the last 20% of train.in, then trains on all data.
Exports: int32 magic 0x44424D45, d, E, H, C; float32 embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning (E, H)
GRID = [(8, 16), (8, 32), (8, 64), (16, 16), (16, 32), (16, 64)]
SEED = 1
LR, L2, BATCH = 3e-3, 1e-4, 64
SAMPLES, MAX_EPOCHS = 600000, 100
TUNE_EPOCHS = 5  # quick tuning epochs


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_cross_entropy_grad(logits, y):
    """Return gradient of cross-entropy w.r.t logits (already averaged over batch)."""
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)


def train_model(X, y, X_val, y_val, E, H, epochs, seed):
    """Train on (X,y), validate on (X_val,y_val). Return (params, val_acc)."""
    n, d = X.shape
    C = 65
    rng = np.random.default_rng(seed)

    # Initialize parameters
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 * E))).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X_batch):
        # X_batch: (batch, d) uint8
        batch = X_batch.shape[0]
        # Gather embeddings: (batch, d, E)
        emb_batch = np.stack([emb[j, X_batch[:, j]] for j in range(d)], axis=1)
        concat = emb_batch.reshape(batch, d * E)  # (batch, d*E)
        h_pre = concat @ W1 + b1
        h = np.maximum(h_pre, 0)
        logits = h @ W2 + b2
        return concat, h_pre, h, logits

    def backward(concat, h_pre, h, grad_logits):
        # grad_logits: (batch, C)
        batch = grad_logits.shape[0]
        # Output layer gradients
        dW2 = h.T @ grad_logits + L2 * W2
        db2 = grad_logits.sum(axis=0)
        dh = grad_logits @ W2.T
        # ReLU
        dh_pre = dh * (h_pre > 0)
        # Hidden layer gradients
        dW1 = concat.T @ dh_pre + L2 * W1
        db1 = dh_pre.sum(axis=0)
        dconcat = dh_pre @ W1.T  # (batch, d*E)
        # Embedding gradients: scatter-add
        demb = np.zeros_like(emb)
        dconcat_split = dconcat.reshape(batch, d, E)
        for j in range(d):
            np.add.at(demb[j], X_batch[:, j], dconcat_split[:, j])
        return [demb, dW1, db1, dW2, db2]

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            X_batch, y_batch = X[idx], y[idx]
            concat, h_pre, h, logits = forward(X_batch)
            grad_logits = softmax_cross_entropy_grad(logits, y_batch)
            grads = backward(concat, h_pre, h, grad_logits)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Validation accuracy
    concat, h_pre, h, logits = forward(X_val)
    pred = logits.argmax(axis=1)
    val_acc = (pred == y_val).mean()
    return params, val_acc


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(X)

    # Split: last 20% for tuning
    tune_size = int(0.2 * n)
    train_X, train_y = X[:-tune_size], y[:-tune_size]
    tune_X, tune_y = X[-tune_size:], y[-tune_size:]

    # Hyperparameter search on tuning split
    best_acc, best_params, best_EH = -1, None, None
    tune_epochs = min(TUNE_EPOCHS, math.ceil(SAMPLES / len(train_X)))
    for E, H in GRID:
        params, acc = train_model(train_X, train_y, tune_X, tune_y, E, H, tune_epochs, SEED)
        if acc > best_acc:
            best_acc, best_params, best_EH = acc, params, (E, H)
        print(f"E={E} H={H} tune_acc={acc:.4f}", file=sys.stderr)

    E, H = best_EH
    print(f"Selected E={E} H={H} with tune_acc={best_acc:.4f}", file=sys.stderr)

    # Final training on all data
    final_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    final_params, _ = train_model(X, y, X[:1], y[:1], E, H, final_epochs, SEED)  # dummy val
    emb, W1, b1, W2, b2 = final_params

    # Write weights.bin
    magic = 0x44424D45  # 'EMBD' little-endian
    header = struct.pack("<5i", magic, d, E, H, C)
    out = header
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
/* Embedding MLP inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index (0..64).
 * Weight format: int32 magic 0x44424D45, d, E, H, C;
 * float32 embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x44424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }

    long expected_floats = (long)d * 256 * E + (long)d * E * H + H + (long)H * C + C;
    if (nw != 20 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            float *dst = concat + j * E;
            memcpy(dst, src, E * sizeof(float));
        }

        // Hidden layer: hidden = relu(concat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float acc = b1[h];
            for (int idx = 0; idx < d * E; idx++) {
                acc += concat[idx] * W1[(long)idx * H + h];
            }
            hidden[h] = acc > 0.0f ? acc : 0.0f;
        }

        // Output layer: scores = hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float acc = b2[c];
            for (int h = 0; h < H; h++) {
                acc += hidden[h] * W2[(long)h * C + c];
            }
            scores[c] = acc;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```