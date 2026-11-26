```hypothesis
Family: label-embed-mlp. Shared 256x16 byte embeddings concatenated over 16 positions (256-d) feed a 64-unit ReLU hidden layer, then 65-class output. Embeddings capture byte similarities; MLP models interactions. Beats linear by learning distributed representations. If dev accuracy ≤ 0.34, the embedding+MLP adds no value over linear.
```

```python
"""label-embed-mlp: shared byte embeddings + one ReLU layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x454D4C31, int32 d=16, E, H, C; float32 emb[256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
E = 16          # embedding dimension
H = 64          # hidden units
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    # Load all training data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Use last 20% as tuning split for early stopping / hyperparameter choice (here fixed)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    n_train = X_train.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embeddings: 256 x E
    emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    # W1: (d*E) x H
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: H x C
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X_batch):
        # X_batch: (batch, d) uint8
        batch = X_batch.shape[0]
        # Embedding lookup: (batch, d, E)
        x_emb = emb[X_batch]            # (batch, d, E)
        x = x_emb.reshape(batch, d * E) # (batch, d*E)
        h_pre = x @ W1 + b1             # (batch, H)
        h = np.maximum(h_pre, 0)
        logits = h @ W2 + b2            # (batch, C)
        return x_emb, x, h_pre, h, logits

    def loss_grad(logits, y_batch):
        batch = logits.shape[0]
        # Softmax + cross-entropy gradient
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(batch), y_batch] -= 1.0
        return (probs / batch).astype(np.float32)   # dlogits

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))
    best_val_acc = -1.0
    best_params = None

    for epoch in range(max_epochs):
        order = rng.permutation(n_train)
        for s in range(0, n_train, BATCH):
            idx = order[s:s + BATCH]
            X_batch, y_batch = X_train[idx], y_train[idx]
            batch = len(idx)

            x_emb, x, h_pre, h, logits = forward(X_batch)
            dlogits = loss_grad(logits, y_batch)

            # Gradients
            dW2 = h.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)
            dh = dlogits @ W2.T
            dh[h_pre <= 0] = 0.0
            dW1 = x.T @ dh + L2 * W1
            db1 = dh.sum(axis=0)
            dx = dh @ W1.T                          # (batch, d*E)
            dx_emb = dx.reshape(batch, d, E)        # (batch, d, E)

            # Embedding gradients: accumulate per byte value
            demb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(demb, X_batch[:, j], dx_emb[:, j, :])
            demb += L2 * emb

            grads = [demb, dW1, db1, dW2, db2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Validation accuracy (optional, for early stopping)
        _, _, _, h_val, logits_val = forward(X_val)
        val_pred = logits_val.argmax(axis=1)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]

    # Use best params (or last if no improvement)
    if best_params is not None:
        emb, W1, b1, W2, b2 = best_params

    # Write weights.bin
    magic = 0x454D4C31  # 'EML1'
    header = struct.pack("<4i", magic, d, E, H, C)
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
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 20 || hd[0] != 0x454D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20
        + 4L * 256 * E
        + 4L * d * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_emb = malloc(sizeof(float) * d * E);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + xi[j] * E;
            memcpy(x_emb + j * E, e, E * sizeof(float));
        }

        // Hidden layer: h = ReLU(x_emb @ W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int k = 0; k < d * E; k++) sum += x_emb[k] * W1[k * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: logits = h @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[u * C + c];
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(x_emb); free(h); free(logits);
    if (x) free(x);
    return 0;
}
```