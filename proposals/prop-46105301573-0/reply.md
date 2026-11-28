```hypothesis
Family: label-embedding-mlp. Uses per-feature embeddings (dim=4) for all 14 columns, then a single ReLU hidden layer (32 units) and a 2-class output. Embeddings capture non-linear relationships in the quantized ordinal and categorical features better than one-hot + linear, while staying tiny (~65 KB). Hyperparameters (E=4, H=32, lr=1e-3, L2=1e-4, batch=64, epochs=50) chosen on the last 20% of train.in. If dev accuracy ≤ 0.8524 or inference cost > label-linear, the approach fails.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (tuned on the last 20% of train.in)
E = 4          # embedding dimension per feature
H = 32         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 50

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X_bytes.shape[0]

    # Split: last 20% for validation (file order)
    val_start = int(0.8 * n)
    train_idx = np.arange(val_start)
    val_idx = np.arange(val_start, n)

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: (d, 256, E)
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # MLP weights
    in_dim = d * E
    W1 = (rng.normal(0, math.sqrt(2.0 / in_dim), (in_dim, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X_idx):
        # X_idx: (batch, d) uint8
        batch = X_idx.shape[0]
        # Embedding lookup
        x_emb = emb[np.arange(d), X_idx.T]  # (d, batch, E)
        x_emb = x_emb.transpose(1, 0, 2).reshape(batch, in_dim)  # (batch, d*E)
        # Hidden layer
        h = x_emb @ W1 + b1
        h = np.maximum(h, 0)
        # Output
        out = h @ W2 + b2
        return out, h, x_emb

    def loss_grad(logits, y_batch):
        # Softmax cross-entropy gradient
        max_log = logits.max(axis=1, keepdims=True)
        exp_log = np.exp(logits - max_log)
        probs = exp_log / exp_log.sum(axis=1, keepdims=True)
        probs[np.arange(len(y_batch)), y_batch] -= 1.0
        grad = probs / len(y_batch)
        return grad.astype(np.float32)

    # Training loop
    for epoch in range(EPOCHS):
        order = rng.permutation(len(train_idx))
        for s in range(0, len(train_idx), BATCH):
            idx = train_idx[order[s:s+BATCH]]
            X_batch = X_bytes[idx]
            y_batch = y[idx]
            logits, h, x_emb = forward(X_batch)
            g_out = loss_grad(logits, y_batch)  # (batch, C)

            # Gradients
            grad_W2 = h.T @ g_out + L2 * W2
            grad_b2 = g_out.sum(axis=0)
            g_h = g_out @ W2.T * (h > 0)
            grad_W1 = x_emb.T @ g_h + L2 * W1
            grad_b1 = g_h.sum(axis=0)
            # Embedding gradients
            grad_emb = np.zeros_like(emb)
            g_emb = g_h @ W1.T  # (batch, d*E)
            g_emb = g_emb.reshape(BATCH, d, E)
            # Scatter-add
            np.add.at(grad_emb, (np.arange(d)[:, None], X_batch.T), g_emb.transpose(1, 0, 2))
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Save weights
    # Format: magic, d, E, H, C, then embeddings (d*256*E), W1 (in_dim*H), b1 (H), W2 (H*C), b2 (C)
    magic = 0x454D4244  # "EMBD"
    out = struct.pack("<5i", magic, d, E, H, C)
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
/* label-embedding-mlp inference: ./infer weights.bin in.rec N */
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
    if (C != 2 || d != 14 || E <= 0 || H <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 20L
        + 4L * d * 256 * E
        + 4L * d * E * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding concatenation + W1 + b1
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int h = 0; h < H; h++) {
                float sum = 0.0f;
                for (int ee = 0; ee < E; ee++) sum += e[ee] * W1[((long)j * E + ee) * H + h];
                hidden[h] += sum;
            }
        }
        // ReLU
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        // Output layer
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            for (int c = 0; c < C; c++) scores[c] += hv * W2[(long)h * C + c];
        }

        unsigned char best = scores[1] > scores[0] ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```