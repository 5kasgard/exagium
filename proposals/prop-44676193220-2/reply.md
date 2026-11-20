```hypothesis
Family: label-embmlp (position-specific embedding + one-hidden-layer MLP).
Why: Character-level language modeling benefits from distributed representations; embeddings capture semantic similarity between characters (e.g., letters vs punctuation), while a hidden ReLU layer models non-linear interactions across the 16-byte context. Position-specific embeddings preserve order information without the 4096-dim one-hot blowup.
Failure signal: Dev accuracy ≤ 0.34 (label-linear) or inference instructions/item > 15000.
```

```python
"""label-embmlp: position-specific embeddings + 1-hidden-layer ReLU MLP.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Weights format (little-endian):
  int32 magic=0x454D424C, d=16, E=32, H=128, C=65
  float32 emb[d][256][E]   (position-major)
  float32 W1[d*E][H], b1[H]
  float32 W2[H][C], b2[C]"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 32          # embedding dimension per position
H = 128         # hidden size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
SAMPLES = 600000  # same as label-linear


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
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    # Split: last 20% for tuning (validation), first 80% for training
    split = int(0.8 * n)
    X_tr, y_tr = X_bytes[:split], y[:split]
    X_va, y_va = X_bytes[split:], y[split:]
    n_tr = X_tr.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embeddings: d x 256 x E
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # First layer: (d*E) x H
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    # Second layer: H x C
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X, emb, W1, b1, W2, b2):
        # X: (batch, d) uint8
        batch = X.shape[0]
        # Embedding lookup: (batch, d, E)
        # We'll do it by indexing (vectorized)
        x_emb = emb[np.arange(d), X]  # (d, batch, E) -> transpose to (batch, d, E)
        x_emb = np.transpose(x_emb, (1, 0, 2)).reshape(batch, d * E)
        h = x_emb @ W1 + b1
        h_relu = np.maximum(h, 0)
        logits = h_relu @ W2 + b2
        return logits, h_relu, x_emb

    def softmax_grad(logits, y):
        # logits: (batch, C), y: (batch,)
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(y)), y] -= 1.0
        return probs / len(y)  # (batch, C)

    # Training loop
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
    for epoch in range(max_epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            logits, h_relu, x_emb = forward(Xb, emb, W1, b1, W2, b2)
            g = softmax_grad(logits, yb)  # (batch, C)

            # Gradients
            # W2: (H, C), b2: (C,)
            grad_W2 = h_relu.T @ g + L2 * W2
            grad_b2 = g.sum(axis=0)

            # Backprop through ReLU
            gh = g @ W2.T * (h_relu > 0)  # (batch, H)

            # W1: (d*E, H), b1: (H,)
            grad_W1 = x_emb.T @ gh + L2 * W1
            grad_b1 = gh.sum(axis=0)

            # Embedding gradients: we need to scatter-add per position
            # gh: (batch, H) -> gx_emb = gh @ W1.T  (batch, d*E)
            gx_emb = gh @ W1.T  # (batch, d*E)
            gx_emb = gx_emb.reshape(len(idx), d, E)  # (batch, d, E)
            grad_emb = np.zeros_like(emb)
            # For each position j, accumulate gradients for the indices Xb[:, j]
            for j in range(d):
                np.add.at(grad_emb[j], Xb[:, j], gx_emb[:, j, :])
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

    # After training, write weights.bin
    # Header: magic, d, E, H, C
    header = struct.pack("<5i", 0x454D424C, d, E, H, C)
    # Data: emb (d,256,E), W1 (d*E,H), b1 (H), W2 (H,C), b2 (C)
    data = (emb.astype("<f4").tobytes() +
            W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes() +
            W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + data)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format (little-endian):
 *   int32 magic=0x454D424C, d=16, E=32, H=128, C=65
 *   float32 emb[d][256][E]
 *   float32 W1[d*E][H], b1[H]
 *   float32 W2[H][C], b2[C] */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected_floats = (long)d * 256 * E + (long)d * E * H + H + (long)H * C + C;
    long expected_bytes = 20 + 4 * expected_floats;
    if (nw != expected_bytes) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected_bytes); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_emb = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(x_emb + j * E, e, E * sizeof(float));
        }

        // Hidden layer: ReLU(x_emb * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) sum += x_emb[k] * W1[(long)k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden * W2 + b2
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