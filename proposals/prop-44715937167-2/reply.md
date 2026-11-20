```hypothesis
Family: label-embmlp (embedding MLP with one hidden ReLU layer).
Why: The 16-byte context benefits from modeling interactions between positions. A linear model on one-hot features (label-linear) cannot capture these. Using per-position embeddings (dim=16) reduces input dimension from 4096 to 256, then a hidden layer (dim=64) captures non-linear interactions. This should improve accuracy over label-linear (0.3375) while keeping inference cost low (~20k multiply-adds vs 266k for linear).
Failure: Dev accuracy ≤ 0.3375 or inference ns/item ≥ label-linear (665.8 ns/item).
```

```python
"""label-embmlp: Embedding MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: per-position embeddings (16 positions x 256 vocab x emb_dim),
concatenated -> ReLU hidden layer (hidden_dim) -> linear output (65 classes).
Adam lr=3e-3, L2=1e-4, batch=64, target samples=600k, max epochs=100.
Weights format: magic 0x454D424D, int32 d, emb_dim, hidden_dim, C;
float32 emb[d][256][emb_dim], W1[d*emb_dim][hidden_dim], b1[hidden_dim],
W2[hidden_dim][C], b2[C] (all row-major)."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (tuned on tuning split)
EMB_DIM = 16
HIDDEN_DIM = 64
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
TARGET_SAMPLES = 600000
MAX_EPOCHS = 100
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
MAGIC = 0x454D424D  # 'EMBM'


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: (d, 256, EMB_DIM)
    emb = rng.normal(0, 0.01, size=(d, 256, EMB_DIM)).astype(np.float32)
    # W1: (d*EMB_DIM, HIDDEN_DIM)  He init for ReLU
    fan_in = d * EMB_DIM
    W1 = rng.normal(0, math.sqrt(2.0 / fan_in), size=(fan_in, HIDDEN_DIM)).astype(np.float32)
    b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
    # W2: (HIDDEN_DIM, C)  Xavier init
    W2 = rng.normal(0, math.sqrt(2.0 / (HIDDEN_DIM + C)), size=(HIDDEN_DIM, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state: m, v for each param
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    epochs = min(MAX_EPOCHS, math.ceil(TARGET_SAMPLES / n))
    t = 0

    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward
            # Embedding lookup: (B, d, EMB_DIM)
            emb_seq = emb[np.arange(d)[None, :], xb]   # (B, d, EMB_DIM)
            concat = emb_seq.reshape(B, -1)            # (B, d*EMB_DIM)
            hidden_pre = concat @ W1 + b1              # (B, HIDDEN_DIM)
            hidden = np.maximum(hidden_pre, 0)         # ReLU
            logits = hidden @ W2 + b2                  # (B, C)

            # Softmax cross-entropy loss gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dlogits = probs / B                        # (B, C)

            # Backward
            dW2 = hidden.T @ dlogits                   # (HIDDEN_DIM, C)
            db2 = dlogits.sum(axis=0)                  # (C,)
            dhidden = dlogits @ W2.T                   # (B, HIDDEN_DIM)
            dhidden_pre = dhidden * (hidden_pre > 0)   # (B, HIDDEN_DIM)
            dW1 = concat.T @ dhidden_pre               # (d*EMB_DIM, HIDDEN_DIM)
            db1 = dhidden_pre.sum(axis=0)              # (HIDDEN_DIM,)
            dconcat = dhidden_pre @ W1.T               # (B, d*EMB_DIM)

            # Gradient for embeddings: scatter dconcat back
            demb = np.zeros_like(emb)
            dconcat_reshaped = dconcat.reshape(B, d, EMB_DIM)
            for j in range(d):
                np.add.at(demb[j], xb[:, j], dconcat_reshaped[:, j])

            # Add L2 regularization gradients (not on biases)
            demb += L2 * emb
            dW1 += L2 * W1
            dW2 += L2 * W2

            grads = [demb, dW1, db1, dW2, db2]

            # Adam update
            t += 1
            lr_t = LR * math.sqrt(1 - BETA2 ** t) / (1 - BETA1 ** t)
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= BETA1; m += (1 - BETA1) * gr
                v *= BETA2; v += (1 - BETA2) * gr * gr
                p -= lr_t * m / (np.sqrt(v) + EPS)

    # Pack weights
    # header: magic, d, emb_dim, hidden_dim, C
    header = struct.pack("<5i", MAGIC, d, EMB_DIM, HIDDEN_DIM, C)
    # flatten in order: emb, W1, b1, W2, b2 (all float32 little-endian)
    body = (
        emb.astype("<f4").tobytes() +
        W1.astype("<f4").tobytes() +
        b1.astype("<f4").tobytes() +
        W2.astype("<f4").tobytes() +
        b2.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format: magic 0x454D424D, int32 d, emb_dim, hidden_dim, C;
 * float32 emb[d][256][emb_dim], W1[d*emb_dim][hidden_dim], b1[hidden_dim],
 * W2[hidden_dim][C], b2[C] (row-major). */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], emb_dim = hd[2], hidden_dim = hd[3], C = hd[4];
    if (d != 16 || C != 65 || emb_dim <= 0 || hidden_dim <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * emb_dim +
                                (long)d * emb_dim * hidden_dim +
                                hidden_dim +
                                (long)hidden_dim * C +
                                C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * emb_dim;
    const float *b1 = W1 + (long)d * emb_dim * hidden_dim;
    const float *W2 = b1 + hidden_dim;
    const float *b2 = W2 + (long)hidden_dim * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float concat[16 * 16];        // max d*emb_dim = 16*16 = 256
    float hidden_pre[64];         // max hidden_dim = 64
    float hidden[64];
    float logits[65];             // C = 65

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * emb_dim;
            memcpy(concat + j * emb_dim, src, emb_dim * sizeof(float));
        }

        // Hidden layer: hidden_pre = concat * W1 + b1
        for (int h = 0; h < hidden_dim; h++) {
            float sum = b1[h];
            for (int idx = 0; idx < d * emb_dim; idx++) {
                sum += concat[idx] * W1[(long)idx * hidden_dim + h];
            }
            hidden_pre[h] = sum;
            hidden[h] = sum > 0.0f ? sum : 0.0f;  // ReLU
        }

        // Output layer: logits = hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < hidden_dim; h++) {
                sum += hidden[h] * W2[(long)h * C + c];
            }
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (logits[c] > logits[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```