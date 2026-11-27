```hypothesis
We propose "label-embed", a factorized neural network with per-position byte embeddings summed and passed through a single hidden ReLU layer. This captures positional character interactions more efficiently than a full one-hot linear model, and should exceed label-linear's 0.3375 accuracy on dev while keeping inference cost moderate (target < 15k Ir/item). If dev accuracy is ≤ 0.34 or cost > 20k Ir/item, the family does not meet its goal.
```

```python
"""label-embed: factorized embedding + one hidden ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: magic 0x454D4244, int32 d, D, H, C; float32 embeddings[d][256][D], W1[D][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
D = 32          # embedding dimension
H = 64          # hidden layer size
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: per position (d) a 256 x D matrix
    # Use Xavier uniform for embeddings
    embeddings = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / (256 + D))).astype(np.float32)

    # Hidden layer: D -> H
    W1 = (rng.uniform(-1, 1, (D, H)) * math.sqrt(6.0 / D)).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)

    # Output layer: H -> C
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [embeddings, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward
            # Sum embeddings: (B, d, D) -> (B, D)
            emb = embeddings[np.arange(d), xb.T]  # (d, B, D)
            emb_sum = np.transpose(emb, (1, 0, 2)).sum(axis=1)  # (B, D)

            hidden_pre = emb_sum @ W1 + b1          # (B, H)
            hidden = np.maximum(hidden_pre, 0)      # ReLU
            logits = hidden @ W2 + b2               # (B, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B                 # (B, C)

            # Backprop output layer
            grad_W2 = hidden.T @ grad_logits + L2 * W2   # (H, C)
            grad_b2 = grad_logits.sum(axis=0)            # (C,)

            grad_hidden = grad_logits @ W2.T             # (B, H)
            grad_hidden[hidden_pre <= 0] = 0.0           # ReLU gradient

            # Backprop hidden layer
            grad_W1 = emb_sum.T @ grad_hidden + L2 * W1  # (D, H)
            grad_b1 = grad_hidden.sum(axis=0)            # (H,)

            grad_emb_sum = grad_hidden @ W1.T            # (B, D)

            # Backprop embeddings: scatter add per position
            grad_embeddings = np.zeros_like(embeddings)
            for j in range(d):
                # grad_emb_sum[:,:] added to position j for each byte value
                # We accumulate using np.add.at for each batch item
                np.add.at(grad_embeddings[j], xb[:, j], grad_emb_sum)

            grads = [grad_embeddings, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Header: magic, d, D, H, C (5 int32)
    header = struct.pack("<5i", 0x454D4244, d, D, H, C)
    # Embeddings: d * 256 * D floats
    emb_bytes = embeddings.astype("<f4").tobytes()
    # W1, b1, W2, b2
    w1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    w2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()

    out = header + emb_bytes + w1_bytes + b1_bytes + w2_bytes + b2_bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Each input record is d bytes (d=16); output is one byte class index. */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }

    // Expected weight size:
    // embeddings: d * 256 * D
    // W1: D * H
    // b1: H
    // W2: H * C
    // b2: C
    long expected = 20
        + 4L * d * 256 * D
        + 4L * D * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + d * 256 * D;
    const float *b1 = W1 + D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * D);
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Initialize emb_sum to zero
        for (int j = 0; j < D; j++) emb_sum[j] = 0.0f;

        // Sum embeddings for each position
        for (int j = 0; j < d; j++) {
            unsigned char byte_val = xi[j];
            const float *emb_j = emb + (j * 256 + byte_val) * D;
            for (int dd = 0; dd < D; dd++) emb_sum[dd] += emb_j[dd];
        }

        // Hidden layer: ReLU(emb_sum * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int dd = 0; dd < D; dd++) sum += emb_sum[dd] * W1[dd * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[h * C + c];
            scores[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb_sum); free(hidden); free(scores);
    return 0;
}
```