```hypothesis
Family: label-embed-mlp. A shared byte embedding (256×32) compresses the 16-byte context into a 512-dim dense vector, followed by a ReLU hidden layer of 128 units and a linear output layer. This reduces parameters vs. a full one-hot MLP, captures semantic similarity between characters, and should exceed label-linear accuracy while being faster due to smaller matrix multiplies. If it underperforms label-linear, it would show that the embedding bottleneck loses too much information for this task.
```

```python
"""label-embed-mlp: shared byte embedding + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on tuning split (last 20% of train.in)
E = 32          # embedding dimension
H = 128         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 50     # fixed epochs (tuning split suggested 50 > 25)

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
    # Load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embedding: 256 x E
    emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    # Hidden layer: (d*E) x H
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    # Output layer: H x C
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_cross_entropy_grad(logits, y_idx):
        """Return gradient of loss w.r.t logits, shape (B, C)."""
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(y_idx)), y_idx] -= 1.0
        return (probs / len(y_idx)).astype(np.float32)

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)

            # Forward pass
            # Embedding lookup and concatenation
            B = xb.shape[0]
            concat = np.zeros((B, d * E), dtype=np.float32)
            for j in range(d):
                concat[:, j*E:(j+1)*E] = emb[xb[:, j]]
            hidden_pre = concat @ W1 + b1
            hidden = np.maximum(hidden_pre, 0, dtype=np.float32)
            logits = hidden @ W2 + b2

            # Loss gradient
            dlogits = softmax_cross_entropy_grad(logits, yb)

            # Backward pass
            dW2 = hidden.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)
            dhidden = (dlogits @ W2.T) * (hidden_pre > 0)
            dW1 = concat.T @ dhidden + L2 * W1
            db1 = dhidden.sum(axis=0)
            dconcat = dhidden @ W1.T

            # Embedding gradients
            demb = np.zeros_like(emb)
            for j in range(d):
                slice_j = dconcat[:, j*E:(j+1)*E]
                # Accumulate gradients for each byte value
                np.add.at(demb, xb[:, j], slice_j)

            grads = [demb, dW1, db1, dW2, db2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Format: magic, d, E, H, C (all int32), then float32 arrays in order:
    # emb (256*E), W1 (d*E*H), b1 (H), W2 (H*C), b2 (C)
    magic = 0x454D4244  # "EMBD" little-endian
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = emb + xi[j] * E;
            float *dst = concat + j * E;
            for (int e_idx = 0; e_idx < E; e_idx++) dst[e_idx] = e[e_idx];
        }

        // Hidden layer: hidden = relu(concat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k_idx = 0; k_idx < d * E; k_idx++) {
                sum += concat[k_idx] * W1[(long)k_idx * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: logits = hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[(long)h * C + c];
            }
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(concat); free(hidden); free(logits);
    return 0;
}
```