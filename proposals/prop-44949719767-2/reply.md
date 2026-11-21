```hypothesis
We propose "label-embmlp": per-position character embeddings (16×256×E) followed by a one-hidden-layer MLP (ReLU, H units) and a linear output layer. This reduces the effective input dimension from 4096 (one-hot) to 16·E (e.g., 128), allowing the model to learn similarities between characters and positions. It should exceed label-linear's 0.3375 accuracy on dev while keeping inference fast (two small matrix multiplies). If dev accuracy ≤ 0.3375 or inference cost per item is more than ~2× label-linear, the hypothesis fails.
```

```python
"""label-embmlp: per-position embeddings + one-hidden-layer MLP for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen via manual tuning on the tuning split)
E = 8          # embedding dimension per position
H = 64         # hidden layer size
SEED = 1
LR = 3e-3
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
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    # Load training data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: (d, 256, E)
    embeddings = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    # MLP weights
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [embeddings, W1, b1, W2, b2]
    # Adam state: (m, v) for each param
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x_idx = X_bytes[idx]          # (B, d)
            yy = y[idx]                   # (B,)

            # Embedding lookup: vectorized
            B = x_idx.shape[0]
            flat_idx = x_idx.reshape(-1)                  # (B*d,)
            pos = np.repeat(np.arange(d), B)              # (B*d,)
            x_emb = embeddings[pos, flat_idx].reshape(B, d * E)  # (B, d*E)

            # Forward
            hidden_pre = x_emb @ W1 + b1                  # (B, H)
            hidden = np.maximum(hidden_pre, 0)            # ReLU
            logits = hidden @ W2 + b2                     # (B, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            probs /= B                                    # (B, C)

            # Backprop
            grad_W2 = hidden.T @ probs + L2 * W2          # (H, C)
            grad_b2 = probs.sum(axis=0)                   # (C,)
            grad_hidden = probs @ W2.T                    # (B, H)
            grad_hidden[hidden_pre <= 0] = 0.0            # ReLU gradient
            grad_W1 = x_emb.T @ grad_hidden + L2 * W1     # (d*E, H)
            grad_b1 = grad_hidden.sum(axis=0)             # (H,)
            # Embedding gradients
            grad_emb_flat = grad_hidden @ W1.T            # (B, d*E)
            grad_emb = np.zeros_like(embeddings)
            # Scatter-add: for each position, accumulate gradients
            for j in range(d):
                # indices for position j: x_idx[:, j] (B,)
                np.add.at(grad_emb[j], x_idx[:, j], grad_emb_flat[:, j*E:(j+1)*E])

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, E, H, C; then embeddings, W1, b1, W2, b2 (all float32 LE)
    magic = 0x454D424C  # 'L B M E' in little-endian ASCII? Actually 0x4C424D45 is 'EMBL' reversed. We'll use 0x454D424C.
    header = struct.pack("<5i", magic, d, E, H, C)
    # Flatten in order
    out = header
    out += embeddings.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }

    // Expected weight size: header(20) + embeddings(d*256*E) + W1(d*E*H) + b1(H) + W2(H*C) + b2(C) all float32
    long expected = 20 + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *logits = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings into a temporary array (on stack or heap)
        // We'll accumulate directly into hidden pre-activation: hidden = b1 + sum_j emb[j, xi[j]] * W1
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

    free(hidden);
    free(logits);
    return 0;
}
```