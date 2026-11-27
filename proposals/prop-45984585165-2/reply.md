```hypothesis
Family: pos-mlp (positional embeddings + 1-hidden-layer MLP). Each of the 16 context positions gets a learned embedding (dim D=16), concatenated and passed through a ReLU layer (H=64) to 65 logits. Captures position-specific character statistics and pairwise interactions missed by linear model, while remaining fast (O(D*d*H + H*C) per item). Fails if embeddings overfit or ReLU capacity insufficient for character-level dependencies.
```

```python
"""pos-mlp: Positional embeddings + 1-hidden-layer ReLU MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weight format: magic 0x504F534D, int32 d,D,H,C; float32 E[d][256][D], W1[d*D][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen a priori; could be tuned on last 20% of train.in)
D = 16          # embedding dimension per position
H = 64          # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 30
CLIP = 1.0      # gradient clipping

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    """Return gradient of cross-entropy w.r.t logits (already averaged over batch)."""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits, dtype=np.float32)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 16 and C == 65

    # Load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Deterministic RNG
    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embeddings: [d, 256, D]
    E = rng.normal(0, 0.01, size=(d, 256, D)).astype(np.float32)
    # W1: [d*D, H] (Glorot uniform)
    limit = math.sqrt(6.0 / (d * D + H))
    W1 = rng.uniform(-limit, limit, size=(d * D, H)).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    # W2: [H, C]
    limit2 = math.sqrt(6.0 / (H + C))
    W2 = rng.uniform(-limit2, limit2, size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [E, W1, b1, W2, b2]
    # Adam state: m, v for each param
    adam_m = [np.zeros_like(p) for p in params]
    adam_v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    # Training loop
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # [B, d]
            yb = y[idx]          # [B]

            # Forward: embeddings lookup
            # emb = E[positions, xb] -> [B, d, D]
            # We can do: for each position, gather rows of E[pos]
            # Vectorized: emb = E[np.arange(d), xb.T].transpose(1,0,2) but let's do loop for clarity and memory
            B = xb.shape[0]
            emb = np.empty((B, d, D), dtype=np.float32)
            for pos in range(d):
                emb[:, pos, :] = E[pos, xb[:, pos]]
            x = emb.reshape(B, d * D)  # [B, d*D]

            # Hidden layer
            h_pre = x @ W1 + b1
            h = np.maximum(h_pre, 0, dtype=np.float32)

            # Output layer
            logits = h @ W2 + b2

            # Loss gradient
            g_logits = softmax_cross_entropy_grad(logits, yb)  # [B, C]

            # Backprop
            g_W2 = h.T @ g_logits + L2 * W2
            g_b2 = g_logits.sum(axis=0)
            g_h = g_logits @ W2.T
            g_h_pre = g_h * (h_pre > 0)
            g_W1 = x.T @ g_h_pre + L2 * W1
            g_b1 = g_h_pre.sum(axis=0)
            # Gradient for embeddings
            g_x = g_h_pre @ W1.T  # [B, d*D]
            g_emb = g_x.reshape(B, d, D)
            g_E = np.zeros_like(E)
            for pos in range(d):
                # Accumulate gradients for each byte value at this position
                np.add.at(g_E[pos], xb[:, pos], g_emb[:, pos, :])
            g_E += L2 * E

            grads = [g_E, g_W1, g_b1, g_W2, g_b2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                # Clip gradient
                if CLIP:
                    norm = np.sqrt(np.sum(gr * gr))
                    if norm > CLIP:
                        gr = gr * (CLIP / norm)
                m, v = adam_m[i], adam_v[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Serialize weights
    # Magic: 0x504F534D ("POSM" little-endian)
    # d, D, H, C as int32
    # E: [d, 256, D] float32 (row-major: pos, byte, dim)
    # W1: [d*D, H] float32
    # b1: [H] float32
    # W2: [H, C] float32
    # b2: [C] float32
    header = struct.pack("<5i", 0x504F534D, d, D, H, C)
    body = (
        E.astype("<f4").tobytes() +
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
/* pos-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weight format: magic 0x504F534D, int32 d,D,H,C; float32 E[d][256][D], W1[d*D][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x504F534D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }

    // Expected weight size: header(20) + E[d*256*D] + W1[d*D*H] + b1[H] + W2[H*C] + b2[C]
    long expected = 20 +
        4L * d * 256 * D +
        4L * d * D * H +
        4L * H +
        4L * H * C +
        4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *E = (const float *)(hd + 5);
    const float *W1 = E + (long)d * 256 * D;
    const float *b1 = W1 + (long)d * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_concat = malloc(sizeof(float) * d * D);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int pos_idx = 0; pos_idx < d; pos_idx++) {
            const float *e = E + ((long)pos_idx * 256 + xi[pos_idx]) * D;
            memcpy(emb_concat + pos_idx * D, e, D * sizeof(float));
        }

        // Hidden layer: ReLU(emb_concat @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int j = 0; j < d * D; j++) {
                sum += emb_concat[j] * W1[(long)j * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden @ W2 + b2
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

    free(emb_concat);
    free(hidden);
    free(logits);
    return 0;
}
```