```hypothesis
Family: label-embedmlp (shared character embeddings + one ReLU hidden layer).
Why: The 16-byte context uses a small vocabulary (65 chars). One-hot linear models ignore character similarities; embeddings capture them and drastically reduce input dimensionality (256*D vs 16*256). A modest hidden layer (H=64) then models position interactions efficiently. This should exceed label-linear's 0.3375 accuracy while keeping inference fast (≈25k multiply-adds/item).
Failure: Dev accuracy ≤ 0.3375 or inference cost (Ir/item) > 2× label-linear.
```

```python
"""label-embedmlp: shared byte embeddings + one ReLU layer for next-char prediction.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only.
Weights format (little-endian):
  magic(0x454D4244), d=16, D, H, C=65
  E[256*D], W1[16*D*H], b1[H], W2[H*65], b2[65]  (all float32)"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
SEED = 1
D = 16          # embedding dimension
H = 64          # hidden units
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
    C = card["classes"]
    d = X.shape[1]          # 16
    N = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embedding matrix: 256 x D
    E = (rng.normal(0, 1.0/math.sqrt(D), (256, D))).astype(np.float32)
    # Hidden layer: (16*D) x H
    W1 = (rng.normal(0, math.sqrt(2.0/(16*D)), (16*D, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer: H x C
    W2 = (rng.normal(0, math.sqrt(2.0/H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [E, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    # Precompute embedding indices for all positions
    # X shape (N, 16), each entry 0..255

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / N))
    for _ in range(epochs):
        order = rng.permutation(N)
        for s in range(0, N, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]          # (B, 16)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward
            # Embeddings: for each position, gather rows of E
            emb = np.empty((B, 16*D), np.float32)
            for j in range(16):
                emb[:, j*D:(j+1)*D] = E[xb[:, j]]
            hidden_pre = emb @ W1 + b1          # (B, H)
            hidden = np.maximum(hidden_pre, 0)  # ReLU
            logits = hidden @ W2 + b2           # (B, C)

            # Softmax + cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B                            # dL/dlogits

            # Backward
            # Output layer
            dW2 = hidden.T @ probs + L2 * W2
            db2 = probs.sum(axis=0)
            dhidden = probs @ W2.T                # (B, H)
            # ReLU
            dhidden[hidden_pre <= 0] = 0
            # Hidden layer
            dW1 = emb.T @ dhidden + L2 * W1
            db1 = dhidden.sum(axis=0)
            # Embeddings gradient
            demb = dhidden @ W1.T                 # (B, 16*D)
            dE = np.zeros_like(E)
            for j in range(16):
                np.add.at(dE, xb[:, j], demb[:, j*D:(j+1)*D])
            dE += L2 * E

            grads = [dE, dW1, db1, dW2, db2]

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * g * g
                m_hat = m[i] / (1 - beta1**t)
                v_hat = v[i] / (1 - beta2**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Pack weights
    magic = 0x454D4244  # 'EMBD'
    header = struct.pack("<5i", magic, d, D, H, C)
    body = (E.astype("<f4").tobytes() +
            W1.astype("<f4").tobytes() +
            b1.astype("<f4").tobytes() +
            W2.astype("<f4").tobytes() +
            b2.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-embedmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte (class index 0..64).
 * Weights format (little-endian):
 *   uint32 magic=0x454D4244, int32 d=16, D, H, C=65
 *   float32 E[256*D], W1[16*D*H], b1[H], W2[H*65], b2[65] */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 20L + 4L*(256L*D + 16L*D*H + H + (long)H*C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *E = (const float *)(hd + 5);
    const float *W1 = E + 256L*D;
    const float *b1 = W1 + 16L*D*H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H*C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *emb = (float *)malloc(16 * D * sizeof(float));
    float *hidden = (float *)malloc(H * sizeof(float));
    float *logits = (float *)malloc(C * sizeof(float));
    if (!emb || !hidden || !logits) { fprintf(stderr, "alloc failed\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup
        for (int j = 0; j < 16; j++) {
            const float *src = E + (long)xi[j] * D;
            float *dst = emb + j * D;
            memcpy(dst, src, D * sizeof(float));
        }

        // Hidden layer: ReLU(emb * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k2 = 0; k2 < 16*D; k2++) sum += emb[k2] * W1[(long)k2 * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb); free(hidden); free(logits);
    if (x) free(x);
    return 0;
}
```