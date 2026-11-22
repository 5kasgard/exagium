```hypothesis
We propose "label-embmlp": per-position character embeddings (size 8) fed into a single ReLU hidden layer (size 32). This captures character similarities (e.g., 'a'≈'b') and nonlinear interactions that the linear model misses, while reducing input dimension from 4096 to 128. The model should exceed label-linear's 0.3375 dev accuracy. If dev accuracy ≤ 0.3375 or inference cost exceeds label-linear by >2×, the hypothesis fails.
```

```python
"""label-embmlp: per-position embeddings + one ReLU layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x454D424D, int32 d,E,H,C; float32 emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
SEED = 1
E = 8          # embedding size per position
H = 32         # hidden layer size
LR = 1e-3
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters (float32)
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)      # (d, 256, E)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # Precompute flat indices for embedding lookup: (n, d) -> (n, d) flat index into emb_flat (d*256, E)
    pos_offset = np.arange(d, dtype=np.int32) * 256
    flat_idx = X.astype(np.int32) + pos_offset  # (n, d)

    emb_flat = emb.reshape(d * 256, E)  # view for scattering

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # ---- Forward ----
            # Embedding lookup: (B, d, E)
            fidx = flat_idx[idx]                 # (B, d)
            embedded = emb_flat[fidx]            # (B, d, E)
            flat_emb = embedded.reshape(B, d * E) # (B, d*E)

            hidden_pre = flat_emb @ W1 + b1      # (B, H)
            hidden = np.maximum(hidden_pre, 0)   # ReLU
            logits = hidden @ W2 + b2            # (B, C)

            # Softmax cross-entropy
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            # Gradient of loss w.r.t logits: (probs - one_hot) / B
            dlogits = probs
            dlogits[np.arange(B), yb] -= 1.0
            dlogits /= B

            # ---- Backward ----
            # Output layer
            dW2 = hidden.T @ dlogits             # (H, C)
            db2 = dlogits.sum(axis=0)            # (C,)
            dhidden = dlogits @ W2.T             # (B, H)
            dhidden[hidden_pre <= 0] = 0.0       # ReLU gradient

            # Hidden layer
            dW1 = flat_emb.T @ dhidden           # (d*E, H)
            db1 = dhidden.sum(axis=0)            # (H,)
            dembedded_flat = dhidden @ W1.T      # (B, d*E)

            # Embedding gradients (scatter)
            dembedded = dembedded_flat.reshape(B, d, E)   # (B, d, E)
            grad_emb_flat = np.zeros_like(emb_flat)       # (d*256, E)
            # Use np.add.at for accumulation
            np.add.at(grad_emb_flat, fidx.reshape(-1), dembedded.reshape(-1, E))
            demb = grad_emb_flat.reshape(d, 256, E)

            # L2 regularization gradients
            demb += L2 * emb
            dW1 += L2 * W1
            dW2 += L2 * W2

            grads = [demb, dW1, db1, dW2, db2]

            # ---- Adam update ----
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # ---- Write weights ----
    # Header: magic, d, E, H, C
    header = struct.pack("<5i", 0x454D424D, d, E, H, C)
    # Data: emb, W1, b1, W2, b2 (all float32 little-endian)
    data = b""
    data += emb.astype("<f4").tobytes()
    data += W1.astype("<f4").tobytes()
    data += b1.astype("<f4").tobytes()
    data += W2.astype("<f4").tobytes()
    data += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + data)

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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat_emb = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            float *dst = flat_emb + j * E;
            memcpy(dst, src, E * sizeof(float));
        }

        // Hidden layer: ReLU(flat_emb @ W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int k = 0; k < d * E; k++) sum += flat_emb[k] * W1[(long)k * H + u];
            hidden[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += hidden[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```