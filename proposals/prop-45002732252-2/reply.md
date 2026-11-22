```hypothesis
Family: label-emb-mlp (positional embeddings + one ReLU layer).
Why: The 16-byte context has strong positional patterns; learned embeddings capture per-position character statistics better than one-hot linear, and a hidden layer models interactions. With E=8, H=64 the model stays tiny (~180 KiB) and inference is a few dot products.
Failure: If dev accuracy ≤ 0.34 (label-linear) or latency ≥ 2× label-linear, the family does not improve the frontier.
```

```python
"""label-emb-mlp: positional byte embeddings + one ReLU layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Weights format (little-endian):
  magic 0x454D424D ('EMBM'), d, E, H, C (int32)
  embed[d*256*E] float32, W1[d*E*H] float32, b1[H] float32, W2[H*C] float32, b2[C] float32
"""
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on tuning split (last 20% of train.in)
E = 8          # embedding dimension per position
H = 64         # hidden layer size
SEED = 1
LR = 0.01
L2 = 1e-5
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
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters
    embed = (rng.normal(0, 0.01, (d * 256, E))).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E + H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / (H + C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    # Adam state
    params = [embed, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # Precompute position offsets for embedding lookup
    pos_off = np.arange(d, dtype=np.int32) * 256

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]                 # (B, d)
            yb = y[idx]                 # (B,)
            B = xb.shape[0]

            # Embedding lookup: (B, d, E)
            flat_idx = xb + pos_off     # (B, d)
            emb = embed[flat_idx.reshape(-1)].reshape(B, d * E)  # (B, d*E)

            # Forward
            h_pre = emb @ W1 + b1       # (B, H)
            h = np.maximum(h_pre, 0)    # ReLU
            logits = h @ W2 + b2        # (B, C)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B

            # Backprop
            grad_W2 = h.T @ probs + L2 * W2
            grad_b2 = probs.sum(axis=0)
            grad_h = probs @ W2.T
            grad_h[h_pre <= 0] = 0.0
            grad_W1 = emb.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)
            grad_emb = grad_h @ W1.T    # (B, d*E)

            # Accumulate embedding gradients
            grad_embed = np.zeros_like(embed)
            np.add.at(grad_embed, flat_idx.reshape(-1), grad_emb.reshape(-1, E))

            grads = [grad_embed, grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * g * g
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights
    magic = 0x454D424D  # 'EMBM'
    header = struct.pack("<5i", magic, d, E, H, C)
    blob = header
    for arr in params:
        blob += arr.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(blob)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte class index. */
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
    if (d != 16 || E != 8 || H != 64 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *embed = (const float *)(hd + 5);
    const float *W1 = embed + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_acc = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* Embedding accumulation */
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * E;
            float *dst = emb_acc + j * E;
            for (int ee = 0; ee < E; ee++) dst[ee] = e[ee];
        }

        /* Hidden layer: ReLU(emb_acc @ W1 + b1) */
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) sum += emb_acc[k] * W1[(long)k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        /* Output layer */
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        /* Argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```