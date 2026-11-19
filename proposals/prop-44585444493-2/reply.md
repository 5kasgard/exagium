```hypothesis
Family: label-emb-mlp (positional embeddings + one ReLU layer).
Why: The 16-byte context benefits from modeling positional interactions. A linear model (label-linear) treats each position independently via one-hot features (4096-dim). Using low-dimensional embeddings (D=8) per position captures similarity between characters and reduces parameters, while a hidden layer (H=64) adds non-linearity. This should exceed label-linear's 0.3375 accuracy with moderate CPU cost (expected ~10k Ir/item).
Failure signal: If dev accuracy ≤ 0.3375 or Ir/item > 20000, the family does not meet its goal.
```

```python
"""label-emb-mlp: positional embeddings (D=8) + one ReLU layer (H=64) for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib+numpy only.
Exports: magic 0x454D424C, d, H, C, D; float32 T[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then fixed)
D = 8          # embedding dimension per position
H = 64         # hidden layer size
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
    d = X.shape[1]          # 16
    C = card["classes"]     # 65
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: (d, 256, D)
    emb = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / (256 + D))).astype(np.float32)
    # W1: (d*D, H)
    W1 = (rng.uniform(-1, 1, (d * D, H)) * math.sqrt(6.0 / (d * D + H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: (H, C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / (H + C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Precompute indices for fast embedding lookup
    # For each example, we need to gather emb[j, X[i, j]] for j=0..d-1
    # We'll do this per batch.

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward: embeddings -> concat -> linear -> ReLU -> linear
            # Embedding lookup: (B, d, D)
            emb_seq = emb[np.arange(d), xb.T].transpose(1, 0, 2)  # (B, d, D)
            # Concat: (B, d*D)
            flat = emb_seq.reshape(B, d * D)
            # Hidden pre-activation
            h_pre = flat @ W1 + b1          # (B, H)
            h = np.maximum(h_pre, 0)        # ReLU
            # Logits
            logits = h @ W2 + b2            # (B, C)

            # Softmax + cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)   # (B, C)

            # Backprop
            # grad_W2 = h.T @ grad_logits + L2 * W2
            grad_W2 = h.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            # grad_h = grad_logits @ W2.T
            grad_h = grad_logits @ W2.T
            # ReLU mask
            grad_h_pre = grad_h * (h_pre > 0)
            # grad_W1 = flat.T @ grad_h_pre + L2 * W1
            grad_W1 = flat.T @ grad_h_pre + L2 * W1
            grad_b1 = grad_h_pre.sum(axis=0)
            # grad_flat = grad_h_pre @ W1.T  (B, d*D)
            grad_flat = grad_h_pre @ W1.T
            # grad_emb_seq = grad_flat.reshape(B, d, D)
            grad_emb_seq = grad_flat.reshape(B, d, D)
            # grad_emb: accumulate per (j, v)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                # xb[:, j] gives byte values for position j across batch
                np.add.at(grad_emb[j], xb[:, j], grad_emb_seq[:, j])

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # After training, precompute T[j][v] = emb[j][v] @ W1  (d, 256, H)
    T = np.zeros((d, 256, H), np.float32)
    for j in range(d):
        T[j] = emb[j] @ W1   # (256, D) @ (D, H) -> (256, H)

    # Write weights.bin
    # Header: magic, d, H, C, D (all int32 little-endian)
    header = struct.pack("<5i", 0x454D424C, d, H, C, D)
    # T: d * 256 * H floats
    # b1: H floats
    # W2: H * C floats
    # b2: C floats
    body = T.astype("<f4").tobytes() + b1.astype("<f4").tobytes() + W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x454D424C, d, H, C, D; float32 T[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], D = hd[4];
    if (d != 16 || C != 65 || H <= 0 || C > 256 || D <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected_floats = (long)d * 256 * H + H + (long)H * C + C;
    if (nw != 20 + 4L * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *sc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // acc = b1
        memcpy(acc, b1, H * sizeof(float));
        // add T[j][xi[j]] for each position j
        for (int j = 0; j < d; j++) {
            const float *tp = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += tp[h];
        }
        // ReLU
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;
        // output layer: sc = b2 + acc @ W2
        for (int c = 0; c < C; c++) sc[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            if (ah != 0.0f) {
                const float *w2h = W2 + (long)h * C;
                for (int c = 0; c < C; c++) sc[c] += ah * w2h[c];
            }
        }
        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(acc); free(sc);
    return 0;
}
```