```hypothesis
Family: label-emb-mlp (shared byte embeddings + positional embeddings, one ReLU hidden layer).
Why: The linear model treats each position-byte independently; embeddings capture byte similarities (e.g., letters cluster) and the hidden layer models cross-position interactions. With H=128 the extra compute is modest (~10k ops/item) but should lift accuracy above 0.3375.
Failure: If dev accuracy ≤ 0.3375 or instructions/item exceeds 2× label-linear (~15k), the family does not justify its cost.
```

```python
"""label-emb-mlp: shared byte embeddings + positional embeddings, one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 64          # embedding dimension
H = 128         # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

MAGIC = 0x4D4C5032  # "MLP2" in little-endian


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
    assert d == 16 and C == 65

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embed: (256, E), pos_embed: (d, E), W1: (d*E, H), b1: (H,), W2: (H, C), b2: (C,)
    scale_embed = math.sqrt(6.0 / 256)
    embed = (rng.uniform(-1, 1, (256, E)) * scale_embed).astype(np.float32)
    pos_embed = np.zeros((d, E), dtype=np.float32)
    scale_W1 = math.sqrt(6.0 / (d * E))
    W1 = (rng.uniform(-1, 1, (d * E, H)) * scale_W1).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    scale_W2 = math.sqrt(6.0 / H)
    W2 = (rng.uniform(-1, 1, (H, C)) * scale_W2).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [embed, pos_embed, W1, b1, W2, b2]
    # Adam state: m, v for each param
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            e = embed[xb] + pos_embed[None, :, :]   # (B, d, E)
            flat = e.reshape(B, d * E)              # (B, d*E)
            h_pre = flat @ W1 + b1                  # (B, H)
            h = np.maximum(h_pre, 0)                # ReLU
            logits = h @ W2 + b2                    # (B, C)

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            # Backprop
            grad_W2 = h.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_h = grad_logits @ W2.T
            grad_h_pre = grad_h * (h_pre > 0)
            grad_W1 = flat.T @ grad_h_pre + L2 * W1
            grad_b1 = grad_h_pre.sum(axis=0)
            grad_flat = grad_h_pre @ W1.T           # (B, d*E)
            grad_e = grad_flat.reshape(B, d, E)     # (B, d, E)

            # Gradients for embeddings
            grad_embed = np.zeros_like(embed)
            for j in range(d):
                np.add.at(grad_embed, xb[:, j], grad_e[:, j, :])
            grad_embed += L2 * embed
            grad_pos_embed = grad_e.sum(axis=0) + L2 * pos_embed

            grads = [grad_embed, grad_pos_embed, grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Precompute T[j][b] = (embed[b] + pos_embed[j]) @ W1[j*E:(j+1)*E, :]
    T = np.zeros((d, 256, H), dtype=np.float32)
    for j in range(d):
        w1_slice = W1[j * E:(j + 1) * E, :]
        for b in range(256):
            T[j, b] = (embed[b] + pos_embed[j]) @ w1_slice

    # Pack weights.bin
    # Header: magic, d, H, C (4 int32)
    # Data: T (d*256*H floats), b1 (H), W2 (H*C), b2 (C) -- all float32 little-endian
    header = struct.pack("<4i", MAGIC, d, H, C)
    body = (
        T.astype("<f4").tobytes() +
        b1.astype("<f4").tobytes() +
        W2.astype("<f4").tobytes() +
        b2.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Each input record is 16 bytes; output is one byte (class index 0..64). */
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
    if (nw < 16 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (d != 16 || C != 65 || H <= 0 || H > 1024) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // acc = b1
        memcpy(acc, b1, H * sizeof(float));
        // Add T[j][xi[j]] for each position
        for (int j = 0; j < d; j++) {
            const float *tp = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += tp[h];
        }
        // ReLU
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;
        // Output layer: scores = b2 + acc @ W2
        memcpy(scores, b2, C * sizeof(float));
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            const float *w2h = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += ah * w2h[c];
        }
        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```