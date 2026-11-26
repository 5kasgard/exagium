```hypothesis
A factorized embedding + one hidden layer MLP (label-embed-mlp) should capture per-position character statistics and their interactions better than a linear model on one-hot vectors, while keeping inference fast via precomputed hidden-layer contributions. If dev accuracy does not exceed label-linear's 0.3375, the family is not worthwhile.
```

```python
"""label-embed-mlp: factorized embeddings + one ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x4C424531, int32 d,E,H,C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 8          # embedding size per position
H = 64         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Parameters
    emb = (rng.normal(0, 0.1, (d, 256, E))).astype(np.float32)          # (d,256,E)
    W1 = (rng.normal(0, math.sqrt(2.0 / (E * d)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    # Precompute block slices for W1
    W1_blocks = [W1[j * E:(j + 1) * E] for j in range(d)]

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward
            # Embedding lookup: (B, d, E)
            embedded = np.stack([emb[j, xb[:, j]] for j in range(d)], axis=1).astype(np.float32)
            flat_emb = embedded.reshape(B, d * E)
            h_pre = flat_emb @ W1 + b1
            h = np.maximum(h_pre, 0, dtype=np.float32)
            logits = h @ W2 + b2

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B

            # Backward
            dW2 = h.T @ probs + L2 * W2
            db2 = probs.sum(axis=0)
            dh = probs @ W2.T
            dh_pre = dh * (h_pre > 0)
            dW1 = flat_emb.T @ dh_pre + L2 * W1
            db1 = dh_pre.sum(axis=0)
            demb_flat = dh_pre @ W1.T
            demb = demb_flat.reshape(B, d, E)

            # Embedding gradients
            demb_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(demb_emb[j], xb[:, j], demb[:, j])

            grads = [demb_emb, dW1, db1, dW2, db2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = BETA1 * m[i] + (1 - BETA1) * gr
                v[i] = BETA2 * v[i] + (1 - BETA2) * (gr * gr)
                m_hat = m[i] / (1 - BETA1 ** t)
                v_hat = v[i] / (1 - BETA2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + EPS)

    # Build inference tables: T[j][b] = emb[j,b] @ W1_block[j]  (size H)
    T = np.empty((d, 256, H), dtype=np.float32)
    for j in range(d):
        T[j] = emb[j] @ W1_blocks[j]

    # Write weights
    magic = 0x4C424531  # "LBE1"
    header = struct.pack("<5i", magic, d, E, H, C)
    body = T.astype("<f4").tobytes() + b1.astype("<f4").tobytes() + W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: magic 0x4C424531, int32 d,E,H,C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x4C424531) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(acc, b1, sizeof(float) * H);
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += t[h];
        }
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;

        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += acc[h] * W2[(long)h * C + c];
            scores[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```