```hypothesis
Family: label-mlpemb (shared byte embeddings + one ReLU hidden layer).
Why: Character-level language modeling benefits from modeling interactions between context characters. A shared embedding layer (256×E) reduces parameters vs per-position one-hot, and a hidden layer (H units) captures non-linear interactions. This should improve accuracy over the linear baseline while keeping inference cost moderate.
Failure: Dev accuracy ≤ 0.3375 (label-linear) or instructions/item > 20000.
```
```python
"""Embedding + one hidden layer MLP for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x4D4C5045, d, E, H, C; float32 embed[256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 32          # embedding dimension
H = 128         # hidden layer size
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
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    embed = (rng.normal(0, 0.01, (256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [embed, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]                 # (B, d)
            yb = y[idx]                 # (B,)
            B = len(yb)

            # Forward
            x_emb = embed[xb].reshape(B, d * E)          # (B, d*E)
            h = x_emb @ W1 + b1                          # (B, H)
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2                    # (B, C)

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dlogits = (probs / B).astype(np.float32)     # (B, C)

            # Backward
            dW2 = h_relu.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)
            dh_relu = dlogits @ W2.T
            dh = dh_relu * (h > 0).astype(np.float32)
            dW1 = x_emb.T @ dh + L2 * W1
            db1 = dh.sum(axis=0)
            dx_emb = dh @ W1.T                           # (B, d*E)

            # Embedding gradients
            dembed = np.zeros_like(embed)
            dx_emb_3d = dx_emb.reshape(B, d, E)
            for j in range(d):
                np.add.at(dembed, xb[:, j], dx_emb_3d[:, j, :])

            grads = [dembed, dW1, db1, dW2, db2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x4D4C5045  # 'E','P','L','M' in little-endian
    header = struct.pack("<5i", magic, d, E, H, C)
    blob = header
    blob += embed.astype("<f4").tobytes()
    blob += W1.astype("<f4").tobytes()
    blob += b1.astype("<f4").tobytes()
    blob += W2.astype("<f4").tobytes()
    blob += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(blob)

if __name__ == "__main__":
    main()
```
```c
/* Embedding + one ReLU hidden layer MLP for next-character prediction.
 * ./infer weights.bin in.rec N
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x4D4C5045) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }

    size_t expected = 20
        + 256 * E * sizeof(float)
        + (size_t)d * E * H * sizeof(float)
        + H * sizeof(float)
        + (size_t)H * C * sizeof(float)
        + C * sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    float *embed = (float *)(wb + 20);
    float *W1 = embed + 256 * E;
    float *b1 = W1 + (size_t)d * E * H;
    float *W2 = b1 + H;
    float *b2 = W2 + (size_t)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * H);
    float *s = malloc(sizeof(float) * C);
    float *x_emb = malloc(sizeof(float) * d * E);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *row = embed + xi[j] * E;
            memcpy(x_emb + j * E, row, E * sizeof(float));
        }

        // Hidden layer
        for (int u = 0; u < H; u++) h[u] = b1[u];
        for (int w = 0; w < d * E; w++) {
            float xw = x_emb[w];
            if (xw == 0.0f) continue;
            const float *col = W1 + (size_t)w * H;
            for (int u = 0; u < H; u++) h[u] += xw * col[u];
        }
        for (int u = 0; u < H; u++) if (h[u] < 0.0f) h[u] = 0.0f;

        // Output layer
        for (int c = 0; c < C; c++) s[c] = b2[c];
        for (int u = 0; u < H; u++) {
            float hu = h[u];
            if (hu == 0.0f) continue;
            const float *row = W2 + (size_t)u * C;
            for (int c = 0; c < C; c++) s[c] += hu * row[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```