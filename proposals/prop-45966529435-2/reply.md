```hypothesis
We propose a position-wise embedding model (family "label-embed") that learns a distinct embedding vector for each byte at each of the 16 context positions, followed by a linear classifier. This captures position-specific character statistics and pairwise interactions via the linear layer, while being more parameter-efficient than the full one-hot linear model (label-linear) and faster than tree-based methods. We expect it to exceed label-linear's 33.75% accuracy on dev. If dev accuracy is ≤ 33.75%, the hypothesis fails.
```

```python
"""Position-wise embedding + linear classifier for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x454D4244, int32 d, C, ctx_len; float32 E[ctx_len][256][d], W[ctx_len*d][C], b[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (fixed, no tuning split used)
D = 16            # embedding dimension per position
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
MAGIC = 0x454D4244  # 'EMBD' in little-endian byte order

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    ctx_len = card["input"]["bytes"]
    C = card["classes"]
    # Load data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, ctx_len)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embeddings: (ctx_len, 256, D)
    scale = math.sqrt(6.0 / (256 + D))
    E = (rng.uniform(-1, 1, (ctx_len, 256, D)) * scale).astype(np.float32)
    # Linear layer: W (ctx_len*D, C), b (C,)
    W = (rng.uniform(-1, 1, (ctx_len * D, C)) * math.sqrt(6.0 / (ctx_len * D))).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    mE, vE = np.zeros_like(E), np.zeros_like(E)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (batch, ctx_len)
            yb = y[idx]          # (batch,)
            batch = xb.shape[0]

            # Embedding lookup: (ctx_len, batch, D) -> (batch, ctx_len, D) -> (batch, ctx_len*D)
            emb = E[np.arange(ctx_len), xb.T].transpose(1, 0, 2).reshape(batch, ctx_len * D)

            # Forward
            logits = emb @ W + b                     # (batch, C)
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(batch), yb] -= 1.0
            probs /= batch                           # gradient w.r.t logits

            # Gradients
            gW = emb.T @ probs + L2 * W              # (ctx_len*D, C)
            gb = probs.sum(axis=0)                   # (C,)
            g_emb = probs @ W.T                      # (batch, ctx_len*D)
            g_emb = g_emb.reshape(batch, ctx_len, D) # (batch, ctx_len, D)

            # Gradient for E: scatter-add per position
            gE = np.zeros_like(E)
            for j in range(ctx_len):
                np.add.at(gE[j], xb[:, j], g_emb[:, j, :])
            gE += L2 * E

            # Adam updates
            t += 1
            # E
            mE = BETA1 * mE + (1 - BETA1) * gE
            vE = BETA2 * vE + (1 - BETA2) * (gE * gE)
            mE_hat = mE / (1 - BETA1 ** t)
            vE_hat = vE / (1 - BETA2 ** t)
            E -= LR * mE_hat / (np.sqrt(vE_hat) + EPS)
            # W
            mW = BETA1 * mW + (1 - BETA1) * gW
            vW = BETA2 * vW + (1 - BETA2) * (gW * gW)
            mW_hat = mW / (1 - BETA1 ** t)
            vW_hat = vW / (1 - BETA2 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + EPS)
            # b
            mb = BETA1 * mb + (1 - BETA1) * gb
            vb = BETA2 * vb + (1 - BETA2) * (gb * gb)
            mb_hat = mb / (1 - BETA1 ** t)
            vb_hat = vb / (1 - BETA2 ** t)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + EPS)

    # Save weights
    # Flatten in C order: E (ctx_len, 256, D), W (ctx_len*D, C), b (C,)
    out = struct.pack("<Iiii", MAGIC, D, C, ctx_len)
    out += E.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Position-wise embedding + linear classifier for next-byte prediction.
 * ./infer weights.bin in.rec N
 * Each input record is 16 bytes; output is one byte (class index). */
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = (int)hd[1], C = (int)hd[2], ctx_len = (int)hd[3];
    if (ctx_len != 16 || C < 1 || C > 256 || d < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L
        + (long)ctx_len * 256 * d * 4L
        + (long)ctx_len * d * C * 4L
        + (long)C * 4L;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)ctx_len * 256 * d;
    const float *b = W + (long)ctx_len * d * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = (float *)malloc(sizeof(float) * ctx_len * d);
    float *logits = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)ctx_len) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, ctx_len); return 3; }

        // acc = sum of embeddings
        memset(acc, 0, sizeof(float) * ctx_len * d);
        for (int j = 0; j < ctx_len; j++) {
            const float *e = E + ((long)j * 256 + xi[j]) * d;
            for (int k = 0; k < d; k++) acc[j * d + k] += e[k];
        }

        // logits = acc @ W + b
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int idx = 0; idx < ctx_len * d; idx++) sum += acc[idx] * W[(long)idx * C + c];
            logits[c] = sum;
        }

        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(acc); free(logits);
    return 0;
}
```