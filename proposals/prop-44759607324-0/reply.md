```hypothesis
Family: label-emb (per-feature embeddings + 1-hidden-layer MLP).
Why: The adult dataset has 14 features (mixed ordinal/categorical) each encoded as a byte. Linear models use one-hot for categoricals, which ignores category similarities. Embeddings learn dense representations per feature value, capturing non-linear interactions cheaply. A summed embedding vector (dim=16) followed by a ReLU layer (16 units) and a linear output should exceed label-linear's 0.8524 accuracy while keeping inference fast (only 14 embedding lookups, a 16-dim ReLU, and a 16x2 matmul).
Failure: If dev accuracy ≤ 0.8524 or inference cost (instructions/item) exceeds label-linear by >2×, the hypothesis is invalid.
```

```python
"""label-emb: per-feature embeddings summed, then one ReLU layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only.
Weights format (little-endian):
  magic(0x454D4244), d, emb_dim, H, C (int32 each)
  embeddings[d][256][emb_dim] (float32)
  W1[emb_dim][W], b1[W] (float32)  where W = H if H>0 else C
  if H>0: W2[H][C], b2[C] (float32)
"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
EMB_DIM = 16
H = 16
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

MAGIC = 0x454D4244  # "EMBD"

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
    d = X.shape[1]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: d x 256 x EMB_DIM
    scale = math.sqrt(6.0 / (256 + EMB_DIM))
    emb = (rng.uniform(-scale, scale, (d, 256, EMB_DIM))).astype(np.float32)

    # MLP weights
    W = H if H > 0 else C
    W1 = (rng.uniform(-1, 1, (EMB_DIM, W)) * math.sqrt(6.0 / EMB_DIM)).astype(np.float32)
    b1 = np.zeros(W, np.float32)
    if H > 0:
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        params = [emb, W1, b1]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward: embedding sum
            emb_sum = np.zeros((B, EMB_DIM), np.float32)
            for j in range(d):
                emb_sum += emb[j, xb[:, j]]   # (B, EMB_DIM)

            if H > 0:
                a = emb_sum @ W1 + b1         # (B, H)
                h = np.maximum(a, 0)
                logits = h @ W2 + b2          # (B, C)
            else:
                logits = emb_sum @ W1 + b1    # (B, C)

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)   # (B, C)

            # Backward
            grad_emb = np.zeros_like(emb)
            if H > 0:
                grad_W2 = h.T @ grad_logits + L2 * W2
                grad_b2 = grad_logits.sum(axis=0)
                grad_h = grad_logits @ W2.T
                grad_a = grad_h * (a > 0)
                grad_W1 = emb_sum.T @ grad_a + L2 * W1
                grad_b1 = grad_a.sum(axis=0)
                grad_emb_sum = grad_a @ W1.T
            else:
                grad_W1 = emb_sum.T @ grad_logits + L2 * W1
                grad_b1 = grad_logits.sum(axis=0)
                grad_emb_sum = grad_logits @ W1.T

            # Gradient for embeddings: accumulate per feature
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_emb_sum)

            grads = [grad_emb, grad_W1, grad_b1]
            if H > 0:
                grads += [grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<5i", MAGIC, d, EMB_DIM, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H > 0:
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index).
 * Weights format matches train.py. */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], emb_dim = hd[2], H = hd[3], C = hd[4];
    if (d <= 0 || emb_dim <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    int W = H > 0 ? H : C;
    long expected = 20L
        + 4L * d * 256 * emb_dim
        + 4L * emb_dim * W
        + 4L * W
        + (H > 0 ? (4L * H * C + 4L * C) : 0);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * emb_dim;
    const float *b1 = W1 + (long)emb_dim * W;
    const float *W2 = b1 + W;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * emb_dim);
    float *hidden = H > 0 ? malloc(sizeof(float) * H) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // embedding sum
        for (int e = 0; e < emb_dim; e++) emb_sum[e] = b1[e];
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * emb_dim;
            for (int e = 0; e < emb_dim; e++) emb_sum[e] += eptr[e];
        }

        const float *logits;
        if (H > 0) {
            // hidden layer: ReLU(emb_sum * W1 + b1) -> hidden
            for (int h = 0; h < H; h++) {
                float sum = 0.0f;
                for (int e = 0; e < emb_dim; e++) sum += emb_sum[e] * W1[(long)e * H + h];
                hidden[h] = sum > 0.0f ? sum : 0.0f;
            }
            // output layer
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
                scores[c] = sum;
            }
            logits = scores;
        } else {
            // linear: emb_sum already has b1 added, multiply by W1 (emb_dim x C)
            for (int c = 0; c < C; c++) {
                float sum = 0.0f;
                for (int e = 0; e < emb_dim; e++) sum += emb_sum[e] * W1[(long)e * C + c];
                scores[c] = sum;
            }
            logits = scores;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```