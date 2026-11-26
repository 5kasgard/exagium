```hypothesis
Family: label-embmlp (positional embeddings + one ReLU hidden layer).  
Why: The linear model already uses position-specific one-hot features but cannot capture interactions between positions. Learned embeddings reduce dimensionality and allow the hidden layer to model pairwise interactions efficiently. A hidden size of 128 with embedding dim 32 should exceed label-linear's 0.3375 accuracy while keeping inference cost low (~80k multiply-adds vs linear's 266k).  
Failure sign: If dev accuracy ≤ 0.3375 or inference cost exceeds label-gbdt's 131k Ir/item.
```

```python
"""label-embmlp: positional embeddings + one ReLU hidden layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x454D424C, d, D, H, C; float32 embeddings[d][256][D], W1[(d*D)][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen a priori; no tuning split used)
D = 32          # embedding dimension per position
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: (d, 256, D)
    scale_emb = 1.0 / math.sqrt(256)
    emb = (rng.normal(0, scale_emb, (d, 256, D))).astype(np.float32)
    # W1: (d*D, H)
    scale_w1 = math.sqrt(6.0 / (d * D))
    W1 = (rng.uniform(-1, 1, (d * D, H)) * scale_w1).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: (H, C)
    scale_w2 = math.sqrt(6.0 / H)
    W2 = (rng.uniform(-1, 1, (H, C)) * scale_w2).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state: (m, v) for each param
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward
            # Embedding lookup: (B, d, D)
            emb_batch = emb[np.arange(d), xb.T].transpose(1, 0, 2)  # (B, d, D)
            flat = emb_batch.reshape(B, d * D)                      # (B, d*D)
            hidden_pre = flat @ W1 + b1                             # (B, H)
            hidden = np.maximum(hidden_pre, 0)                      # ReLU
            logits = hidden @ W2 + b2                               # (B, C)

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dlogits = probs / B                                     # (B, C)

            # Backward
            dW2 = hidden.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)
            dhidden = dlogits @ W2.T
            dhidden[hidden_pre <= 0] = 0.0
            dW1 = flat.T @ dhidden + L2 * W1
            db1 = dhidden.sum(axis=0)
            dflat = dhidden @ W1.T                                  # (B, d*D)
            demb_batch = dflat.reshape(B, d, D)                     # (B, d, D)

            # Gradient for embeddings: accumulate per position
            demb = np.zeros_like(emb)
            for j in range(d):
                # demb[j, xb[:, j], :] += demb_batch[:, j, :]
                np.add.at(demb[j], xb[:, j], demb_batch[:, j, :])

            grads = [demb, dW1, db1, dW2, db2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights.bin
    # magic, d, D, H, C
    header = struct.pack("<5i", 0x454D424C, d, D, H, C)
    # embeddings: d * 256 * D floats
    emb_bytes = emb.astype("<f4").tobytes()
    # W1: (d*D) * H
    W1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    # W2: H * C
    W2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(header + emb_bytes + W1_bytes + b1_bytes + W2_bytes + b2_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte (class index).
 * Weights format: magic 0x454D424C, int32 d, D, H, C;
 *   float32 emb[d][256][D], W1[(d*D)][H], b1[H], W2[H][C], b2[C] (all row-major). */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || D <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad header dims\n"); return 3; }

    long expected = 20
        + 4L * d * 256 * D
        + 4L * d * D * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + d * 256 * D;
    const float *b1 = W1 + d * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *hidden = (float *)malloc(sizeof(float) * H);
    float *output = (float *)malloc(sizeof(float) * C);
    float *concat = (float *)malloc(sizeof(float) * d * D);
    if (!hidden || !output || !concat) { fprintf(stderr, "alloc failed\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build concatenated embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            memcpy(concat + j * D, e, D * sizeof(float));
        }

        // Hidden layer: hidden = ReLU(concat @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int idx = 0; idx < d * D; idx++) {
                sum += concat[idx] * W1[(long)idx * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: output = hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[(long)h * C + c];
            }
            output[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (output[c] > output[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }

    free(hidden); free(output); free(concat);
    if (x) free(x);
    free(hd);
    return 0;
}
```