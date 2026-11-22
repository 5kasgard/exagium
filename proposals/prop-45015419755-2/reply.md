```hypothesis
An embedding-based MLP (EmbMLP) with position-specific byte embeddings (dim=4) and a single ReLU hidden layer (32 units) replaces the linear model's sparse one-hot encoding with dense distributed representations. This captures character similarities and positional interactions, improving accuracy over label-linear (0.3375) while keeping inference fast (~10k Ir/item) due to only ~20k parameters. Failure would be dev accuracy ≤ 0.3375 or instruction cost exceeding label-gbdt.
```

```python
"""Embedding MLP for next-byte prediction: position-specific embeddings + one ReLU layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
EMB_DIM = 4
HIDDEN = 32
SEED = 1
LR = 1e-3
L2 = 1e-4
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
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: (d, 256, EMB_DIM)
    scale_emb = math.sqrt(6.0 / (256 + EMB_DIM))
    emb = (rng.uniform(-1, 1, (d, 256, EMB_DIM)) * scale_emb).astype(np.float32)

    # Hidden layer: input size d*EMB_DIM -> HIDDEN
    k = d * EMB_DIM
    scale1 = math.sqrt(6.0 / (k + HIDDEN))
    W1 = (rng.uniform(-1, 1, (k, HIDDEN)) * scale1).astype(np.float32)
    b1 = np.zeros(HIDDEN, np.float32)

    # Output layer: HIDDEN -> C
    scale2 = math.sqrt(6.0 / (HIDDEN + C))
    W2 = (rng.uniform(-1, 1, (HIDDEN, C)) * scale2).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state: m, v for each param
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward
            # Embedding lookup: (B, d, EMB_DIM)
            emb_seq = emb[:, xb, :]  # (d, B, EMB_DIM) -> transpose
            emb_seq = np.transpose(emb_seq, (1, 0, 2))  # (B, d, EMB_DIM)
            flat = emb_seq.reshape(B, k)                # (B, k)

            hidden_pre = flat @ W1 + b1                 # (B, HIDDEN)
            hidden = np.maximum(hidden_pre, 0)          # ReLU
            logits = hidden @ W2 + b2                   # (B, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dlogits = probs / B                         # (B, C)

            # Backprop
            dW2 = hidden.T @ dlogits + L2 * W2          # (HIDDEN, C)
            db2 = dlogits.sum(axis=0)                   # (C,)

            dhidden = dlogits @ W2.T                    # (B, HIDDEN)
            dhidden[hidden_pre <= 0] = 0.0              # ReLU grad

            dW1 = flat.T @ dhidden + L2 * W1            # (k, HIDDEN)
            db1 = dhidden.sum(axis=0)                   # (HIDDEN,)

            dflat = dhidden @ W1.T                      # (B, k)
            demb_seq = dflat.reshape(B, d, EMB_DIM)     # (B, d, EMB_DIM)
            # Accumulate embedding gradients: for each position j, byte value b, add demb_seq[i, j, :]
            demb = np.zeros_like(emb)
            for j in range(d):
                bytes_j = xb[:, j]                      # (B,)
                # Use np.add.at for unbuffered accumulation
                np.add.at(demb[j], bytes_j, demb_seq[:, j, :])

            grads = [demb, dW1, db1, dW2, db2]

            # Adam update
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Pack weights: magic, d, EMB_DIM, HIDDEN, C; then arrays in order
    magic = 0x454D4231  # 'EMB1' little-endian
    header = struct.pack("<5i", magic, d, EMB_DIM, HIDDEN, C)
    # Flatten all arrays in little-endian float32
    out = bytearray(header)
    for arr in params:
        out += arr.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Embedding MLP inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index (0..C-1).
 * Weights format: magic(0x454D4231), d, D, H, C (int32); then float32 arrays:
 *   emb[d][256][D], W1[d*D][H], b1[H], W2[H][C], b2[C] */
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
    if (nw < 20 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || D != 4 || H != 32 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * D + (long)d * D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * D;
    const float *b1 = W1 + (long)d * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *input_vec = malloc(sizeof(float) * d * D);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build input vector by concatenating embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            memcpy(input_vec + j * D, e, D * sizeof(float));
        }

        // Hidden layer: ReLU(input_vec @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * D; k++) sum += input_vec[k] * W1[(long)k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(input_vec); free(hidden); free(logits);
    if (x) free(x);
    return 0;
}
```