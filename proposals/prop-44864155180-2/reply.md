```hypothesis
We propose label-embmlp: an embedding-based MLP that learns a dense vector for each (position, byte) pair, concatenates them, and passes through one ReLU hidden layer. This captures positional interactions more efficiently than one-hot linear models, reducing parameters (≈45k vs 266k) and inference cost while improving accuracy. If dev accuracy does not exceed label-linear's 0.3375 or if instructions per item exceed label-linear's 7359.9, the hypothesis fails.
```

```python
"""label-embmlp: embedding + one hidden ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
SEED = 1
E = 8          # embedding dimension per (position, byte)
H = 64         # hidden layer size
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

    # Flatten embeddings: (d*256, E)
    emb = rng.normal(0, 0.01, size=(d * 256, E)).astype(np.float32)

    # MLP weights
    fan_in = d * E
    W1 = rng.normal(0, math.sqrt(2.0 / fan_in), size=(fan_in, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, math.sqrt(2.0 / H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state: (m, v) for each param
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Precompute position offsets for fast gathering
    pos_offset = np.arange(d, dtype=np.int32) * 256  # (d,)

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X[idx]                 # (b, d)
            yb = y[idx]                 # (b,)
            bsize = Xb.shape[0]

            # Embedding lookup: flat indices (b, d)
            flat_idx = Xb.astype(np.int32) + pos_offset  # (b, d)
            emb_batch = emb[flat_idx]                    # (b, d, E)
            x = emb_batch.reshape(bsize, -1)             # (b, d*E)

            # Forward
            hidden = x @ W1 + b1                         # (b, H)
            np.maximum(hidden, 0, out=hidden)            # ReLU in-place
            logits = hidden @ W2 + b2                    # (b, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsize), yb] -= 1.0
            dlogits = (probs / bsize).astype(np.float32)  # (b, C)

            # Backprop
            dW2 = hidden.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)

            dhidden = dlogits @ W2.T
            dhidden[hidden <= 0] = 0.0

            dW1 = x.T @ dhidden + L2 * W1
            db1 = dhidden.sum(axis=0)

            dx = dhidden @ W1.T                          # (b, d*E)
            dx = dx.reshape(bsize, d, E)                 # (b, d, E)

            # Embedding gradient via scatter-add
            demb = np.zeros_like(emb)
            np.add.at(demb, flat_idx, dx)
            demb += L2 * emb

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

    # Pack weights: magic, d, E, H, C; then emb, W1, b1, W2, b2 (all float32 little-endian)
    magic = 0x454D4244  # 'EMBD'
    header = struct.pack("<5i", magic, d, E, H, C)
    body = (
        emb.astype("<f4").tobytes() +
        W1.astype("<f4").tobytes() +
        b1.astype("<f4").tobytes() +
        W2.astype("<f4").tobytes() +
        b2.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)

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
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 8 || H != 64 || C != 65) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float concat[16 * 8];   // d*E = 128
    float hidden[64];       // H = 64
    float scores[65];       // C = 65

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, src, E * sizeof(float));
        }

        // Hidden layer: hidden = ReLU(concat @ W1 + b1)
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        for (int c_in = 0; c_in < d * E; c_in++) {
            float xv = concat[c_in];
            if (xv != 0.0f) {
                const float *w = W1 + (long)c_in * H;
                for (int h = 0; h < H; h++) hidden[h] += xv * w[h];
            }
        }
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        // Output layer: scores = hidden @ W2 + b2
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            if (hv != 0.0f) {
                const float *w = W2 + (long)h * C;
                for (int c = 0; c < C; c++) scores[c] += hv * w[c];
            }
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```