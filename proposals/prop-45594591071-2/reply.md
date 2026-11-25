```hypothesis
Family: label-mlp (per-position byte embeddings + one ReLU hidden layer). 
Why: The linear model treats each position independently; a hidden layer can capture n-gram interactions. 
Embeddings (16 positions × 256 values × 16 dims) compress the one-hot input, reducing parameters vs. label-linear 
(266k vs 266k) but adding non-linearity. Hidden size 128 gives capacity to model character transitions. 
Failure: If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds ~10k Ir/item.
```

```python
"""label-mlp: per-position embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weights format (little-endian):
  int32 magic=0x4D4C5031, d=16, E=16, H=128, C=65
  float32 emb[d][256][E]       (16*256*16 = 65536 floats)
  float32 W1[d*E][H]           (256*128 = 32768 floats)
  float32 b1[H]                (128 floats)
  float32 W2[H][C]             (128*65 = 8320 floats)
  float32 b2[C]                (65 floats)
"""
import json, math, struct, sys

import numpy as np

E = 16          # embedding dimension per position
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
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 16 and C == 65

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Embeddings: [d, 256, E]
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # Hidden layer: input size d*E -> H
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer: H -> C
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # [B, d]
            yb = y[idx]          # [B]
            B = xb.shape[0]

            # Forward
            # Embedding lookup: [B, d, E]
            e = emb[:, xb.T, :].transpose(1, 0, 2)   # [B, d, E]
            flat = e.reshape(B, d * E)               # [B, d*E]
            h_pre = flat @ W1 + b1                   # [B, H]
            h = np.maximum(h_pre, 0)                 # ReLU
            logits = h @ W2 + b2                     # [B, C]

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B                                 # [B, C]  dL/dlogits

            # Backprop
            # Output layer
            dW2 = h.T @ probs + L2 * W2                # [H, C]
            db2 = probs.sum(axis=0)                    # [C]

            # Hidden layer
            dh = probs @ W2.T                          # [B, H]
            dh[h_pre <= 0] = 0.0                       # ReLU gradient
            dW1 = flat.T @ dh + L2 * W1                # [d*E, H]
            db1 = dh.sum(axis=0)                       # [H]

            # Embeddings
            # dh_flat = dh.reshape(B, d, E)  # not needed explicitly
            # We need to accumulate gradient per (position, byte value)
            demb = np.zeros_like(emb)                  # [d, 256, E]
            # For each position j, byte value v, add dh[b, j, :] to demb[j, v, :]
            # Vectorized: use scatter_add via advanced indexing
            for j in range(d):
                bytes_j = xb[:, j]                     # [B]
                # For each sample b, add dh[b, j, :] to demb[j, bytes_j[b], :]
                np.add.at(demb[j], bytes_j, dh[:, j, :])

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

    # Pack weights
    out = struct.pack("<5i", 0x4D4C5031, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte (class index 0..64). */
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
    if (nw < 20 || hd[0] != 0x4D4C5031) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 16 || H != 128 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 20L + 4L * (d * 256 * E + d * E * H + H + H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* Embedding lookup + concatenate */
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(flat + j * E, e, E * sizeof(float));
        }

        /* Hidden layer: ReLU(flat @ W1 + b1) */
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int f = 0; f < d * E; f++) sum += flat[f] * W1[(long)f * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        /* Output layer */
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        /* Argmax (first max wins) */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```