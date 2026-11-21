```hypothesis
The label-emb-mlp family learns a 64‑dim embedding for each of the 256 byte values, concatenates the 16 embeddings (1024‑dim), and passes them through a 128‑unit ReLU hidden layer before the 65‑class output. This captures positional interactions and byte similarities (e.g. letters cluster) that the one‑hot linear model misses, while the embedding lookup and two small matvecs keep inference fast. Expected dev accuracy >0.34 (beating label‑linear’s 0.3375) with CPU cost under 2× label‑linear. If accuracy ≤0.34 or instructions/item >15k, the hypothesis fails.
```

```python
"""label-emb-mlp: embedding + one ReLU hidden layer for 16‑byte categorical input.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 64          # embedding dimension
H = 128         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 10

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
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    emb = (rng.normal(0, 1/math.sqrt(E), (256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0/(d*E)), (d*E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0/H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            x_idx = X_bytes[idx]          # (B, 16)
            yy = y[idx]                   # (B,)

            # Forward
            x_emb = emb[x_idx]            # (B, 16, E)
            x_flat = x_emb.reshape(len(idx), -1)   # (B, 16*E)
            h_pre = x_flat @ W1 + b1      # (B, H)
            h = np.maximum(h_pre, 0)      # ReLU
            logits = h @ W2 + b2          # (B, C)

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            probs /= len(yy)              # (B, C)  dL/dlogits

            # Backprop
            dW2 = h.T @ probs + L2 * W2
            db2 = probs.sum(axis=0)
            dh = probs @ W2.T             # (B, H)
            dh[h_pre <= 0] = 0.0          # ReLU grad
            dW1 = x_flat.T @ dh + L2 * W1
            db1 = dh.sum(axis=0)
            dx_flat = dh @ W1.T           # (B, 16*E)
            dx_emb = dx_flat.reshape(len(idx), d, E)   # (B, 16, E)

            # Embedding gradient: scatter-add
            demb = np.zeros_like(emb)
            np.add.at(demb, x_idx.reshape(-1), dx_emb.reshape(-1, E))
            demb += L2 * emb

            grads = [demb, dW1, db1, dW2, db2]

            # Adam update
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights.bin
    # Magic: 'EMB1' = 0x31424D45? Use little-endian: 0x454D4231
    magic = 0x454D4231
    header = struct.pack("<5i", magic, d, E, H, C)
    # Order: emb (256*E), W1 (d*E*H), b1 (H), W2 (H*C), b2 (C)
    body = (emb.astype("<f4").tobytes() +
            W1.astype("<f4").tobytes() +
            b1.astype("<f4").tobytes() +
            W2.astype("<f4").tobytes() +
            b2.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint32_t *hd32 = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd32[0] != 0x454D4231) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = (int)hd32[1], E = (int)hd32[2], H = (int)hd32[3], C = (int)hd32[4];
    if (d != 16 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 20 + 4L*(256L*E + (long)d*E*H + H + (long)H*C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd32 + 5);
    const float *emb = p; p += 256L * E;
    const float *W1 = p; p += (long)d * E * H;
    const float *b1 = p; p += H;
    const float *W2 = p; p += (long)H * C;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* Embedding lookup and flatten */
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * E;
            memcpy(flat + j * E, e, E * sizeof(float));
        }

        /* Hidden layer: ReLU(flat * W1 + b1) */
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int f = 0; f < d * E; f++) sum += flat[f] * W1[(long)f * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        /* Output layer: hidden * W2 + b2 */
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        /* Argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```