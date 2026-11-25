```hypothesis
Family: label-embmlp — per-position embeddings (32-d) concatenated, one ReLU hidden layer (128 units), then 65-class output.
Why: The linear model uses 256-d one-hot per position (4096 features) with no non-linearity. Embeddings compress each byte to 32-d, preserving positional order via concatenation (512 features), then a hidden ReLU layer captures interactions. This yields ~205k parameters (vs linear's 266k) and should better model character dependencies. If dev accuracy does not exceed 0.3375 or inference cost rises significantly, the hypothesis fails.
```

```python
"""label-embmlp: per-position embeddings + one ReLU hidden layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x454D4231, d, E, H, C; then embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 32          # embedding dimension per position
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

    # Parameters
    # embeddings: (d, 256, E)
    scale_emb = 0.05
    emb = rng.uniform(-scale_emb, scale_emb, (d, 256, E)).astype(np.float32)

    D = d * E
    # W1: (D, H)  He uniform for ReLU
    lim1 = math.sqrt(6.0 / D)
    W1 = rng.uniform(-lim1, lim1, (D, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)

    # W2: (H, C)  Xavier uniform
    lim2 = math.sqrt(6.0 / (H + C))
    W2 = rng.uniform(-lim2, lim2, (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state: m, v for each param
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward: build concatenated embeddings Z (B, D)
            Z = np.empty((B, D), np.float32)
            for j in range(d):
                Z[:, j*E:(j+1)*E] = emb[j, Xb[:, j]]

            A = Z @ W1 + b1                 # (B, H)
            H_act = np.maximum(A, 0)        # ReLU
            logits = H_act @ W2 + b2        # (B, C)

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dlogits = probs / B             # (B, C)

            # Backprop
            dW2 = H_act.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)

            dH_act = dlogits @ W2.T         # (B, H)
            dA = dH_act * (A > 0)           # (B, H)

            dW1 = Z.T @ dA + L2 * W1
            db1 = dA.sum(axis=0)

            dZ = dA @ W1.T                  # (B, D)

            # Gradient for embeddings
            demb = np.zeros_like(emb)
            for j in range(d):
                # dZ[:, j*E:(j+1)*E] is (B, E)
                np.add.at(demb[j], Xb[:, j], dZ[:, j*E:(j+1)*E])

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

    # Write weights
    magic = 0x454D4231  # "EMB1"
    header = struct.pack("<5i", magic, d, E, H, C)
    out = header
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
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index.
 * Weights format: magic 0x454D4231, d, E, H, C; embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C] (float32 LE). */
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
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    int D = d * E;
    long expected = 20
        + (long)d * 256 * E * 4
        + (long)D * H * 4
        + (long)H * 4
        + (long)H * C * 4
        + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *Z = malloc(sizeof(float) * D);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(Z + j * E, e, sizeof(float) * E);
        }

        // Hidden layer: ReLU(Z @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int d_ = 0; d_ < D; d_++) sum += Z[d_] * W1[(long)d_ * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```