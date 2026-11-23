```hypothesis
Family: label-embmlp (embedding + one-hidden-layer MLP). Each of the 16 context positions gets a learned 16-dim embedding; concatenated embeddings feed a 32-unit ReLU layer then a 65-class linear layer. This captures positional character interactions non-linearly with far fewer parameters than one-hot linear (16*256*16 + 256*32 + 32*65 ≈ 65k vs 4096*65=266k) and should exceed label-linear's 33.8% accuracy. If dev accuracy ≤ 0.34 or latency > 2× label-linear, the hypothesis fails.
```

```python
"""label-embmlp: embedding + one-hidden-layer ReLU MLP for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Weights format: magic 0x454D424D, d=16, E=16, H=32, C=65;
float32 arrays: emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 16          # embedding dimension per position
H = 32          # hidden layer size
SEED = 1
LR = 0.01
BATCH = 64
EPOCHS = 30
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
L2 = 1e-5

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
    # Load data
    X_bytes = b"".join(records(sys.argv[2]))
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # embeddings: (d, 256, E)
    emb = rng.normal(0, 0.01, size=(d, 256, E)).astype(np.float32)
    # W1: (d*E, H), Xavier init
    W1 = rng.uniform(-1, 1, (d * E, H)).astype(np.float32) * math.sqrt(6.0 / (d * E + H))
    b1 = np.zeros(H, np.float32)
    # W2: (H, C), Xavier init
    W2 = rng.uniform(-1, 1, (H, C)).astype(np.float32) * math.sqrt(6.0 / (H + C))
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state: m, v for each param
    adam_m = [np.zeros_like(p) for p in params]
    adam_v = [np.zeros_like(p) for p in params]
    t = 0

    # Precompute indices for embedding lookup: for each position, we need to gather rows.
    # We'll do it per batch with advanced indexing.
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # ---- Forward ----
            # Embedding lookup: (B, d, E)
            emb_lookup = emb[np.arange(d)[None, :], xb]   # (B, d, E)
            # Concatenate: (B, d*E)
            concat = emb_lookup.reshape(B, d * E)
            # Hidden layer
            hidden_pre = concat @ W1 + b1                 # (B, H)
            hidden = np.maximum(hidden_pre, 0)            # ReLU
            # Output layer
            logits = hidden @ W2 + b2                     # (B, C)
            # Softmax + cross-entropy
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            # Gradient on logits
            dlogits = probs
            dlogits[np.arange(B), yb] -= 1.0
            dlogits /= B

            # ---- Backward ----
            # Output layer gradients
            dW2 = hidden.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)
            dhidden = dlogits @ W2.T
            # ReLU gradient
            dhidden[hidden_pre <= 0] = 0.0
            # Hidden layer gradients
            dW1 = concat.T @ dhidden + L2 * W1
            db1 = dhidden.sum(axis=0)
            dconcat = dhidden @ W1.T                      # (B, d*E)
            # Embedding gradients: reshape dconcat to (B, d, E)
            demb_concat = dconcat.reshape(B, d, E)
            demb = np.zeros_like(emb)
            # Scatter-add: for each position, accumulate gradients for the used byte indices.
            # We can use np.add.at for deterministic behavior (single-threaded).
            for pos in range(d):
                np.add.at(demb[pos], xb[:, pos], demb_concat[:, pos, :])

            grads = [demb, dW1, db1, dW2, db2]

            # ---- Adam update ----
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m, v = adam_m[i], adam_v[i]
                m *= BETA1; m += (1 - BETA1) * g
                v *= BETA2; v += (1 - BETA2) * g * g
                m_hat = m / (1 - BETA1 ** t)
                v_hat = v / (1 - BETA2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + EPS)

    # ---- Write weights.bin ----
    magic = 0x454D424D  # 'EMBM'
    header = struct.pack("<5i", magic, d, E, H, C)
    # Flatten arrays in the agreed order
    out_bytes = header
    out_bytes += emb.astype("<f4").tobytes()
    out_bytes += W1.astype("<f4").tobytes()
    out_bytes += b1.astype("<f4").tobytes()
    out_bytes += W2.astype("<f4").tobytes()
    out_bytes += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (context). Output record: 1 byte (class index 0..64).
 * Weights format (little-endian):
 *   uint32 magic=0x454D424D, int32 d=16, E=16, H=32, C=65
 *   float32 emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C] */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 16 || H != 32 || C != 65) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected_floats = (long)d * 256 * E + (long)d * E * H + H + (long)H * C + C;
    if (nw != 20 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float hidden[H];
    float output[C];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // hidden = b1
        for (int h = 0; h < H; h++) hidden[h] = b1[h];

        // Accumulate embedding contributions: for each position, add emb[pos][byte] * W1 slice
        for (int p = 0; p < d; p++) {
            unsigned char byte = xi[p];
            const float *e = emb + ((long)p * 256 + byte) * E;
            for (int ee = 0; ee < E; ee++) {
                float eval = e[ee];
                const float *w1_col = W1 + ((long)p * E + ee) * H;
                for (int h = 0; h < H; h++) hidden[h] += eval * w1_col[h];
            }
        }
        // ReLU
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        // Output layer
        for (int c = 0; c < C; c++) output[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            const float *w2_row = W2 + (long)h * C;
            for (int c = 0; c < C; c++) output[c] += hv * w2_row[c];
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (output[c] > output[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```