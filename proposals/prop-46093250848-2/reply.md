```hypothesis
Family: label-embed-mlp. Learns a shared 16-dim embedding for each of the 256 byte values plus a 16-dim position embedding for each of the 16 context positions. The summed embeddings are concatenated (256-d) and fed to a 64-unit ReLU hidden layer, then a 65-class linear output. Sharing embeddings across positions captures character similarities (e.g., letters vs punctuation) and reduces parameters vs one-hot linear. If dev accuracy does not exceed label-linear (0.3375) or inference cost grows beyond ~2× label-linear, the hypothesis fails.
```

```python
"""label-embed-mlp: shared byte embeddings + position embeddings -> ReLU hidden -> linear.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
E = 16          # embedding dimension
H = 64          # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 20
SAMPLES = 600000  # not used directly; we use MAX_EPOCHS

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
    # byte_emb: (256, E), pos_emb: (d, E)
    scale = 1.0 / math.sqrt(E)
    byte_emb = rng.normal(0, scale, (256, E)).astype(np.float32)
    pos_emb  = rng.normal(0, scale, (d, E)).astype(np.float32)

    # Hidden layer: (d*E) -> H
    fan_in = d * E
    W1 = rng.uniform(-1, 1, (fan_in, H)).astype(np.float32) * math.sqrt(6.0 / fan_in)
    b1 = np.zeros(H, np.float32)
    # Output layer: H -> C
    W2 = rng.uniform(-1, 1, (H, C)).astype(np.float32) * math.sqrt(6.0 / H)
    b2 = np.zeros(C, np.float32)

    params = [byte_emb, pos_emb, W1, b1, W2, b2]
    # Adam state: (m, v) for each param
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            # embeddings: (B, d, E)
            emb = byte_emb[xb] + pos_emb  # broadcasting
            flat = emb.reshape(B, -1)     # (B, d*E)
            h = flat @ W1 + b1            # (B, H)
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2     # (B, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dlogits = (probs / B).astype(np.float32)   # (B, C)

            # Backward output layer
            dW2 = h_relu.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)
            dh_relu = dlogits @ W2.T

            # Backward hidden layer
            dh = dh_relu * (h > 0).astype(np.float32)
            dW1 = flat.T @ dh + L2 * W1
            db1 = dh.sum(axis=0)
            dflat = dh @ W1.T                     # (B, d*E)

            # Backward embeddings
            demb = dflat.reshape(B, d, E)         # (B, d, E)
            # byte_emb gradients: accumulate over batch and positions
            dbyte_emb = np.zeros_like(byte_emb)
            # Use flat indices for np.add.at
            np.add.at(dbyte_emb, xb.ravel(), demb.reshape(-1, E))
            # pos_emb gradients: sum over batch for each position
            dpos_emb = demb.sum(axis=0)           # (d, E)

            grads = [dbyte_emb, dpos_emb, dW1, db1, dW2, db2]

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Pack weights.bin
    # Header: magic, d, E, H, C
    magic = 0x454D424C  # 'EMBL'
    header = struct.pack("<5i", magic, d, E, H, C)
    # Arrays in order: byte_emb (256*E), pos_emb (d*E), W1 (d*E*H), b1 (H), W2 (H*C), b2 (C)
    body = (
        byte_emb.astype("<f4").tobytes() +
        pos_emb.astype("<f4").tobytes() +
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
/* label-embed-mlp inference: ./infer weights.bin in.rec N
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected = 20L + 4L * (256L * E + (long)d * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *byte_emb = (const float *)(hd + 5);
    const float *pos_emb  = byte_emb + 256L * E;
    const float *W1       = pos_emb  + (long)d * E;
    const float *b1       = W1       + (long)d * E * H;
    const float *W2       = b1       + H;
    const float *b2       = W2       + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *h    = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build concatenated embedding vector
        for (int j = 0; j < d; j++) {
            const float *be = byte_emb + (long)xi[j] * E;
            const float *pe = pos_emb  + (long)j * E;
            float *dst = flat + (long)j * E;
            for (int e = 0; e < E; e++) dst[e] = be[e] + pe[e];
        }

        // Hidden layer: flat (d*E) -> h (H) with ReLU
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            const float *w = W1 + (long)u;  // W1 is (d*E, H) stored row-major? In train.py: W1 shape (fan_in, H), so row-major: W1[flat_idx, u] at W1[flat_idx * H + u]. But we stored W1 as (d*E, H) in row-major, so element (i,u) at W1[i*H + u]. Here flat is (d*E). So we need to iterate over flat_idx.
            // Better: loop over flat_idx
            for (int idx = 0; idx < d * E; idx++) {
                sum += flat[idx] * W1[(long)idx * H + u];
            }
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: h (H) -> logits (C)
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) {
                sum += h[u] * W2[(long)u * C + c];
            }
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