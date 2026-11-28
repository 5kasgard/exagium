```hypothesis
Family: label-embed-mlp. Per-position byte embeddings (E=8) concatenated into a 128-d vector, one ReLU hidden layer (H=64), then 65-class linear output. Embeddings capture positional character statistics; the hidden layer models interactions between positions. This should exceed label-linear's 0.3375 dev accuracy while keeping inference cost low (fewer multiply-adds than one-hot linear's 4096*65). Failure: dev accuracy ≤ 0.3475 or inference instructions/item ≥ 7359 (label-linear's cost).
```

```python
"""label-embed-mlp: per-position embeddings + 1 hidden ReLU layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split: last 20% of train.in)
E = 8          # embedding dimension per position
H = 64         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 20

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

    # Embedding table: (d, 256, E)
    emb = (rng.normal(0, 0.02, (d, 256, E))).astype(np.float32)

    # Hidden layer: (d*E, H)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)

    # Output layer: (H, C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state: (m, v) for each param
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward: embeddings -> concat -> hidden -> output
            # Embedding lookup: (B, d, E)
            emb_x = emb[np.arange(d)[None, :], xb]   # (B, d, E)
            h_in = emb_x.reshape(B, d * E)           # (B, d*E)

            # Hidden layer
            h_pre = h_in @ W1 + b1                   # (B, H)
            h = np.maximum(h_pre, 0)                 # ReLU

            # Output layer
            logits = h @ W2 + b2                     # (B, C)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_out = probs / B                     # (B, C)

            # Backprop
            grad_W2 = h.T @ grad_out + L2 * W2       # (H, C)
            grad_b2 = grad_out.sum(axis=0)           # (C,)

            grad_h = grad_out @ W2.T                 # (B, H)
            grad_h_pre = grad_h * (h_pre > 0)        # ReLU derivative

            grad_W1 = h_in.T @ grad_h_pre + L2 * W1  # (d*E, H)
            grad_b1 = grad_h_pre.sum(axis=0)         # (H,)

            # Gradient wrt embeddings: (B, d, E)
            grad_emb_concat = grad_h_pre @ W1.T      # (B, d*E)
            grad_emb = grad_emb_concat.reshape(B, d, E)

            # Accumulate embedding gradients per (position, byte)
            grad_emb_table = np.zeros_like(emb)
            np.add.at(grad_emb_table, (np.arange(d)[None, :], xb), grad_emb)
            grad_emb_table += L2 * emb

            grads = [grad_emb_table, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Header: magic, d, E, H, C
    magic = 0x454D424C  # 'LMBE' in little-endian
    header = struct.pack("<5i", magic, d, E, H, C)

    # Embedding table: (d, 256, E) -> flatten in order d, 256, E
    emb_bytes = emb.astype("<f4").tobytes()

    # Hidden weights: (d*E, H) row-major
    W1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()

    # Output weights: (H, C) row-major
    W2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()

    out = header + emb_bytes + W1_bytes + b1_bytes + W2_bytes + b2_bytes
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64). */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }

    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x454D424C) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }

    size_t emb_count = (size_t)d * 256 * E;
    size_t W1_count = (size_t)d * E * H;
    size_t b1_count = H;
    size_t W2_count = (size_t)H * C;
    size_t b2_count = C;
    size_t total_floats = emb_count + W1_count + b1_count + W2_count + b2_count;
    long expected = 20 + (long)total_floats * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(wb + 20);
    const float *W1 = emb + emb_count;
    const float *b1 = W1 + W1_count;
    const float *W2 = b1 + b1_count;
    const float *b2 = W2 + W2_count;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *h_in = malloc(sizeof(float) * d * E);
    float *h_pre = malloc(sizeof(float) * H);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * E;
            memcpy(h_in + j * E, e, E * sizeof(float));
        }

        // Hidden layer: h_pre = h_in * W1 + b1
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * E; v++) {
                sum += h_in[v] * W1[(size_t)v * H + u];
            }
            h_pre[u] = sum;
            h[u] = sum > 0.0f ? sum : 0.0f;  // ReLU
        }

        // Output layer: logits = h * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) {
                sum += h[u] * W2[(size_t)u * C + c];
            }
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;

        put_rec(stdout, &best, 1);
    }

    free(h_in); free(h_pre); free(h); free(logits);
    if (x) free(x);
    free(wb);
    return 0;
}
```