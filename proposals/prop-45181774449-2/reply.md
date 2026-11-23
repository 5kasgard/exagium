```hypothesis
Family: label-shared-embedding-mlp
Shared embeddings capture character similarities across positions, reducing parameters vs. one-hot MLPs. A single hidden layer models position interactions. If it doesn't beat label-linear, shared embeddings are insufficient for this positional task.
```
```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
D = 32        # embedding dimension
H = 128       # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-5
BATCH = 128
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
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]  # 16
    n = X_bytes.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embedding: 256 x D
    E = (rng.normal(0, 0.01, (256, D))).astype(np.float32)
    # Position matrices: d x D x H
    W_pos = (rng.normal(0, math.sqrt(2.0 / D), (d, D, H))).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [E, W_pos, b1, W2, b2]
    shapes = [p.shape for p in params]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X_bytes[idx]          # (B, d)
            yb = y[idx]                # (B,)
            B = len(yb)

            # Forward
            # Embedding lookup: (B, d, D)
            emb = E[xb]                # (B, d, D)
            # Position-wise linear: (B, d, D) @ (d, D, H) -> (B, d, H)
            # We can do: for each position, multiply and sum
            h = np.zeros((B, H), dtype=np.float32)
            for pos in range(d):
                h += emb[:, pos, :] @ W_pos[pos]   # (B, D) @ (D, H) -> (B, H)
            h += b1
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2              # (B, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B                 # (B, C)

            # Backprop
            grad_W2 = h_relu.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_h_relu = grad_logits @ W2.T        # (B, H)
            grad_h = grad_h_relu * (h > 0)
            grad_b1 = grad_h.sum(axis=0)
            grad_W_pos = np.zeros_like(W_pos)
            grad_E = np.zeros_like(E)
            for pos in range(d):
                grad_W_pos[pos] = emb[:, pos, :].T @ grad_h + L2 * W_pos[pos]
                # Gradient wrt embedding for this position
                grad_emb_pos = grad_h @ W_pos[pos].T   # (B, D)
                # Accumulate into grad_E using scatter-add
                np.add.at(grad_E, xb[:, pos], grad_emb_pos)
            grad_E += L2 * E

            grads = [grad_E, grad_W_pos, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights file
    # Magic: 0x53454D4C ("SEML")
    # Header: d, D, H, C (int32)
    # Data: E (256*D), W_pos (d*D*H), b1 (H), W2 (H*C), b2 (C) all float32 little-endian
    out = struct.pack("<5i", 0x53454D4C, d, D, H, C)
    out += E.astype("<f4").tobytes()
    out += W_pos.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
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
    if (hd[0] != 0x53454D4C) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    size_t expected = 20
        + 256L * D * 4
        + (long)d * D * H * 4
        + H * 4
        + (long)H * C * 4
        + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *E = (const float *)(wb + 20);
    const float *W_pos = E + 256 * D;
    const float *b1 = W_pos + (long)d * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(H * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Initialize hidden with bias
        memcpy(h, b1, H * sizeof(float));

        // For each position: embedding lookup + matrix-vector multiply
        for (int p = 0; p < d; p++) {
            const float *emb = E + xi[p] * D;
            const float *Wp = W_pos + (long)p * D * H;
            // h += emb @ Wp
            for (int u = 0; u < H; u++) {
                float sum = 0.0f;
                for (int v = 0; v < D; v++) {
                    sum += emb[v] * Wp[v * H + u];
                }
                h[u] += sum;
            }
        }

        // ReLU
        for (int u = 0; u < H; u++) if (h[u] < 0.0f) h[u] = 0.0f;

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[u * C + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(h); free(logits);
    return 0;
}
```