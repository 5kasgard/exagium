```hypothesis
Family: label-emb (embedding-based MLP). It learns a 16-dim embedding per position (16 positions × 256 byte values), concatenates them (256-d vector), passes through a 64-unit ReLU hidden layer, then a 65-class output layer. This captures byte similarities and position interactions with far fewer parameters than label-linear (≈86k vs 266k) and should be faster at inference (≈20k FLOPs/item). If dev accuracy does not exceed label-linear's 0.3375 or instruction count is not lower, the hypothesis fails.
```

```python
import json, struct, sys
import numpy as np

# Hyperparameters (chosen after tuning on the last 20% of train.in)
E = 16          # embedding dimension per position
H = 64          # hidden layer size
SEED = 1
LR = 1e-3
BATCH = 64
EPOCHS = 12
L2 = 1e-5

def records(path):
    data = open(path, "rb").read()
    out = []
    i = 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]   # 16
    C = card["classes"]          # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Embeddings: list of (256, E) for each position
    embeddings = [rng.normal(0, 0.01, size=(256, E)).astype(np.float32) for _ in range(d)]
    # Hidden layer
    W1 = rng.normal(0, np.sqrt(2.0/(d*E)), size=(d*E, H)).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    # Output layer
    W2 = rng.normal(0, np.sqrt(2.0/H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    # Adam state for each parameter array
    param_arrays = embeddings + [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in param_arrays]
    v = [np.zeros_like(p) for p in param_arrays]

    t = 0
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            x_batch = X[idx]          # (B, d)
            y_batch = y[idx]          # (B,)
            B = len(idx)

            # Forward: embedding lookup
            emb = np.empty((B, d, E), dtype=np.float32)
            for j in range(d):
                emb[:, j] = embeddings[j][x_batch[:, j]]
            flat = emb.reshape(B, d*E)                     # (B, d*E)
            hidden = flat @ W1 + b1                        # (B, H)
            hidden_relu = np.maximum(hidden, 0)            # ReLU
            logits = hidden_relu @ W2 + b2                 # (B, C)

            # Softmax cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(B), y_batch] -= 1.0
            grad_logits /= B

            # Backprop
            grad_W2 = hidden_relu.T @ grad_logits          # (H, C)
            grad_b2 = grad_logits.sum(axis=0)              # (C,)
            grad_hidden_relu = grad_logits @ W2.T          # (B, H)
            grad_hidden = grad_hidden_relu * (hidden > 0)  # ReLU grad

            grad_W1 = flat.T @ grad_hidden                 # (d*E, H)
            grad_b1 = grad_hidden.sum(axis=0)              # (H,)
            grad_flat = grad_hidden @ W1.T                 # (B, d*E)

            grad_emb = grad_flat.reshape(B, d, E)
            grad_embeddings = []
            for j in range(d):
                g = np.zeros((256, E), dtype=np.float32)
                np.add.at(g, x_batch[:, j], grad_emb[:, j])
                grad_embeddings.append(g)

            # L2 regularization
            for j in range(d):
                grad_embeddings[j] += L2 * embeddings[j]
            grad_W1 += L2 * W1
            grad_W2 += L2 * W2

            # Adam update
            t += 1
            all_grads = grad_embeddings + [grad_W1, grad_b1, grad_W2, grad_b2]
            for i, (p, gr) in enumerate(zip(param_arrays, all_grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9**t)
                v_hat = v[i] / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Header: magic (0x454D424C), d, E, H, C (all int32 little-endian)
    magic = 0x454D424C  # "LMBE" in ASCII (little-endian)
    out = struct.pack("<5i", magic, d, E, H, C)
    for j in range(d):
        out += embeddings[j].astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 16 || H != 64 || C != 65) { fprintf(stderr, "unexpected dimensions\n"); return 3; }

    size_t expected = 20
        + (size_t)d * 256 * E * 4
        + (size_t)d * E * H * 4
        + (size_t)H * 4
        + (size_t)H * C * 4
        + (size_t)C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (size_t)d * 256 * E;
    const float *b1 = W1 + (size_t)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (size_t)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * E;
            memcpy(flat + j * E, e, E * sizeof(float));
        }

        // Hidden layer: flat (d*E) x W1 (d*E x H) -> hidden (H)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int f = 0; f < d * E; f++) sum += flat[f] * W1[(size_t)f * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;  // ReLU
        }

        // Output layer: hidden (H) x W2 (H x C) -> logits (C)
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(size_t)h * C + c];
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