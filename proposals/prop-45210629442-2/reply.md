```hypothesis
The label-embmlp family uses per-position character embeddings (size E=8) and a single ReLU hidden layer (H=64). This reduces the input dimensionality from 16×256 (one-hot) to 128, allowing the model to learn character similarities and context interactions. It should beat label-linear (0.3375) on dev accuracy while keeping inference fast (small dense multiplies). If dev accuracy ≤ 0.34, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameters (chosen on tuning split)
E = 8          # embedding dimension per position
H = 64         # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]  # 16

    n = X.shape[0]
    # Use all data for training (hyperparameters fixed after tuning)
    X_train, y_train = X, y

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: (d, 256, E)
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)
    # W1: (d*E, H)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: (H, C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(X_batch):
        # X_batch: (batch, d) uint8
        batch = X_batch.shape[0]
        # Vectorized embedding lookup: (d, batch, E) -> (batch, d*E)
        emb_out = emb[np.arange(d), X_batch.T].transpose(1, 0, 2).reshape(batch, d * E)
        h = emb_out @ W1 + b1
        h = np.maximum(h, 0, dtype=np.float32)  # ReLU
        logits = h @ W2 + b2
        return emb_out, h, logits

    def backward(emb_out, h, logits, y_batch, X_batch):
        batch = y_batch.shape[0]
        # Softmax cross-entropy gradient
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(batch), y_batch] -= 1.0
        grad_logits = probs / batch  # (batch, C)

        # Gradients for W2, b2
        grad_W2 = h.T @ grad_logits + L2 * W2
        grad_b2 = grad_logits.sum(0)

        # Gradient for h
        grad_h = grad_logits @ W2.T
        grad_h[h <= 0] = 0  # ReLU backward

        # Gradients for W1, b1
        grad_W1 = emb_out.T @ grad_h + L2 * W1
        grad_b1 = grad_h.sum(0)

        # Gradient for embeddings
        grad_emb = np.zeros_like(emb)
        # Reshape grad_h to (batch, d, E)
        grad_h_reshaped = grad_h.reshape(batch, d, E)
        # Accumulate using advanced indexing
        for j in range(d):
            np.add.at(grad_emb[j], X_batch[:, j], grad_h_reshaped[:, j])

        return [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

    t = 0
    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            X_batch, y_batch = X_train[idx], y_train[idx]
            emb_out, h, logits = forward(X_batch)
            grads = backward(emb_out, h, logits, y_batch, X_batch)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Export weights
    magic = 0x454D424D  # "EMBM"
    out = struct.pack("<5i", magic, d, E, H, C)
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected_floats = (long)d * 256 * E + (long)d * E * H + H + (long)H * C + C;
    long expected_bytes = 20 + expected_floats * 4;
    if (nw != expected_bytes) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_out = malloc(sizeof(float) * d * E);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int e_idx = 0; e_idx < E; e_idx++) {
                emb_out[j * E + e_idx] = e[e_idx];
            }
        }

        // Hidden layer: h = ReLU(emb_out * W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * E; v++) {
                sum += emb_out[v] * W1[v * H + u];
            }
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: logits = h * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) {
                sum += h[u] * W2[u * C + c];
            }
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (logits[c] > logits[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }

    free(emb_out);
    free(h);
    free(logits);
    return 0;
}
```