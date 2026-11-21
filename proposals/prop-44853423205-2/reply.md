```hypothesis
We propose "label-embed-mlp": position-specific byte embeddings (dim 16) concatenated and fed through a single ReLU hidden layer (64 units) to 65 output classes. This reduces the 4096 one-hot dimensions to 256 dense features, capturing character similarities and position interactions with far fewer parameters (86k vs 266k) than the linear model. The hidden layer adds non-linear capacity while keeping inference fast (~10k Ir/item). If the embedding dimension is too small or the hidden layer too narrow, it may underperform the linear baseline.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen a priori; no tuning split search to keep training fast and deterministic)
E = 16          # embedding dimension per position
H = 64          # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

MAGIC = 0x454D4244  # "EMBD" in little-endian

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

    # Split off last 20% for validation (tuning split) – but we hardcode hyperparameters,
    # so we just train on all data. The split is kept for potential future use.
    split = int(0.8 * n)
    X_train, y_train = X_bytes[:split], y[:split]
    # X_val, y_val = X_bytes[split:], y[split:]  # unused

    # Initialize parameters
    # Embeddings: for each position (d), 256 vectors of size E
    emb = (rng.normal(0, 1/math.sqrt(256), (d, 256, E))).astype(np.float32)
    # W1: (d*E, H)
    W1 = (rng.uniform(-1, 1, (d*E, H)) * math.sqrt(6.0 / (d*E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: (H, C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state: (m, v) for each param
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X_bytes[idx]
            yb = y[idx]
            bs = Xb.shape[0]

            # Forward
            # Embedding lookup: for each position, gather embeddings
            # Shape: (bs, d, E)
            emb_lookup = emb[np.arange(d), Xb.T]  # (d, bs, E)
            emb_lookup = np.transpose(emb_lookup, (1, 0, 2))  # (bs, d, E)
            flat = emb_lookup.reshape(bs, d*E)  # (bs, d*E)

            h_pre = flat @ W1 + b1
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2

            # Softmax + cross-entropy gradient
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_logits = (probs / bs).astype(np.float32)  # (bs, C)

            # Backward
            grad_W2 = h.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_h = grad_logits @ W2.T
            grad_h[h_pre <= 0] = 0.0  # ReLU gradient
            grad_W1 = flat.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)
            # Gradient for embeddings: grad_h @ W1.T -> (bs, d*E) -> reshape to (bs, d, E)
            grad_flat = grad_h @ W1.T
            grad_emb_lookup = grad_flat.reshape(bs, d, E)
            # Now we need to accumulate gradients for each (position, byte) pair
            grad_emb = np.zeros_like(emb)
            # For each position j, we add grad_emb_lookup[:, j, :] to grad_emb[j, Xb[:, j], :]
            # Vectorized using np.add.at
            for j in range(d):
                np.add.at(grad_emb[j], Xb[:, j], grad_emb_lookup[:, j, :])
            grad_emb += L2 * emb  # L2 regularization on embeddings

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # After training, write weights file
    # Format: magic (int32), d (int32), H (int32), C (int32), E (int32)
    # Then embeddings: d * 256 * E float32 (row-major: position, byte, embedding)
    # Then W1: (d*E) * H float32
    # Then b1: H float32
    # Then W2: H * C float32
    # Then b2: C float32
    header = struct.pack("<5i", MAGIC, d, H, C, E)
    out = bytearray()
    out += header
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
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (d=16). Output record: 1 byte (class index 0..64).
 * Weights format: magic(0x454D4244), d, H, C, E (int32 each), then float32 arrays:
 *   emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C] (all row-major). */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], E = hd[4];
    if (d != 16 || C != 65 || E != 16 || H <= 0 || H > 1024) {
        fprintf(stderr, "unexpected dimensions\n"); return 3;
    }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }

        // Hidden layer: h = ReLU(concat * W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * E; v++) {
                sum += concat[v] * W1[(long)v * H + u];
            }
            hidden[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: logits = hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) {
                sum += hidden[u] * W2[(long)u * C + c];
            }
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (logits[c] > logits[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }

    free(concat); free(hidden); free(logits);
    return 0;
}
```