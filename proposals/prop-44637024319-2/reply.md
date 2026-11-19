```hypothesis
Family: label-embmlp (per-position embeddings + one ReLU layer).
Why: 16-byte context benefits from learned embeddings (256→E) per position, capturing character similarities better than one-hot. A hidden layer (H=256) adds non-linearity. Expected to beat label-linear (0.3375) on dev accuracy with moderate CPU cost (~2x label-linear).
Failure: dev accuracy ≤ 0.34 or instructions/item > 20000.
```
```python
"""label-embmlp: per-position embeddings + one ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Exports: magic 0x454D4244, d, E, H, C; embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

E = 64          # embedding dimension per position
H = 256         # hidden layer size
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x454D4244  # 'EMBD' little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Initialize parameters
    emb = (rng.normal(0, 0.02, (d, 256, E)).astype(np.float32))   # [d, 256, E]
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H)).astype(np.float32))
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C)).astype(np.float32))
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Precompute embedded matrix for all training data to avoid repeated indexing in inner loop
    # Memory: n * d * E * 4 bytes ≈ 24k * 16 * 64 * 4 ≈ 94 MB, well within 8 GiB.
    X_emb = np.empty((n, d * E), dtype=np.float32)
    for j in range(d):
        X_emb[:, j * E:(j + 1) * E] = emb[j, X[:, j]]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = X_emb[idx], y[idx]

            # Forward
            h = z @ W1 + b1
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_logits = (probs / len(yy)).astype(np.float32)

            # Backward
            grad_W2 = h_relu.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_h_relu = grad_logits @ W2.T
            grad_h = grad_h_relu * (h > 0)
            grad_W1 = z.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)
            grad_z = grad_h @ W1.T

            # Embedding gradients (scatter-add)
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], X[idx, j], grad_z[:, j * E:(j + 1) * E])
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    out = bytearray()
    out += struct.pack("<5i", MAGIC, d, E, H, C)
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
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format: magic(0x454D4244), d, E, H, C (int32 LE);
 *   embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C] (float32 LE). */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 64 || H != 256 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }

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
    float *hidden = malloc(H * sizeof(float));
    float *logits = malloc(C * sizeof(float));
    float *concat = malloc(d * E * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((size_t)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, src, E * sizeof(float));
        }

        // Hidden layer: ReLU(concat @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int w = 0; w < d * E; w++) sum += concat[w] * W1[(size_t)w * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden @ W2 + b2
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

    free(hidden); free(logits); free(concat);
    return 0;
}
```