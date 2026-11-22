```hypothesis
Family: label-embed-mlp (shared byte embeddings + 1-hidden-layer ReLU MLP).
Why: The linear model uses 16×256 one-hot features (4096 dims) ignoring character similarities. Shared 8-dim embeddings (256×8) let the model generalize across characters (e.g., 'a'≈'b'), and a 32-unit hidden layer captures n-gram interactions from the last 16 bytes. This adds only ~25k parameters and a few thousand FLOPs/item, so it should beat label-linear (0.3375) on accuracy while staying fast.
Failure: If dev accuracy ≤ 0.34 or inference cost > 2× label-linear, the family does not deliver.
```
```python
"""label-embed-mlp: shared byte embeddings + 1-hidden-layer MLP for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only. Exports weights for fast C inference."""
import json, math, struct, sys

import numpy as np

# Fixed architecture hyperparameters (chosen a priori for tinyshakespeare 16-byte context)
E = 8          # embedding dimension per byte
H = 32         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-5
BATCH = 128
SAMPLES = 600000
MAX_EPOCHS = 100

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
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    assert d == 16 and C == 65

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Parameters: emb[256, E], W1[d*E, H], b1[H], W2[H, C], b2[C]
    emb = (rng.normal(0, 0.1, (256, E))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # [B, 16] uint8
            yb = y[idx]          # [B] int64
            B = xb.shape[0]

            # Embedding lookup: [B, 16, E] -> [B, 16*E]
            x_emb = emb[xb].reshape(B, d * E)

            # Forward
            h_pre = x_emb @ W1 + b1
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)

            # Backprop
            grad_W2 = h.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_h = grad_logits @ W2.T
            grad_h[h_pre <= 0] = 0.0
            grad_W1 = x_emb.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(axis=0)

            # Embedding gradients: scatter-add
            grad_emb = np.zeros_like(emb)
            # grad_h: [B, H], W1: [d*E, H] -> grad_x_emb = grad_h @ W1.T  [B, d*E]
            grad_x_emb = grad_h @ W1.T
            grad_x_emb = grad_x_emb.reshape(B, d, E)
            # scatter-add into grad_emb
            for j in range(d):
                np.add.at(grad_emb, xb[:, j], grad_x_emb[:, j, :])
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            beta1, beta2 = 0.9, 0.999
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= beta1; m += (1 - beta1) * gr
                v *= beta2; v += (1 - beta2) * gr * gr
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights: magic, d, E, H, C, then flat arrays in order
    header = struct.pack("<5i", MAGIC, d, E, H, C)
    blob = (header +
            emb.astype("<f4").tobytes() +
            W1.astype("<f4").tobytes() +
            b1.astype("<f4").tobytes() +
            W2.astype("<f4").tobytes() +
            b2.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(blob)


if __name__ == "__main__":
    main()
```
```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format (little-endian):
 *   int32 magic(0x454D4244), d(16), E(8), H(32), C(65)
 *   float32 emb[256][E], W1[d*E][H], b1[H], W2[H][C], b2[C] */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 8 || H != 32 || C != 65) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected = 20L
        + 256L * E
        + (long)d * E * H
        + H
        + (long)H * C
        + C;
    expected *= 4;  // float32 bytes
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *emb = p; p += 256 * E;
    const float *W1 = p; p += d * E * H;
    const float *b1 = p; p += H;
    const float *W2 = p; p += H * C;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup + concat into h_pre = x_emb @ W1 + b1
        // We'll compute h_pre directly without materializing x_emb
        for (int u = 0; u < H; u++) h[u] = b1[u];
        for (int j = 0; j < d; j++) {
            const float *e = emb + xi[j] * E;
            const float *w = W1 + (j * E) * H;
            for (int u = 0; u < H; u++) {
                float sum = 0.0f;
                for (int k_ = 0; k_ < E; k_++) sum += e[k_] * w[k_ * H + u];
                h[u] += sum;
            }
        }
        // ReLU
        for (int u = 0; u < H; u++) if (h[u] < 0.0f) h[u] = 0.0f;

        // Output layer
        for (int c = 0; c < C; c++) logits[c] = b2[c];
        for (int u = 0; u < H; u++) {
            float hu = h[u];
            if (hu == 0.0f) continue;
            const float *w = W2 + u * C;
            for (int c = 0; c < C; c++) logits[c] += hu * w[c];
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```