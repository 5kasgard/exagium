```hypothesis
Family: label-embmlp — position-specific byte embeddings (D=8) + 64-unit ReLU hidden layer.
Why: The linear model uses 16×256×65 one-hot weights (266k params) but cannot model cross-position interactions. Learned embeddings share statistical strength across byte values and the hidden layer captures n-gram-like dependencies at ~45k params. Tinyshakespeare has strong local character correlations (e.g., "qu", "th", "ing") that embeddings+MLP should exploit.
Failure signal: dev accuracy ≤ 0.3375 (label-linear) or inference cost > 20k Ir/item.
```
```python
"""Embedding MLP for next-byte prediction: 16 position-specific byte embeddings (D=8) -> concat -> ReLU(64) -> 65 logits.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic: seed 1, single thread)
Exported format: int32 magic 0x454D424D, d, H, C, D; float32 E[d][256][D], W1[d*D][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

D, H = 8, 64
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    C = card["classes"]
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # Initialize embeddings: [d, 256, D]
    E = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / (256 + D))).astype(np.float32)
    # W1: [d*D, H], b1: [H]
    k = d * D
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: [H, C], b2: [C]
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [E, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi = X[idx]           # [B, d]
            yy = y[idx]           # [B]

            # Forward
            emb = E[np.arange(d), xi.T].transpose(1, 0, 2)   # [B, d, D]
            flat = emb.reshape(len(idx), -1)                 # [B, d*D]
            hidden_pre = flat @ W1 + b1                      # [B, H]
            hidden = np.maximum(hidden_pre, 0)               # ReLU
            logits = hidden @ W2 + b2                        # [B, C]

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_logits = (probs / len(yy)).astype(np.float32)   # [B, C]

            # Backprop
            grad_W2 = hidden.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ W2.T
            grad_hidden[hidden_pre <= 0] = 0.0
            grad_W1 = flat.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(axis=0)
            grad_flat = grad_hidden @ W1.T                     # [B, d*D]
            grad_emb = grad_flat.reshape(len(idx), d, D)       # [B, d, D]

            # grad_E: [d, 256, D]
            grad_E = np.zeros_like(E)
            for j in range(d):
                np.add.at(grad_E[j], xi[:, j], grad_emb[:, j, :])

            grads = [grad_E, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export
    magic = 0x454D424D  # 'EMBM'
    out = struct.pack("<5i", magic, d, H, C, D)
    out += E.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Embedding MLP inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format: int32 magic(0x454D424D), d, H, C, D;
 *   float32 E[d][256][D], W1[d*D][H], b1[H], W2[H][C], b2[C]. */
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
    int d = hd[1], H = hd[2], C = hd[3], D = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * D + (long)d * D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *E = (const float *)(hd + 5);
    const float *W1 = E + (long)d * 256 * D;
    const float *b1 = W1 + (long)d * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * D);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup + concat
        for (int j = 0; j < d; j++) {
            const float *e = E + ((long)j * 256 + xi[j]) * D;
            memcpy(flat + j * D, e, D * sizeof(float));
        }

        // Hidden layer: ReLU(flat @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * D; k++) sum += flat[k] * W1[(long)k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```