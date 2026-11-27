```hypothesis
Family: label-embmlp (embedding + MLP). Learns 8-dim embeddings per byte position (16×256×8 params),
concatenates to 128-dim dense vector, then 64-unit ReLU hidden layer to 65-class output (~45k params).
Captures character similarities and position interactions that linear one-hot cannot, at ~12k FLOPs/item.
Fails if dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds label-gbdt.
```
```python
"""label-embmlp: Embedding + MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, numpy only.
Architecture: 16 positions × 256 values × 8-dim embeddings -> concat (128) -> ReLU(64) -> Linear(65).
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n))."""
import json, math, struct, sys

import numpy as np

D, H = 8, 64
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x454D424D  # 'EMBM'


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Embeddings: [d, 256, D]
    E = (rng.normal(0, 0.01, (d, 256, D))).astype(np.float32)
    # MLP: 128 -> H -> C
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * D)), (d * D, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [E, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = len(yb)

            # Forward: embedding lookup -> concat
            z = np.concatenate([E[j, xb[:, j]] for j in range(d)], axis=1)  # [B, d*D]
            a1 = z @ W1 + b1
            h = np.maximum(a1, 0, dtype=np.float32)
            logits = h @ W2 + b2

            # Softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            g = (probs / bsz).astype(np.float32)  # [B, C]

            # Backprop
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(0)
            grad_h = g @ W2.T
            grad_a1 = grad_h * (a1 > 0)
            grad_W1 = z.T @ grad_a1 + L2 * W1
            grad_b1 = grad_a1.sum(0)
            grad_z = grad_a1 @ W1.T  # [B, d*D]

            # Embedding gradients
            grad_E = np.zeros_like(E)
            for j in range(d):
                sl = slice(j * D, (j + 1) * D)
                np.add.at(grad_E[j], xb[:, j], grad_z[:, sl])
            grad_E += L2 * E

            grads = [grad_E, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights
    total_floats = d * 256 * D + d * D * H + H + H * C + C
    out = struct.pack("<4i", MAGIC, d, D, H, C)  # wait, 5 ints but pack 4i? Fix below.
    # Actually need 5 ints: magic, d, D, H, C
    out = struct.pack("<5i", MAGIC, d, D, H, C)
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
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Embedding + MLP: 16×256×8 embeddings -> concat(128) -> ReLU(64) -> Linear(65).
 * Weight format: magic(0x454D424D), d, D, H, C (int32), then float32 arrays:
 *   E[d][256][D], W1[d*D][H], b1[H], W2[H][C], b2[C]. */
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
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || D != 8 || H != 64 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * D + (long)d * D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *E = (const float *)(hd + 5);
    const float *W1 = E + (long)d * 256 * D;
    const float *b1 = W1 + (long)d * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d * D);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup + concat
        for (int j = 0; j < d; j++) {
            const float *src = E + ((long)j * 256 + xi[j]) * D;
            float *dst = z + j * D;
            for (int u = 0; u < D; u++) dst[u] = src[u];
        }

        // Hidden layer: ReLU(z @ W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * D; v++) sum += z[v] * W1[(long)v * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer: h @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
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