```hypothesis
Propose "label-embed-mlp": learned embeddings per (position, byte) fed into a 2-layer MLP (ReLU hidden). This captures byte similarities and positional interactions more compactly than one-hot linear models, potentially exceeding label-linear's 0.3375 accuracy with similar or lower inference cost (dense matvecs vs. large sparse lookup). Failure if dev accuracy ≤ 0.34 or instructions/item > 10000.
```

```python
"""label-embed-mlp: embeddings + 2-layer MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
K = 32          # embedding size
H = 64          # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

MAGIC = 0x454D4245  # 'EMBE' little-endian


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
    # Load data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # Embeddings: (d, 256, K)
    E = rng.normal(0.0, 1.0 / math.sqrt(K), size=(d, 256, K)).astype(np.float32)
    # Layer 1: (K, H)
    W1 = rng.normal(0.0, math.sqrt(2.0 / (K + H)), size=(K, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Layer 2: (H, C)
    W2 = rng.normal(0.0, math.sqrt(2.0 / (H + C)), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [E, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X_bytes[idx]          # (B, d)
            yb = y[idx]                # (B,)
            B = len(yb)

            # Forward pass
            h = np.zeros((B, K), np.float32)
            for j in range(d):
                h += E[j, xb[:, j], :]
            h2 = np.maximum(h @ W1 + b1, 0.0, dtype=np.float32)
            logits = h2 @ W2 + b2

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)   # (B, C)

            # Backward pass
            grad_h2 = grad_logits @ W2.T                    # (B, H)
            grad_h2[h2 <= 0] = 0.0
            grad_W2 = h2.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_W1 = h.T @ grad_h2 + L2 * W1
            grad_b1 = grad_h2.sum(axis=0)
            grad_h = grad_h2 @ W1.T                         # (B, K)

            grad_E = np.zeros_like(E)
            for j in range(d):
                np.add.at(grad_E[j], xb[:, j], grad_h)
            grad_E += L2 * E

            grads = [grad_E, grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] *= 0.9; m[i] += 0.1 * g
                v[i] *= 0.999; v[i] += 0.001 * g * g
                m_hat = m[i] / (1.0 - 0.9 ** t)
                v_hat = v[i] / (1.0 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Layout: magic, d, K, H, C (int32), then E, W1, b1, W2, b2 (float32)
    header = struct.pack("<5i", MAGIC, d, K, H, C)
    body = (
        E.astype("<f4").tobytes() +
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
 * Input record: d bytes (d=16). Output record: 1 byte (class index). */
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
    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x454D4245) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], K = hd[2], H = hd[3], C = hd[4];
    if (d <= 0 || K <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * K + (long)K * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *E = p; p += (long)d * 256 * K;
    const float *W1 = p; p += (long)K * H;
    const float *b1 = p; p += H;
    const float *W2 = p; p += (long)H * C;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *h = (float *)malloc(sizeof(float) * K);
    float *h2 = (float *)malloc(sizeof(float) * H);
    float *logits = (float *)malloc(sizeof(float) * C);
    if (!h || !h2 || !logits) { fprintf(stderr, "oom\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // h = sum_j E[j, xi[j], :]
        memset(h, 0, sizeof(float) * K);
        for (int j = 0; j < d; j++) {
            const float *e = E + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) h[kk] += e[kk];
        }

        // h2 = ReLU(h * W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int kk = 0; kk < K; kk++) sum += h[kk] * W1[(long)kk * H + u];
            h2[u] = sum > 0.0f ? sum : 0.0f;
        }

        // logits = h2 * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h2[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```