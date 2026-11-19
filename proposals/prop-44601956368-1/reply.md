```hypothesis
Family: label-mlp2 (two-hidden-layer MLP with 32 and 16 ReLU units).  
Why: The 8×8 optdigits images have spatial structure that a single hidden layer (label-mlp16, 0.9770) only partly captures. A second hidden layer adds expressive power with modest extra cost (≈2× label-mlp16 Ir/item). The first layer uses per-pixel lookup tables (like label-linear) for fast inference.  
Failure signal: Dev accuracy ≤ 0.96 (worse than label-linear) or Ir/item > 20000.
```

```python
"""label-mlp2: two-hidden-layer MLP (32, 16 ReLU) with per-pixel lookup tables.
train.py: deterministic, single-thread, stdlib+numpy. Exports weights.bin with magic 0x32504C4D.
Format: int32 magic, d, H1, H2, C; float32 T1[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C]."""
import json, math, struct, sys

import numpy as np

H1, H2 = 32, 16
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

    # Encoding: all ordinal -> z-score per pixel
    enc = []
    for j in range(d):
        v = X[:, j].astype(np.float64)
        mu, sd = v.mean(), v.std() or 1.0
        enc.append(((np.arange(256) - mu) / sd)[:, None])  # (256,1)

    # Concatenated encoded input (n, d)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape  # k == d

    # Initialize weights
    W1 = (rng.uniform(-1, 1, (k, H1)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
    b3 = np.zeros(C, np.float32)

    params = [W1, b1, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            m = len(yy)

            # Forward
            a1 = z @ W1 + b1
            h1 = np.maximum(a1, 0)
            a2 = h1 @ W2 + b2
            h2 = np.maximum(a2, 0)
            logits = h2 @ W3 + b3

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(m), yy] -= 1.0
            dlogits = (probs / m).astype(np.float32)

            # Backward
            dW3 = h2.T @ dlogits + L2 * W3
            db3 = dlogits.sum(axis=0)
            dh2 = dlogits @ W3.T
            da2 = dh2 * (a2 > 0).astype(np.float32)
            dW2 = h1.T @ da2 + L2 * W2
            db2 = da2.sum(axis=0)
            dh1 = da2 @ W2.T
            da1 = dh1 * (a1 > 0).astype(np.float32)
            dW1 = z.T @ da1 + L2 * W1
            db1 = da1.sum(axis=0)

            grads = [dW1, db1, dW2, db2, dW3, db3]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m_, v_ = state[i]
                m_ *= 0.9; m_ += 0.1 * gr
                v_ *= 0.999; v_ += 0.001 * gr * gr
                m_hat = m_ / (1 - 0.9 ** t)
                v_hat = v_ / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build per-pixel lookup tables for first layer: T1[j][value] = enc[j][value] * W1[j]
    T1 = np.empty((d, 256, H1), dtype=np.float32)
    for j in range(d):
        T1[j] = enc[j] @ W1[j:j+1]  # (256,1) @ (1,H1) -> (256,H1)

    # Pack weights.bin
    magic = 0x32504C4D  # 'MLP2' little-endian
    header = struct.pack("<5i", magic, d, H1, H2, C)
    body = (
        T1.astype("<f4").tobytes() +
        b1.astype("<f4").tobytes() +
        W2.astype("<f4").tobytes() +
        b2.astype("<f4").tobytes() +
        W3.astype("<f4").tobytes() +
        b3.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-mlp2 inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixels). Output record: 1 byte (class 0-9).
 * Weights format: int32 magic(0x32504C4D), d, H1, H2, C; float32 T1[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C]. */
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
    if (nw < 20 || hd[0] != 0x32504C4D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H1 + H1 + (long)H1 * H2 + H2 + (long)H2 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T1 = (const float *)(hd + 5);
    const float *b1 = T1 + (long)d * 256 * H1;
    const float *W2 = b1 + H1;
    const float *b2 = W2 + (long)H1 * H2;
    const float *W3 = b2 + H2;
    const float *b3 = W3 + (long)H2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc1 = malloc(sizeof(float) * H1);
    float *h1 = malloc(sizeof(float) * H1);
    float *acc2 = malloc(sizeof(float) * H2);
    float *h2 = malloc(sizeof(float) * H2);
    float *acc3 = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Layer 1: acc1 = b1 + sum_j T1[j][xi[j]]
        memcpy(acc1, b1, H1 * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *t = T1 + ((long)j * 256 + xi[j]) * H1;
            for (int u = 0; u < H1; u++) acc1[u] += t[u];
        }
        for (int u = 0; u < H1; u++) h1[u] = acc1[u] > 0.0f ? acc1[u] : 0.0f;

        // Layer 2: acc2 = b2 + h1 @ W2
        memcpy(acc2, b2, H2 * sizeof(float));
        for (int u = 0; u < H1; u++) {
            float hu = h1[u];
            if (hu == 0.0f) continue;
            const float *w = W2 + (long)u * H2;
            for (int v = 0; v < H2; v++) acc2[v] += hu * w[v];
        }
        for (int v = 0; v < H2; v++) h2[v] = acc2[v] > 0.0f ? acc2[v] : 0.0f;

        // Layer 3: acc3 = b3 + h2 @ W3
        memcpy(acc3, b3, C * sizeof(float));
        for (int v = 0; v < H2; v++) {
            float hv = h2[v];
            if (hv == 0.0f) continue;
            const float *w = W3 + (long)v * C;
            for (int c = 0; c < C; c++) acc3[c] += hv * w[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc3[c] > acc3[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```