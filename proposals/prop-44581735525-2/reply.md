```hypothesis
We propose a two-hidden-layer MLP (label-mlp2) with ReLU activations. The first hidden layer (64 units) captures pairwise interactions between context bytes, and the second (32 units) refines features for the 65-class prediction. This deeper non-linearity should surpass the linear model (label-linear) and the shallow MLP (label-mlp16) on this character-level language task, where context interactions matter. If dev accuracy does not exceed 0.34, the architecture is insufficient for this dataset.
```

```python
"""Two-hidden-layer MLP for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weight format: magic 0x4D4C5032, d, H1, H2, C; float32 T[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C]."""
import json, math, struct, sys

import numpy as np

H1, H2 = 64, 32
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-4, 64, 600000, 30


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

    # Factored first layer: per-position lookup tables T1[d][256][H1]
    T1 = (rng.uniform(-1, 1, (d, 256, H1)) * math.sqrt(6.0 / (256 * d))).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
    b3 = np.zeros(C, np.float32)

    params = [T1, b1, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            m = len(yb)

            # Forward
            a1 = np.tile(b1, (m, 1))
            for j in range(d):
                a1 += T1[j, xb[:, j]]
            h1 = np.maximum(a1, 0)
            a2 = h1 @ W2 + b2
            h2 = np.maximum(a2, 0)
            a3 = h2 @ W3 + b3

            # Softmax gradient
            a3 -= a3.max(axis=1, keepdims=True)
            exp_a3 = np.exp(a3)
            probs = exp_a3 / exp_a3.sum(axis=1, keepdims=True)
            probs[np.arange(m), yb] -= 1.0
            da3 = (probs / m).astype(np.float32)

            # Backward
            dW3 = h2.T @ da3 + L2 * W3
            db3 = da3.sum(axis=0)
            dh2 = da3 @ W3.T
            da2 = dh2 * (a2 > 0).astype(np.float32)
            dW2 = h1.T @ da2 + L2 * W2
            db2 = da2.sum(axis=0)
            dh1 = da2 @ W2.T
            da1 = dh1 * (a1 > 0).astype(np.float32)

            # Gradients for factored first layer
            db1 = da1.sum(axis=0)
            dT1 = np.zeros_like(T1)
            for j in range(d):
                for h in range(H1):
                    dT1[j, :, h] = np.bincount(xb[:, j], weights=da1[:, h], minlength=256)

            grads = [dT1, db1, dW2, db2, dW3, db3]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m_, v_ = state[i]
                m_ *= 0.9; m_ += 0.1 * gr
                v_ *= 0.999; v_ += 0.001 * gr * gr
                m_hat = m_ / (1 - 0.9 ** t)
                v_hat = v_ / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<5i", 0x4D4C5032, d, H1, H2, C)
    out += T1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    out += W3.astype("<f4").tobytes()
    out += b3.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Two-hidden-layer MLP (label-mlp2) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class. */
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
    if (nw < 20 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }

    size_t expected = 20
        + (size_t)d * 256 * H1 * sizeof(float)
        + (size_t)H1 * sizeof(float)
        + (size_t)H1 * H2 * sizeof(float)
        + (size_t)H2 * sizeof(float)
        + (size_t)H2 * C * sizeof(float)
        + (size_t)C * sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *T1 = p; p += (size_t)d * 256 * H1;
    const float *b1 = p; p += H1;
    const float *W2 = p; p += (size_t)H1 * H2;
    const float *b2 = p; p += H2;
    const float *W3 = p; p += (size_t)H2 * C;
    const float *b3 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h1 = malloc(sizeof(float) * H1);
    float *h2 = malloc(sizeof(float) * H2);
    float *sc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Layer 1: factored lookup + bias + ReLU
        memcpy(h1, b1, H1 * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *t = T1 + ((size_t)j * 256 + xi[j]) * H1;
            for (int u = 0; u < H1; u++) h1[u] += t[u];
        }
        for (int u = 0; u < H1; u++) if (h1[u] < 0.0f) h1[u] = 0.0f;

        // Layer 2: dense + bias + ReLU
        for (int v = 0; v < H2; v++) {
            float sum = b2[v];
            for (int u = 0; u < H1; u++) sum += h1[u] * W2[(size_t)u * H2 + v];
            h2[v] = sum > 0.0f ? sum : 0.0f;
        }

        // Layer 3: dense + bias
        for (int c = 0; c < C; c++) {
            float sum = b3[c];
            for (int v = 0; v < H2; v++) sum += h2[v] * W3[(size_t)v * C + c];
            sc[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```