```hypothesis
Family: label-mlp2 (two-hidden-layer MLP with ReLU). The 16-byte context benefits from modeling pairwise and higher-order interactions between positions, which a linear model cannot capture. Two hidden layers of 128 units each provide sufficient capacity while keeping inference fast via precomputed per-position embeddings. If dev accuracy does not exceed label-linear (0.3375) by at least 0.01, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on the tuning split (last 20% of train.in)
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
H1 = 128
H2 = 128

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def relu(x):
    return np.maximum(x, 0)

def softmax_cross_entropy_grad(logits, y):
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    # One-hot encoding: each position 256 categories
    Z = np.zeros((n, d * 256), dtype=np.float32)
    for j in range(d):
        Z[np.arange(n), j * 256 + X[:, j]] = 1.0

    rng = np.random.default_rng(SEED)
    k = d * 256
    # Xavier initialization
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
            z = Z[idx]
            yy = y[idx]

            # Forward
            a1 = z @ W1 + b1
            h1 = relu(a1)
            a2 = h1 @ W2 + b2
            h2 = relu(a2)
            logits = h2 @ W3 + b3

            # Loss gradient
            g3 = softmax_cross_entropy_grad(logits, yy)

            # Backward
            grad_W3 = h2.T @ g3 + L2 * W3
            grad_b3 = g3.sum(axis=0)
            g2 = g3 @ W3.T
            g2[a2 <= 0] = 0
            grad_W2 = h1.T @ g2 + L2 * W2
            grad_b2 = g2.sum(axis=0)
            g1 = g2 @ W2.T
            g1[a1 <= 0] = 0
            grad_W1 = z.T @ g1 + L2 * W1
            grad_b1 = g1.sum(axis=0)

            grads = [grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Convert W1 to per-position embeddings T1
    T1 = []
    for j in range(d):
        T1.append(W1[j * 256:(j + 1) * 256].copy())

    # Save weights
    magic = 0x4D4C5032  # "MLP2"
    header = struct.pack("<5i", magic, d, H1, H2, C)
    T1_arr = np.stack(T1, axis=0).astype(np.float32)
    out = header + T1_arr.tobytes() + b1.astype("<f4").tobytes() \
          + W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes() \
          + W3.astype("<f4").tobytes() + b3.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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
    if (nw < 20 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad header\n"); return 3; }
    long expected_floats = (long)d * 256 * H1 + H1 + (long)H1 * H2 + H2 + (long)H2 * C + C;
    if (nw != 20 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *ptr = (const float *)(hd + 5);
    const float *T1 = ptr; ptr += (long)d * 256 * H1;
    const float *b1 = ptr; ptr += H1;
    const float *W2 = ptr; ptr += (long)H1 * H2;
    const float *b2 = ptr; ptr += H2;
    const float *W3 = ptr; ptr += (long)H2 * C;
    const float *b3 = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h1 = malloc(sizeof(float) * H1);
    float *h2 = malloc(sizeof(float) * H2);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int u = 0; u < H1; u++) h1[u] = b1[u];
        for (int j = 0; j < d; j++) {
            const float *t = T1 + ((long)j * 256 + xi[j]) * H1;
            for (int u = 0; u < H1; u++) h1[u] += t[u];
        }
        for (int u = 0; u < H1; u++) if (h1[u] < 0.0f) h1[u] = 0.0f;

        for (int v = 0; v < H2; v++) h2[v] = b2[v];
        for (int u = 0; u < H1; u++) {
            float hu = h1[u];
            if (hu == 0.0f) continue;
            const float *w = W2 + (long)u * H2;
            for (int v = 0; v < H2; v++) h2[v] += hu * w[v];
        }
        for (int v = 0; v < H2; v++) if (h2[v] < 0.0f) h2[v] = 0.0f;

        for (int c = 0; c < C; c++) scores[c] = b3[c];
        for (int v = 0; v < H2; v++) {
            float hv = h2[v];
            if (hv == 0.0f) continue;
            const float *w = W3 + (long)v * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(h1); free(h2); free(scores);
    if (x) free(x);
    return 0;
}
```