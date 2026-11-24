```hypothesis
Family: label-embed-mlp (per-position byte embeddings summed, then one hidden ReLU layer).
Why: The linear model cannot capture interactions between positions. Embeddings provide a low-rank factorization of the first-layer weights, reducing parameters while allowing a hidden layer to model non-linear interactions. This should improve accuracy over linear on this character-prediction task where local context matters.
Failure indicator: Dev accuracy ≤ 0.3375 (label-linear) or inference cost > 2× label-linear.
```
```python
import json, math, struct, sys

import numpy as np

# Hyperparameters
D = 64          # embedding dimension
H = 128         # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

MAGIC = 0x454D424C  # "LMBE" in little-endian


def records(path):
    data = open(path, "rb").read()
    out = []
    i = 0
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

    # Initialize embeddings: list of (256, D) for each position
    embeddings = []
    for j in range(d):
        lim = math.sqrt(6.0 / (1 + D))
        E = rng.uniform(-lim, lim, (256, D)).astype(np.float32)
        embeddings.append(E)

    # Initialize W1, b1, W2, b2
    lim1 = math.sqrt(6.0 / (D + H))
    W1 = rng.uniform(-lim1, lim1, (D, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    lim2 = math.sqrt(6.0 / (H + C))
    W2 = rng.uniform(-lim2, lim2, (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    # Adam state: list of (m, v) for each parameter array
    state = []
    for E in embeddings:
        state.append((np.zeros_like(E), np.zeros_like(E)))
    state.append((np.zeros_like(W1), np.zeros_like(W1)))
    state.append((np.zeros_like(b1), np.zeros_like(b1)))
    state.append((np.zeros_like(W2), np.zeros_like(W2)))
    state.append((np.zeros_like(b2), np.zeros_like(b2)))

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward pass
            h = np.zeros((B, D), np.float32)
            for j in range(d):
                h += embeddings[j][xb[:, j]]
            a1 = h @ W1 + b1
            h1 = np.maximum(a1, 0)
            scores = h1 @ W2 + b2

            # Softmax cross-entropy gradient
            scores_max = scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            grad_scores = probs
            grad_scores[np.arange(B), yb] -= 1.0
            grad_scores /= B

            # Backprop
            grad_W2 = h1.T @ grad_scores + L2 * W2
            grad_b2 = grad_scores.sum(axis=0)
            grad_h1 = grad_scores @ W2.T
            grad_a1 = grad_h1 * (a1 > 0)
            grad_W1 = h.T @ grad_a1 + L2 * W1
            grad_b1 = grad_a1.sum(axis=0)
            grad_h = grad_a1 @ W1.T

            # Embedding gradients
            grad_embeddings = [np.zeros((256, D), np.float32) for _ in range(d)]
            for j in range(d):
                np.add.at(grad_embeddings[j], xb[:, j], grad_h)
                grad_embeddings[j] += L2 * embeddings[j]

            # Adam update
            t += 1
            def update_param(param, grad, state_idx):
                m, v = state[state_idx]
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

            for j in range(d):
                update_param(embeddings[j], grad_embeddings[j], j)
            update_param(W1, grad_W1, d)
            update_param(b1, grad_b1, d + 1)
            update_param(W2, grad_W2, d + 2)
            update_param(b2, grad_b2, d + 3)

    # Write weights.bin
    emb_flat = np.concatenate([E.ravel() for E in embeddings]).astype(np.float32)
    header = struct.pack("<5i", MAGIC, d, D, H, C)
    out = header
    out += emb_flat.tobytes()
    out += W1.astype(np.float32).tobytes()
    out += b1.astype(np.float32).tobytes()
    out += W2.astype(np.float32).tobytes()
    out += b2.astype(np.float32).tobytes()
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 20 + 4L * ((long)d * 256 * D + (long)D * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld vs %ld\n", nw, expected); return 3; }

    float *emb = (float *)(wb + 20);
    float *W1 = emb + (long)d * 256 * D;
    float *b1 = W1 + (long)D * H;
    float *W2 = b1 + H;
    float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * D);
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < D; j++) acc[j] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            for (int l = 0; l < D; l++) acc[l] += e[l];
        }
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int l = 0; l < D; l++) sum += acc[l] * W1[(long)l * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```