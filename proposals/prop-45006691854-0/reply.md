```hypothesis
Family: label-emb (embedding MLP). Adult's 14 features are all binned to 256 values; an embedding layer (8 dims per feature) compresses the sparse one-hot space into 112 dense inputs, then a 32-unit ReLU MLP captures pairwise interactions. This should exceed label-linear (0.8524) and approach label-mlp16 (0.8591) at lower inference cost (no large one-hot expansion). Failure: dev accuracy ≤ 0.8524 or instructions/item > 2× label-linear.
```
```python
"""label-emb: embedding + 1-hidden-layer MLP for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 42), single thread, stdlib + numpy.
Weights format: magic 0x454D424C, d, E, H, C; float32 emb[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hard-coded hyperparameters (chosen on tuning split offline)
E = 8          # embedding dimension per feature
H = 32         # hidden units
SEED = 42
LR = 1e-2
L2 = 1e-4
BATCH = 64
EPOCHS = 50

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
    rng = np.random.default_rng(SEED)

    # Embedding table: (d, 256, E)
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)

    # MLP weights
    in_dim = d * E
    W1 = (rng.normal(0, math.sqrt(2.0 / in_dim), (in_dim, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            # Embedding lookup
            x_emb = emb[xb]                    # (batch, d, E)
            x_flat = x_emb.reshape(len(idx), -1)  # (batch, d*E)

            # Forward
            h = x_flat @ W1 + b1
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = (probs / len(yb)).astype(np.float32)

            # Backprop
            grad_W2 = h_relu.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(0)
            grad_h = grad_logits @ W2.T
            grad_h[h <= 0] = 0.0
            grad_W1 = x_flat.T @ grad_h + L2 * W1
            grad_b1 = grad_h.sum(0)
            grad_x_flat = grad_h @ W1.T

            # Embedding gradients
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_x_flat[:, j*E:(j+1)*E])
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

    # Pack weights
    out = struct.pack("<5i", 0x454D424C, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: 1 byte class index. */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d <= 0 || E <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long in_dim = (long)d * E;
    long expected = 20
        + 4L * d * 256 * E
        + 4L * in_dim * H + 4L * H
        + 4L * H * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + in_dim * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* embedding lookup + concat */
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int h = 0; h < H; h++) {
                float sum = 0.0f;
                for (int e_i = 0; e_i < E; e_i++)
                    sum += e[e_i] * W1[((long)j * E + e_i) * H + h];
                hidden[h] += sum;
            }
        }

        /* ReLU */
        for (int h = 0; h < H; h++)
            if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        /* output layer */
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++)
                sum += hidden[h] * W2[(long)h * C + c];
            scores[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```