```hypothesis
Family: label-emb-mlp (positional embeddings + one ReLU hidden layer). 
Why: tinyshakespeare has strong positional patterns; per-position embeddings capture n-gram statistics better than flat one-hot, and a hidden layer adds non-linear interactions. 
Failure sign: dev accuracy ≤ 0.3375 (label-linear) or inference cost > 2× label-linear.
```

```python
"""label-emb-mlp: positional embeddings + one ReLU hidden layer for next-char prediction.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format (little-endian):
  uint32 magic=0x454D424D ('EMBM'), int32 d, E, H, C
  float32 emb[d][256][E], W1[E][H], b1[H], W2[H][C], b2[C]"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then fixed)
E = 32          # embedding dimension per position
H = 128         # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


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

    # Embeddings: per position, per byte value -> E-dim vector
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # Hidden layer
    W1 = (rng.normal(0, math.sqrt(2.0 / E), (E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    steps_per_epoch = (n + BATCH - 1) // BATCH
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi, yy = X[idx], y[idx]
            # Embedding lookup and sum
            x_emb = np.zeros((len(xi), E), np.float32)
            for j in range(d):
                x_emb += emb[j, xi[:, j]]
            # Hidden layer
            h_pre = x_emb @ W1 + b1
            h = np.maximum(h_pre, 0)
            # Output
            logits = h @ W2 + b2
            g = softmax_grad(logits, yy)
            # Backprop
            gW2 = h.T @ g + L2 * W2
            gb2 = g.sum(0)
            gh = g @ W2.T
            gh[h_pre <= 0] = 0
            gW1 = x_emb.T @ gh + L2 * W1
            gb1 = gh.sum(0)
            # Embedding gradients
            gemb = np.zeros_like(emb)
            for j in range(d):
                # scatter-add gh into gemb[j, xi[:, j]]
                np.add.at(gemb[j], xi[:, j], gh)
            gemb += L2 * emb
            grads = [gemb, gW1, gb1, gW2, gb2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x454D424D
    header = struct.pack("<5i", magic, d, E, H, C)
    blob = header
    blob += emb.astype("<f4").tobytes()
    blob += W1.astype("<f4").tobytes()
    blob += b1.astype("<f4").tobytes()
    blob += W2.astype("<f4").tobytes()
    blob += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(blob)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index.
 * Weights format (little-endian):
 *   uint32 magic=0x454D424D, int32 d, E, H, C
 *   float emb[d][256][E], W1[E][H], b1[H], W2[H][C], b2[C] */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }

    long expected_floats = (long)d * 256 * E + (long)E * H + H + (long)H * C + C;
    if (nw != 20 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_emb = malloc(sizeof(float) * E);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding sum
        for (int e = 0; e < E; e++) x_emb[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) x_emb[e] += eptr[e];
        }

        // Hidden layer: h = ReLU(x_emb @ W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int e = 0; e < E; e++) sum += x_emb[e] * W1[(long)e * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }

        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(x_emb); free(h); free(logits);
    if (x) free(x);
    return 0;
}
```