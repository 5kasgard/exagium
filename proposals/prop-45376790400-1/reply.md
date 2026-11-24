```hypothesis
Family: label-rff (Random Fourier Features). A fixed random projection (cosine) followed by a linear layer. The random projection approximates an RBF kernel, capturing non-linearities that a linear model misses. On optdigits, the spatial structure benefits from kernel methods. If dev accuracy ≤ 0.9613 (label-linear) or CPU cost exceeds label-linear by >2×, the family does not meet its goal.
```

```python
"""label-rff: Random Fourier Features + linear layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then fixed)
D = 32                  # number of random Fourier features
SEED = 1                # training seed
RFF_SEED = 12345        # random projection seed
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n, d = X_bytes.shape

    # Per-pixel mean/std for normalization (same as label-linear)
    mu = X_bytes.mean(axis=0).astype(np.float32)
    sd = X_bytes.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    X = (X_bytes.astype(np.float32) - mu) / sd

    # Fixed random projection: W_rand (d, D), b_rand (D)
    rng_rff = np.random.default_rng(RFF_SEED)
    W_rand = rng_rff.normal(0, 1, size=(d, D)).astype(np.float32)
    b_rand = rng_rff.uniform(0, 2 * math.pi, size=D).astype(np.float32)

    # Compute random Fourier features: cos(X @ W_rand + b_rand)
    Z = np.cos(X @ W_rand + b_rand)  # (n, D)

    # Linear model on features (same training as label-linear)
    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            g = softmax_grad(z @ W + b, yy)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Pack weights: magic, d, D, C, mu[d], sd[d], W_rand[d*D], b_rand[D], W[D*C], b[C]
    magic = 0x52464631  # 'RFF1'
    out = struct.pack("<4i", magic, d, D, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += W_rand.astype("<f4").tobytes()
    out += b_rand.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-rff inference: ./infer weights.bin in.rec N
 * Random Fourier Features (cosine) + linear layer. */
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x52464631) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 16L + 4L * (d + d + d * D + D + D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *mu = (const float *)(hd + 4);
    const float *sd = mu + d;
    const float *W_rand = sd + d;
    const float *b_rand = W_rand + (long)d * D;
    const float *W = b_rand + D;
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Normalize and compute random Fourier features
        for (int j = 0; j < D; j++) {
            float acc = b_rand[j];
            for (int p = 0; p < d; p++) {
                float xp = (xi[p] - mu[p]) / sd[p];
                acc += xp * W_rand[(long)p * D + j];
            }
            feat[j] = cosf(acc);
        }

        // Linear layer
        for (int c = 0; c < C; c++) {
            float sc = b[c];
            for (int j = 0; j < D; j++) sc += feat[j] * W[(long)j * C + c];
            scores[c] = sc;
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```