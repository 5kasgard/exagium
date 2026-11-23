```hypothesis
Family: label-hashngram (linear on hashed unigram+bigram features)
Why: The linear baseline uses only unigram features (each of 16 positions independently). Adding hashed bigram features (15 adjacent pairs) captures local character interactions (e.g., "th", "he") crucial for text, while hashing to a fixed dimension keeps the model small and inference fast (31 hash lookups + 31*C adds per item).
Failure: Dev accuracy ≤ 0.3375 (label-linear) or Ir/item > 2× label-linear.
```
```python
"""label-hashngram: linear model on hashed unigram+bigram features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Features: for each of 16 positions a unigram (type=1, pos, byte); for each of 15 adjacent pairs a bigram (type=2, pos, byte1, byte2).
Hashed to D=2048 with a simple deterministic hash. Linear logistic regression trained with Adam (same hyperparams as label-linear).
Weights format: magic 0x484E4731 ("HNG1"), D, H=0, C; float32 W1[D*C], b1[C]."""
import json, math, struct, sys

import numpy as np

D = 2048          # feature dimension (hashed)
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def hash_feat(*args):
    h = 0
    for a in args:
        h = (h * 31 + a) & 0xFFFFFFFF
    return h % D

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n, d = X.shape
    assert d == 16

    # Build dense feature matrix Z (n x D)
    Z = np.zeros((n, D), dtype=np.float32)
    for i in range(n):
        xi = X[i]
        # unigrams
        for pos in range(d):
            idx = hash_feat(1, pos, xi[pos])
            Z[i, idx] += 1.0
        # bigrams
        for pos in range(d - 1):
            idx = hash_feat(2, pos, xi[pos], xi[pos + 1])
            Z[i, idx] += 1.0

    rng = np.random.default_rng(SEED)
    W1 = np.zeros((D, C), np.float32)
    b1 = np.zeros(C, np.float32)
    params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # softmax gradient
            S = z @ W1 + b1
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Write weights
    magic = 0x484E4731  # "HNG1"
    H = 0
    out = struct.pack("<4i", magic, D, H, C)
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-hashngram inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x484E4731, int32 D, H=0, C; float32 W1[D*C], b1[C]. */
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

/* deterministic hash matching Python's hash_feat */
static uint32_t hash_feat(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    uint32_t h = 0;
    h = h * 31 + a;
    h = h * 31 + b;
    h = h * 31 + c;
    h = h * 31 + d;
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x484E4731) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int D = hd[1], H = hd[2], C = hd[3];
    if (H != 0 || C < 1 || C > 256 || D < 1) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + 4L * ((long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *W1 = (const float *)(hd + 4);
    const float *b1 = W1 + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        for (int c = 0; c < C; c++) sc[c] = b1[c];

        // unigrams
        for (int p = 0; p < 16; p++) {
            uint32_t idx = hash_feat(1, (uint32_t)p, xi[p], 0) % (uint32_t)D;
            const float *w = W1 + (long)idx * C;
            for (int c = 0; c < C; c++) sc[c] += w[c];
        }
        // bigrams
        for (int p = 0; p < 15; p++) {
            uint32_t idx = hash_feat(2, (uint32_t)p, xi[p], xi[p + 1]) % (uint32_t)D;
            const float *w = W1 + (long)idx * C;
            for (int c = 0; c < C; c++) sc[c] += w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```