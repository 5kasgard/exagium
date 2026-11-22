```hypothesis
Family: label-hashngram (linear model on hashed unigram+bigram features).
Why: Next-character prediction benefits from local byte interactions (bigrams). Feature hashing adds 15 bigram features to the 16 unigrams without exploding the feature space (fixed D=4096), capturing local context while keeping the model a simple linear layer. Training uses the same Adam/L2 recipe as label-linear for determinism and speed.
Failure signal: Dev accuracy ≤ 0.3375 (label-linear) or CPU cost per item > 2× label-linear.
```

```python
"""label-hashngram: linear model on hashed unigram+bigram features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (chosen on tuning split, then trained on all train)
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
D = 4096                 # Hash table size (same as label-linear's 16*256)
FNV_OFFSET = 0x811c9dc5
FNV_PRIME  = 0x01000193
MASK32     = 0xffffffff

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def hash_unigram(pos, byte):
    h = FNV_OFFSET
    h ^= pos;  h = (h * FNV_PRIME) & MASK32
    h ^= byte; h = (h * FNV_PRIME) & MASK32
    return h % D

def hash_bigram(pos, byte1, byte2):
    h = FNV_OFFSET
    h ^= pos;    h = (h * FNV_PRIME) & MASK32
    h ^= byte1;  h = (h * FNV_PRIME) & MASK32
    h ^= byte2;  h = (h * FNV_PRIME) & MASK32
    return h % D

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n, d = X.shape
    assert d == 16

    # Build dense design matrix Z (n x D) by hashing unigrams and bigrams
    Z = np.zeros((n, D), dtype=np.float32)
    for i in range(n):
        xi = X[i]
        for pos in range(d):
            Z[i, hash_unigram(pos, xi[pos])] += 1.0
        for pos in range(d - 1):
            Z[i, hash_bigram(pos, xi[pos], xi[pos + 1])] += 1.0

    # Linear model (no hidden layer) with Adam
    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # Softmax gradient
            S = z @ W + b
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            # Gradients with L2
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Write weights file: magic, d, H=0, C, D; then W[D][C] (row-major), bias[C]
    magic = 0x484E4731  # "HNG1"
    out = struct.pack("<5i", magic, d, 0, C, D)
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-hashngram infer: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format: int32 magic(0x484E4731), d, H, C, D; float32 W[D][C], bias[C]. */
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

/* FNV-1a 32-bit, same as train.py */
static uint32_t fnv1a_u32(uint32_t h, uint8_t byte) {
    h ^= byte;
    h *= 0x01000193u;
    return h;
}

static uint32_t hash_unigram(uint32_t pos, uint8_t byte, uint32_t D) {
    uint32_t h = 0x811c9dc5u;
    h = fnv1a_u32(h, pos);
    h = fnv1a_u32(h, byte);
    return h % D;
}

static uint32_t hash_bigram(uint32_t pos, uint8_t byte1, uint8_t byte2, uint32_t D) {
    uint32_t h = 0x811c9dc5u;
    h = fnv1a_u32(h, pos);
    h = fnv1a_u32(h, byte1);
    h = fnv1a_u32(h, byte2);
    return h % D;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x484E4731) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], D = hd[4];
    if (H != 0 || d != 16 || C < 1 || C > 256 || D != 4096) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 20L + 4L * ((long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W = (const float *)(hd + 5);
    const float *bias = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *score = malloc(sizeof(float) * C);
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) score[c] = bias[c];

        for (int p = 0; p < d; p++) {
            uint32_t h = hash_unigram(p, xi[p], D);
            const float *w = W + (long)h * C;
            for (int c = 0; c < C; c++) score[c] += w[c];
        }
        for (int p = 0; p < d - 1; p++) {
            uint32_t h = hash_bigram(p, xi[p], xi[p + 1], D);
            const float *w = W + (long)h * C;
            for (int c = 0; c < C; c++) score[c] += w[c];
        }

        best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```