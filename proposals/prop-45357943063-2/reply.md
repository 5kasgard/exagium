```hypothesis
We propose "label-hashbigram": a linear model extended with hashed bigram features.
The linear baseline (label-linear) uses only unigram features (each of the 16 bytes independently).
Adding bigram features captures local character interactions (digrams) which are crucial for
language modeling. Feature hashing (1024 buckets per bigram position) keeps the model size
and inference cost low: only 31 table lookups (16 unigrams + 15 bigrams) and 31*65 additions
per item. We expect this to beat label-linear's 33.75% dev accuracy. If it does not improve,
it would show that bigram interactions are not sufficiently captured by hashing or that the
unigram model already captures them via the 16-byte context.
```

```python
"""label-hashbigram: linear model with unigram and hashed bigram features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Weights format: magic 0x48424731, d, H, C, has_bias; then float32 tables:
unigram[d][256][C], bigram[d-1][H][C], bias[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
HASH_SIZE = 1024          # buckets per bigram position
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
HASH_MULT = 1234567       # fixed hash multiplier

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
    d = X.shape[1]          # 16
    C = card["classes"]     # 65
    H = HASH_SIZE
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Unigram table: (d * 256, C)
    unigram = rng.normal(0, 0.01, size=(d * 256, C)).astype(np.float32)
    # Bigram table: ((d-1) * H, C)
    bigram = rng.normal(0, 0.01, size=((d - 1) * H, C)).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)

    params = [unigram, bigram, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Compute logits
            logits = bias[None, :]  # (1, C)
            # Unigram contributions
            for j in range(d):
                indices = j * 256 + xb[:, j]   # (B,)
                logits += unigram[indices]
            # Bigram contributions
            for j in range(d - 1):
                h = (xb[:, j].astype(np.uint32) * 257 + xb[:, j + 1]) * HASH_MULT
                h = (h % H).astype(np.int64)
                indices = j * H + h
                logits += bigram[indices]

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dlogits = (probs / B).astype(np.float32)   # (B, C)

            # Gradients
            gunigram = np.zeros_like(unigram)
            gbigram = np.zeros_like(bigram)
            gbias = dlogits.sum(axis=0)

            for j in range(d):
                indices = j * 256 + xb[:, j]
                np.add.at(gunigram, indices, dlogits)
            for j in range(d - 1):
                h = (xb[:, j].astype(np.uint32) * 257 + xb[:, j + 1]) * HASH_MULT
                h = (h % H).astype(np.int64)
                indices = j * H + h
                np.add.at(gbigram, indices, dlogits)

            # L2 regularization
            gunigram += L2 * unigram
            gbigram += L2 * bigram
            gbias += L2 * bias

            grads = [gunigram, gbigram, gbias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Header: magic, d, H, C, has_bias (all int32)
    header = struct.pack("<5i", 0x48424731, d, H, C, 1)
    # Tables in float32 little-endian
    body = unigram.astype("<f4").tobytes() + bigram.astype("<f4").tobytes() + bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-hashbigram inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x48424731, d, H, C, has_bias (int32 each);
 * float32 unigram[d][256][C], bigram[d-1][H][C], bias[C]. */
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
    if (nw < 20 || hd[0] != 0x48424731) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], has_bias = hd[4];
    if (d != 16 || C != 65 || H <= 0 || has_bias != 1) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 20
        + 4L * d * 256 * C
        + 4L * (d - 1) * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *unigram = (const float *)(hd + 5);
    const float *bigram = unigram + (long)d * 256 * C;
    const float *bias = bigram + (long)(d - 1) * H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    const uint32_t HASH_MULT = 1234567u;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Start with bias
        memcpy(scores, bias, C * sizeof(float));

        // Unigram contributions
        for (int j = 0; j < d; j++) {
            const float *src = unigram + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += src[c];
        }
        // Bigram contributions
        for (int j = 0; j < d - 1; j++) {
            uint32_t h = (xi[j] * 257u + xi[j + 1]) * HASH_MULT;
            h %= H;
            const float *src = bigram + ((long)j * H + h) * C;
            for (int c = 0; c < C; c++) scores[c] += src[c];
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```