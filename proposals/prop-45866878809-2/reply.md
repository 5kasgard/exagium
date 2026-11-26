```hypothesis
Family: label-ngram (linear model on unigrams + hashed bigrams).  
Why: Character-level language modeling benefits from local n-gram interactions; a linear model on adjacent byte pairs captures this with minimal overhead.  
Failure: If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds ~10k Ir/item.
```

```python
"""label-ngram: linear model on unigram + hashed bigram features for 16-byte context.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x4D52474E, d=16, C=65, NBUCKETS=4096; float32 unigram[d][256][C], bigram[NBUCKETS][C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
NBUCKETS = 4096  # hash space for bigrams

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
    assert d == 16 and C == 65

    rng = np.random.default_rng(SEED)
    # Parameters
    unigram = (rng.uniform(-1, 1, (d, 256, C)) * math.sqrt(6.0 / (256 * C))).astype(np.float32)
    bigram  = (rng.uniform(-1, 1, (NBUCKETS, C)) * math.sqrt(6.0 / (NBUCKETS * C))).astype(np.float32)
    # Adam moments
    m_u = np.zeros_like(unigram); v_u = np.zeros_like(unigram)
    m_b = np.zeros_like(bigram);  v_b = np.zeros_like(bigram)

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    beta1, beta2 = 0.9, 0.999
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = X[idx]          # (B, 16)
            yy = y[idx]         # (B,)
            B = len(yy)

            # Forward: compute scores for each example
            scores = np.zeros((B, C), dtype=np.float32)
            for j in range(d):
                scores += unigram[j, z[:, j], :]
            for j in range(d - 1):
                bucket = (z[:, j].astype(np.uint32) * 256 + z[:, j + 1]) % NBUCKETS
                scores += bigram[bucket, :]

            # Softmax gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            grad = (probs / B).astype(np.float32)   # (B, C)

            # Accumulate gradients (dense for simplicity, size ~2*4096*65)
            gu = np.zeros_like(unigram)
            gb = np.zeros_like(bigram)
            for i in range(B):
                xi = z[i]
                g = grad[i]
                for j in range(d):
                    gu[j, xi[j], :] += g
                for j in range(d - 1):
                    bucket = (xi[j] * 256 + xi[j + 1]) % NBUCKETS
                    gb[bucket, :] += g

            # L2 regularization
            gu += L2 * unigram
            gb += L2 * bigram

            # Adam update
            t += 1
            m_u = beta1 * m_u + (1 - beta1) * gu
            v_u = beta2 * v_u + (1 - beta2) * gu * gu
            m_b = beta1 * m_b + (1 - beta1) * gb
            v_b = beta2 * v_b + (1 - beta2) * gb * gb
            m_u_hat = m_u / (1 - beta1 ** t)
            v_u_hat = v_u / (1 - beta2 ** t)
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)
            unigram -= LR * m_u_hat / (np.sqrt(v_u_hat) + 1e-8)
            bigram  -= LR * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)

    # Write weights
    magic = 0x4D52474E  # 'NGRM' little-endian
    out = struct.pack("<4i", magic, d, C, NBUCKETS)
    out += unigram.astype("<f4").tobytes()
    out += bigram.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-ngram inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index.
 * Weights: magic 0x4D52474E, d=16, C=65, NBUCKETS=4096;
 *          float32 unigram[d][256][C], bigram[NBUCKETS][C] (row-major). */
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
    if (nw < 16 || hd[0] != 0x4D52474E) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], NBUCKETS = hd[3];
    if (d != 16 || C != 65 || NBUCKETS != 4096) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C + (long)NBUCKETS * C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *unigram = (const float *)(hd + 4);
    const float *bigram  = unigram + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(scores, 0, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *u = unigram + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += u[c];
        }
        for (int j = 0; j < d - 1; j++) {
            uint32_t bucket = (xi[j] * 256 + xi[j + 1]) % NBUCKETS;
            const float *b = bigram + (long)bucket * C;
            for (int c = 0; c < C; c++) scores[c] += b[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```