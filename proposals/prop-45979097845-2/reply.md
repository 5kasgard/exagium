```hypothesis
Family: label-hashngram (hashed n-gram linear model). It generates all n-grams up to length 3 from the 16-byte context, hashes them into a fixed-size feature table (F=65536), and learns a linear classifier on top. This captures local character dependencies (bigrams/trigrams) that are critical for language modeling, while the hashing trick keeps memory bounded and inference fast (only 45 hash lookups per item). It should beat label-linear (unigram only) on accuracy. If dev accuracy does not exceed 0.34, the n-gram features are not helping enough.
```

```python
"""Hashed n-gram linear model for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Features: all n-grams of length 1..3 from the 16-byte context, hashed to F=65536 buckets.
Linear model with Adam (lr=0.01, L2=1e-5), minibatch 64, 15 epochs.
Weights format: magic 0x484E4731, F, C, ngram_max; then float32 W[F, C], b[C]."""
import json, math, struct, sys

import numpy as np

F = 65536          # feature table size (power of two)
NGRAM_MAX = 3      # max n-gram length
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-5, 64, 15


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fnv1a_32(buf):
    """FNV-1a 32-bit hash, returns unsigned 32-bit int."""
    h = 0x811c9dc5
    for b in buf:
        h ^= b
        h = (h * 0x01000193) & 0xffffffff
    return h


def ngram_hashes(context):
    """Yield FNV-1a hashes of all n-grams (length 1..NGRAM_MAX) from context."""
    n = len(context)
    for L in range(1, NGRAM_MAX + 1):
        for i in range(n - L + 1):
            yield fnv1a_32(context[i:i + L]) & (F - 1)


def main():
    card = json.load(open(sys.argv[1]))
    X_raw = records(sys.argv[2])
    y_raw = records(sys.argv[3])
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 16

    X = np.frombuffer(b"".join(X_raw), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in y_raw], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    W = rng.normal(0, 0.01, (F, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    # Adam state
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            B = len(yb)

            # forward: accumulate scores for each example
            scores = np.zeros((B, C), dtype=np.float32)
            for i in range(B):
                ctx = Xb[i]
                for h in ngram_hashes(ctx):
                    scores[i] += W[h]
            scores += b

            # softmax cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B

            # gradients
            grad_W = np.zeros_like(W)
            grad_b = probs.sum(axis=0)
            for i in range(B):
                ctx = Xb[i]
                for h in ngram_hashes(ctx):
                    grad_W[h] += probs[i]
            grad_W += L2 * W

            # Adam update
            t += 1
            for param, grad, m, v in [(W, grad_W, mW, vW), (b, grad_b, mb, vb)]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # write weights
    magic = 0x484E4731  # "HNG1"
    out = struct.pack("<4i", magic, F, C, NGRAM_MAX)
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Hashed n-gram linear model: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x484E4731, F, C, ngram_max; float32 W[F*C], b[C]. */
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

/* FNV-1a 32-bit hash */
static uint32_t fnv1a_32(const unsigned char *buf, int len) {
    uint32_t h = 0x811c9dc5;
    for (int i = 0; i < len; i++) {
        h ^= buf[i];
        h *= 0x01000193;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x484E4731) { fprintf(stderr, "bad weights\n"); return 3; }
    int F = hd[1], C = hd[2], ngram_max = hd[3];
    if (C < 1 || C > 256 || F != 65536 || ngram_max != 3) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16 + 4L * F * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W = (const float *)(hd + 4);
    const float *b = W + (long)F * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        for (int c = 0; c < C; c++) scores[c] = b[c];

        /* all n-grams length 1..3 */
        for (int L = 1; L <= 3; L++) {
            for (int start = 0; start <= 16 - L; start++) {
                uint32_t h = fnv1a_32(xi + start, L) & (F - 1);
                const float *w = W + (long)h * C;
                for (int c = 0; c < C; c++) scores[c] += w[c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```