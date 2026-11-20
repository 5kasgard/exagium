```hypothesis
Family: label-hashngram (hashed n-gram linear model).
Why: Character-level language modeling benefits from local n-gram statistics (e.g., "th"→'e', "ing"→' ').
Label-linear only uses positional unigrams (which character at which absolute position), missing position-invariant patterns.
This model hashes all n-grams of length 1–3 into 2048 buckets and trains a linear classifier with Adam,
capturing local context while staying fast (sparse updates) and small (~0.5 MB weights).
Failure: dev accuracy ≤ 0.3375 (label-linear) or Ir/item > 10000 (3× label-linear's 7360).
```
```python
import json, math, struct, sys

import numpy as np

N = 3
B = 2048
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fnv1a_buckets(context, B, N):
    """Return (unique_buckets, counts) for all n-grams of length 1..N in context."""
    buckets = []
    L = len(context)
    for n in range(1, N + 1):
        for i in range(L - n + 1):
            h = 2166136261
            for b in context[i:i + n]:
                h ^= b
                h = (h * 16777619) & 0xFFFFFFFF
            buckets.append(h % B)
    if not buckets:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    arr = np.array(buckets, dtype=np.int32)
    uniq, cnt = np.unique(arr, return_counts=True)
    return uniq.astype(np.int32), cnt.astype(np.float32)


def main():
    card = json.load(open(sys.argv[1]))
    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(X_recs)
    X = np.array([list(r) for r in X_recs], dtype=np.uint8)
    y = np.array([r[0] for r in y_recs], dtype=np.int64)

    sample_buckets = []
    for i in range(n):
        uniq, cnt = fnv1a_buckets(X[i], B, N)
        sample_buckets.append((uniq, cnt))

    rng = np.random.default_rng(SEED)
    W = np.zeros((B, C), dtype=np.float32)
    b = np.zeros(C, dtype=np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            bs = len(idx)
            X_batch = np.zeros((bs, B), dtype=np.float32)
            for bi, i in enumerate(idx):
                uniq, cnt = sample_buckets[i]
                if uniq.size:
                    X_batch[bi, uniq] = cnt
            scores = X_batch @ W + b
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bs), y[idx]] -= 1.0
            grad = probs.astype(np.float32)
            gW = X_batch.T @ grad + L2 * W
            gb = grad.sum(axis=0)
            t += 1
            mW *= 0.9; mW += 0.1 * gW
            vW *= 0.999; vW += 0.001 * gW * gW
            W -= LR * (mW / (1 - 0.9 ** t)) / (np.sqrt(vW / (1 - 0.999 ** t)) + 1e-8)
            mb *= 0.9; mb += 0.1 * gb
            vb *= 0.999; vb += 0.001 * gb * gb
            b -= LR * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)

    magic = 0x484E4731
    out = struct.pack("<5i", magic, B, C, N, d)
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
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

static uint32_t fnv1a(const unsigned char *data, int len, uint32_t B) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= data[i];
        h *= 16777619u;
    }
    return h % B;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x484E4731) { fprintf(stderr, "bad weights\n"); return 3; }
    int B = hd[1], C = hd[2], Ngram = hd[3], d = hd[4];
    if (C < 1 || C > 256 || B < 1 || Ngram < 1 || d != 16) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 20L + 4L * ((long)B * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(hd + 5);
    const float *b = W + (long)B * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int n = 1; n <= Ngram; n++) {
            for (int start = 0; start <= d - n; start++) {
                uint32_t bucket = fnv1a(xi + start, n, B);
                const float *w = W + (long)bucket * C;
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