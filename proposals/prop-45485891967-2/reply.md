```hypothesis
Family: label-hashlin (feature hashing + linear). Why: The task predicts next character from 16-byte context; local n-gram interactions (unigrams, bigrams, trigrams) are highly predictive in text. Hashing these n-grams into a fixed 2048-dim binary feature vector captures interactions without exploding parameters, while staying faster than tree/ensemble methods. A linear model on these hashed features should exceed label-linear's 0.3375 accuracy. Failure: dev accuracy ≤ 0.3375 or inference cost per item ≥ label-linear's 7360 Ir/item.
```
```python
"""label-hashlin: feature hashing of unigrams/bigrams/trigrams + linear classifier.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters
D = 2048            # hash dimension
HASH_SEED = 12345
NGRAM_ORDERS = (1, 2, 3)   # unigram, bigram, trigram
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def hash_ngram(bytes_arr, seed=HASH_SEED):
    h = seed
    for b in bytes_arr:
        h = (h * 31 + b) & 0xFFFFFFFF
    return h % D


def featurize(X, d):
    """X: (n, d) uint8 -> Z: (n, D) float32 binary features."""
    n = X.shape[0]
    Z = np.zeros((n, D), dtype=np.float32)
    for i in range(n):
        row = X[i]
        for pos in range(d):
            # unigram
            if 1 in NGRAM_ORDERS:
                idx = hash_ngram(bytes([1, pos, row[pos]]))
                Z[i, idx] = 1.0
            # bigram
            if 2 in NGRAM_ORDERS and pos > 0:
                idx = hash_ngram(bytes([2, pos - 1, row[pos - 1], row[pos]]))
                Z[i, idx] = 1.0
            # trigram
            if 3 in NGRAM_ORDERS and pos > 1:
                idx = hash_ngram(bytes([3, pos - 2, row[pos - 2], row[pos - 1], row[pos]]))
                Z[i, idx] = 1.0
    return Z


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    Z = featurize(X, d).astype(np.float32)

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
            logits = z @ W + b
            g = softmax_grad(logits, yy)
            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Write weights.bin: magic, D, d, C, then W (D*C), then b (C)
    magic = 0x4841534C  # "HASH" little-endian
    out = struct.pack("<4i", magic, D, d, C)
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-hashlin infer: ./infer weights.bin in.rec N
 * Each input record is d bytes (d=16). Output: one byte class index per record. */
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

/* Hash function must match Python's hash_ngram exactly. */
static uint32_t hash_ngram(const unsigned char *bytes, int len, uint32_t D) {
    uint32_t h = 12345u;
    for (int i = 0; i < len; i++) {
        h = h * 31u + bytes[i];
    }
    return h % D;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4841534C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int D = hd[1], d = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16 + 4L * ((long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *W = (const float *)(hd + 4);
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = calloc(D, sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(feat, 0, D * sizeof(float));
        for (int p = 0; p < d; p++) {
            /* unigram */
            unsigned char ngram1[3] = {1, (unsigned char)p, xi[p]};
            feat[hash_ngram(ngram1, 3, D)] = 1.0f;
            /* bigram */
            if (p > 0) {
                unsigned char ngram2[4] = {2, (unsigned char)(p-1), xi[p-1], xi[p]};
                feat[hash_ngram(ngram2, 4, D)] = 1.0f;
            }
            /* trigram */
            if (p > 1) {
                unsigned char ngram3[5] = {3, (unsigned char)(p-2), xi[p-2], xi[p-1], xi[p]};
                feat[hash_ngram(ngram3, 5, D)] = 1.0f;
            }
        }

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < D; j++) {
            if (feat[j] == 0.0f) continue;
            const float *wj = W + (long)j * C;
            for (int c = 0; c < C; c++) scores[c] += feat[j] * wj[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(feat); free(scores);
    return 0;
}
```