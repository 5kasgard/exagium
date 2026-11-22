```hypothesis
Family: label-ngramhash (linear model with hashed n-gram features up to trigrams).
Why: Character-level next-char prediction benefits from local n-gram context (e.g., "th", "ing"). Pure unigram linear model (label-linear) misses these. Hashing n-grams to a fixed dimension (8192) captures pairwise/triple interactions with minimal parameters and fast inference (45 hash computations per item). Expected to beat label-linear's 0.3375 accuracy at similar cost (≈10k Ir/item). Failure: if accuracy ≤0.3375, hashing collisions or insufficient capacity hurt more than n-grams help.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
D = 8192
MAGIC = 0x4E475248  # 'NGRH'

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def hash_ngram(pos, ngram_bytes, D):
    h = pos * 3 + len(ngram_bytes)
    for b in ngram_bytes:
        h = (h * 257 + b + 1) & 0xFFFFFFFF
    return h % D

def main():
    card = json.load(open(sys.argv[1]))
    train_in = sys.argv[2]
    train_key = sys.argv[3]
    weights_out = sys.argv[4]

    X_recs = records(train_in)
    y_recs = records(train_key)
    n = len(X_recs)
    d = card["input"]["bytes"]
    C = card["classes"]

    nfeat = 16 + 15 + 14  # unigram + bigram + trigram
    feat_idx = np.zeros((n, nfeat), dtype=np.int32)
    for i in range(n):
        x = X_recs[i]
        f = 0
        for pos in range(d):
            feat_idx[i, f] = hash_ngram(pos, bytes([x[pos]]), D)
            f += 1
        for pos in range(d - 1):
            feat_idx[i, f] = hash_ngram(pos, bytes([x[pos], x[pos+1]]), D)
            f += 1
        for pos in range(d - 2):
            feat_idx[i, f] = hash_ngram(pos, bytes([x[pos], x[pos+1], x[pos+2]]), D)
            f += 1

    y = np.array([r[0] for r in y_recs], dtype=np.int64)

    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            B = len(idx)
            idx_feat = feat_idx[idx]
            scores = b + np.sum(W[idx_feat], axis=1)
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores, dtype=np.float32)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), y[idx]] -= 1.0
            probs /= B

            flat_idx = idx_feat.reshape(-1)
            flat_grad = np.repeat(probs, nfeat, axis=0)
            grad_W = np.zeros_like(W)
            np.add.at(grad_W, flat_idx, flat_grad)
            grad_W += L2 * W
            grad_b = probs.sum(axis=0)

            t += 1
            mW = 0.9 * mW + 0.1 * grad_W
            vW = 0.999 * vW + 0.001 * grad_W * grad_W
            mW_hat = mW / (1 - 0.9 ** t)
            vW_hat = vW / (1 - 0.999 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + 1e-8)

            mb = 0.9 * mb + 0.1 * grad_b
            vb = 0.999 * vb + 0.001 * grad_b * grad_b
            mb_hat = mb / (1 - 0.9 ** t)
            vb_hat = vb / (1 - 0.999 ** t)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + 1e-8)

    header = struct.pack("<5i", MAGIC, d, 0, C, D)
    with open(weights_out, "wb") as f:
        f.write(header)
        f.write(W.astype("<f4").tobytes())
        f.write(b.astype("<f4").tobytes())

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

static uint32_t hash_ngram(int pos, int len, const unsigned char *bytes, uint32_t D) {
    uint32_t h = pos * 3 + len;
    for (int i = 0; i < len; i++) {
        h = h * 257 + bytes[i] + 1;
    }
    return h % D;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x4E475248) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], D = hd[4];
    if (H != 0 || C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 20 + 4L * D * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W = (const float *)(hd + 5);
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int p = 0; p < d; p++) {
            uint32_t idx = hash_ngram(p, 1, xi + p, D);
            const float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        for (int p = 0; p < d - 1; p++) {
            uint32_t idx = hash_ngram(p, 2, xi + p, D);
            const float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        for (int p = 0; p < d - 2; p++) {
            uint32_t idx = hash_ngram(p, 3, xi + p, D);
            const float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```