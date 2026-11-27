```hypothesis
The label-ngram family uses a linear model on hashed unigram and bigram features from the 16-byte context. Bigram features capture local character interactions (e.g., "th" → 'e') that unigrams alone miss, improving accuracy over label-linear. The hash table size (32768) balances collisions and memory. If accuracy does not exceed label-linear on dev, it shows that bigram features are not sufficiently informative or collisions hurt too much.
```
```python
import json, math, struct, sys
import numpy as np

# Hyperparameters
H = 32768  # hash table size
SEED = 1
LR = 1e-3
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

def hash32(x):
    # 32-bit hash from https://github.com/skeeto/hash-prospector
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    x = (x * 0x7feb352d) & 0xFFFFFFFF
    x = (x ^ (x >> 15)) & 0xFFFFFFFF
    x = (x * 0x846ca68b) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    return x

def get_features(context):
    # context: 16 bytes
    feats = []
    # unigrams
    for i, b in enumerate(context):
        key = (i << 8) | b
        feats.append(hash32(key) % H)
    # bigrams
    for i in range(15):
        b1 = context[i]
        b2 = context[i+1]
        key = 0x10000 | (i << 16) | (b1 << 8) | b2
        feats.append(hash32(key) % H)
    return feats

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 16
    n = len(X_bytes)
    # Precompute features
    feat_list = [get_features(x) for x in X_bytes]
    # Initialize weights
    rng = np.random.default_rng(SEED)
    # weights: H x C
    W = rng.normal(0, 1.0/math.sqrt(H), size=(H, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    # Adam state
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            # Compute scores and gradients
            grad_W = np.zeros_like(W)
            grad_b = np.zeros_like(b)
            for i in idx:
                feats = feat_list[i]
                yy = y[i]
                # scores
                scores = b.copy()
                for f in feats:
                    scores += W[f]
                # softmax
                scores -= scores.max()
                exp_scores = np.exp(scores)
                probs = exp_scores / exp_scores.sum()
                probs[yy] -= 1.0
                # gradients
                grad_b += probs
                for f in feats:
                    grad_W[f] += probs
            # Average over batch
            grad_W /= len(idx)
            grad_b /= len(idx)
            # L2 regularization on W only
            grad_W += L2 * W
            # Adam update
            t += 1
            # Update W
            mW = 0.9 * mW + 0.1 * grad_W
            vW = 0.999 * vW + 0.001 * grad_W * grad_W
            mW_hat = mW / (1 - 0.9**t)
            vW_hat = vW / (1 - 0.999**t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            # Update b
            mb = 0.9 * mb + 0.1 * grad_b
            vb = 0.999 * vb + 0.001 * grad_b * grad_b
            mb_hat = mb / (1 - 0.9**t)
            vb_hat = vb / (1 - 0.999**t)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + 1e-8)
    # Write weights
    magic = 0x4E47524D  # 'NGRM'
    out = struct.pack("<3i", magic, H, C)
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

// 32-bit hash function (same as Python)
static uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    uint32_t magic = hd[0];
    if (magic != 0x4E47524D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int H = hd[1];
    int C = hd[2];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    // weights: H * C floats, then C floats bias
    long expected = 12 + (long)H * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(wb + 12);
    const float *b = (const float *)(wb + 12 + (long)H * C * 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        // Initialize scores with bias
        for (int c = 0; c < C; c++) scores[c] = b[c];
        // Unigrams
        for (int p = 0; p < 16; p++) {
            uint32_t key = (p << 8) | xi[p];
            uint32_t idx = hash32(key) % H;
            const float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        // Bigrams
        for (int p = 0; p < 15; p++) {
            uint32_t key = 0x10000 | (p << 16) | (xi[p] << 8) | xi[p+1];
            uint32_t idx = hash32(key) % H;
            const float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```