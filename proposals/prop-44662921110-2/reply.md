```hypothesis
Family: label-ngram (linear model on hashed unigram+bigram features).
Reason: Character-level prediction relies heavily on local n-grams; feature hashing captures bigram interactions with a compact fixed-size weight matrix (D=32768, C=65), avoiding the parameter explosion of explicit n-gram tables while staying faster than tree-based methods.
Failure sign: dev accuracy ≤ 0.3375 (label-linear) or Ir/item > 20000.
```
```python
import json, struct, sys
import numpy as np

# Hyperparameters (chosen on tuning split)
D = 32768          # hash table size (power of two)
SEED = 0x123456789abcdef0
LR = 0.03
L2 = 1e-5
BATCH = 64
EPOCHS = 20
F = 31             # 16 unigrams + 15 bigrams

def splitmix64(x):
    x = (x + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb & 0xFFFFFFFFFFFFFFFF
    x = x ^ (x >> 31)
    return x & 0xFFFFFFFFFFFFFFFF

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 16

    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    n = len(X_bytes)
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(n, d)
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)

    # Precompute hash indices and signs
    uni_idx = np.zeros((16, 256), dtype=np.int32)
    uni_sign = np.zeros((16, 256), dtype=np.int8)
    for i in range(16):
        for v in range(256):
            key = (i << 8) | v
            h = splitmix64(key + SEED)
            uni_idx[i, v] = h & (D - 1)
            uni_sign[i, v] = 1 if (h >> 63) == 0 else -1

    bi_idx = np.zeros((15, 256, 256), dtype=np.int32)
    bi_sign = np.zeros((15, 256, 256), dtype=np.int8)
    for i in range(15):
        for v1 in range(256):
            for v2 in range(256):
                key = (i << 16) | (v1 << 8) | v2
                h = splitmix64(key + SEED)
                bi_idx[i, v1, v2] = h & (D - 1)
                bi_sign[i, v1, v2] = 1 if (h >> 63) == 0 else -1

    # Build feature matrices for all examples
    feat_idx = np.zeros((n, F), dtype=np.int32)
    feat_sign = np.zeros((n, F), dtype=np.int8)
    for b in range(n):
        x = X[b]
        f = 0
        for i in range(16):
            feat_idx[b, f] = uni_idx[i, x[i]]
            feat_sign[b, f] = uni_sign[i, x[i]]
            f += 1
        for i in range(15):
            feat_idx[b, f] = bi_idx[i, x[i], x[i+1]]
            feat_sign[b, f] = bi_sign[i, x[i], x[i+1]]
            f += 1

    # Initialize weights
    rng = np.random.default_rng(1)
    W = rng.normal(0, 0.01, size=(D, C)).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)

    # Adam state
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    m_b = np.zeros_like(bias)
    v_b = np.zeros_like(bias)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    indices = np.arange(n)
    for epoch in range(EPOCHS):
        lr = LR * (0.5 ** (epoch // 10))
        rng.shuffle(indices)
        for start in range(0, n, BATCH):
            batch_idx = indices[start:start+BATCH]
            if len(batch_idx) == 0:
                continue
            bsz = len(batch_idx)
            idx = feat_idx[batch_idx]          # (bsz, F)
            sgn = feat_sign[batch_idx].astype(np.float32)  # (bsz, F)
            y_batch = y[batch_idx]

            # Forward
            W_idx = W[idx]                     # (bsz, F, C)
            scores = np.sum(sgn[:, :, None] * W_idx, axis=1) + bias  # (bsz, C)

            # Softmax
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            # Gradient
            grad_scores = probs
            grad_scores[np.arange(bsz), y_batch] -= 1.0
            grad_scores /= bsz

            # Dense gradient for W
            flat_idx = idx.ravel()
            flat_sgn = sgn.ravel()[:, None]            # (bsz*F, 1)
            flat_grad = np.repeat(grad_scores, F, axis=0)  # (bsz*F, C)
            grad_W = flat_sgn * flat_grad              # (bsz*F, C)

            grad_W_dense = np.zeros_like(W)
            np.add.at(grad_W_dense, flat_idx, grad_W)
            grad_W_dense += L2 * W

            grad_b = grad_scores.sum(axis=0)

            # Adam update
            t += 1
            m_W = beta1 * m_W + (1 - beta1) * grad_W_dense
            v_W = beta2 * v_W + (1 - beta2) * (grad_W_dense * grad_W_dense)
            m_hat = m_W / (1 - beta1 ** t)
            v_hat = v_W / (1 - beta2 ** t)
            W -= lr * m_hat / (np.sqrt(v_hat) + eps)

            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * (grad_b * grad_b)
            m_hat_b = m_b / (1 - beta1 ** t)
            v_hat_b = v_b / (1 - beta2 ** t)
            bias -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

    # Write weights.bin
    header = struct.pack("<3I", 0x4E47524D, D, C)
    out = header + W.astype("<f4").tobytes() + bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
    x = x ^ (x >> 31);
    return x;
}

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
    uint32_t k = buf[*pos] | (buf[*pos+1]<<8) | (buf[*pos+2]<<16) | ((uint32_t)buf[*pos+3]<<24);
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 12 || hd[0] != 0x4E47524D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    uint32_t D = hd[1], C = hd[2];
    if (nw != 12 + (long)D * C * 4 + (long)C * 4) { fprintf(stderr, "bad weights size\n"); return 3; }
    float *W = (float *)(hd + 3);
    float *bias = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    const uint64_t SEED = 0x123456789abcdef0ULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        for (int c = 0; c < C; c++) scores[c] = bias[c];

        // Unigrams
        for (int p = 0; p < 16; p++) {
            uint64_t key = ((uint64_t)p << 8) | xi[p];
            uint64_t h = splitmix64(key + SEED);
            uint32_t idx = h & (D - 1);
            float sign = (h >> 63) ? -1.0f : 1.0f;
            float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += sign * w[c];
        }
        // Bigrams
        for (int p = 0; p < 15; p++) {
            uint64_t key = ((uint64_t)p << 16) | ((uint64_t)xi[p] << 8) | xi[p+1];
            uint64_t h = splitmix64(key + SEED);
            uint32_t idx = h & (D - 1);
            float sign = (h >> 63) ? -1.0f : 1.0f;
            float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += sign * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```