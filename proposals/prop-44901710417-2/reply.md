```hypothesis
Family: label-hashed-ngram (linear model with positional unigrams + hashed bigrams for last 4 positions)
Why: Character-level language modeling has strong local dependencies (e.g., "th", "he", "in"). The linear baseline (0.3375) uses only positional unigrams. Adding 4×8192 hashed bigram features captures these n-grams with minimal collisions, adding ~32K features (total ~37K) while keeping inference at ~20 sparse additions per item. Expected to approach GBDT accuracy (0.39) at near-linear speed.
Failure: If dev accuracy ≤ 0.34 or Ir/item > 20000 (2× linear), hashing collisions or insufficient capacity hurt more than they help.
```

```python
"""Hashed n-gram linear model for label tasks with byte inputs.
Features: positional unigrams (exact) + hashed bigrams for last K positions.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy."""
import json, struct, sys, math

import numpy as np

SEED = 1
LR = 0.03
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 25
K_BIGRAMS = 4
B_BUCKETS = 8192

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    train_in = sys.argv[2]
    train_key = sys.argv[3]
    weights_out = sys.argv[4]

    d = card["input"]["bytes"]
    C = card["classes"]

    X_recs = records(train_in)
    y_recs = records(train_key)
    n = len(X_recs)
    X = np.frombuffer(b"".join(X_recs), dtype=np.uint8).reshape(n, d)
    y = np.array([r[0] for r in y_recs], dtype=np.int64)

    F_uni = d * 256
    F_bi = K_BIGRAMS * B_BUCKETS
    F = F_uni + F_bi

    rng = np.random.default_rng(SEED)
    scale = math.sqrt(6.0 / F)
    weights = rng.uniform(-scale, scale, (F, C)).astype(np.float32)
    bias = np.zeros(C, np.float32)

    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = np.zeros_like(bias)
    v_b = np.zeros_like(bias)

    uni_feats = np.arange(d) * 256 + X
    bi_feats = np.zeros((n, K_BIGRAMS), dtype=np.int32)
    for bp in range(K_BIGRAMS):
        pos = d - 2 - bp
        a = X[:, pos].astype(np.uint32)
        b = X[:, pos + 1].astype(np.uint32)
        h = (a * 257 + b) & 0xFFFFFFFF
        h ^= h >> 16
        h = (h * 0x7feb352d) & 0xFFFFFFFF
        h ^= h >> 15
        h = (h * 0x846ca68b) & 0xFFFFFFFF
        h ^= h >> 16
        bi_feats[:, bp] = (h % B_BUCKETS).astype(np.int32)
    bi_feats += F_uni + np.arange(K_BIGRAMS) * B_BUCKETS

    all_feats = np.concatenate([uni_feats, bi_feats], axis=1).astype(np.int32)
    feat_per_ex = d + K_BIGRAMS

    t = 0
    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for start in range(0, n, BATCH):
            idx = order[start:start+BATCH]
            bs = len(idx)
            batch_feats = all_feats[idx].reshape(-1)
            ex_idx = np.repeat(np.arange(bs), feat_per_ex)

            scores = bias + np.sum(weights[batch_feats].reshape(bs, feat_per_ex, C), axis=1)
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores, dtype=np.float32)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            grad_scores = probs
            grad_scores[np.arange(bs), y[idx]] -= 1.0
            grad_scores /= bs

            weights_grad = np.zeros_like(weights)
            np.add.at(weights_grad, batch_feats, grad_scores[ex_idx])
            bias_grad = grad_scores.sum(axis=0)

            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            weights_grad += L2 * weights
            m_w *= beta1; m_w += (1 - beta1) * weights_grad
            v_w *= beta2; v_w += (1 - beta2) * weights_grad * weights_grad
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            weights -= LR * m_w_hat / (np.sqrt(v_w_hat) + eps)

            bias_grad += L2 * bias
            m_b *= beta1; m_b += (1 - beta1) * bias_grad
            v_b *= beta2; v_b += (1 - beta2) * bias_grad * bias_grad
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)
            bias -= LR * m_b_hat / (np.sqrt(v_b_hat) + eps)

    magic = 0x484E4731
    header = struct.pack("<6i", magic, d, C, K_BIGRAMS, B_BUCKETS, F)
    w_bytes = weights.astype("<f4").tobytes()
    b_bytes = bias.astype("<f4").tobytes()
    with open(weights_out, "wb") as f:
        f.write(header)
        f.write(w_bytes)
        f.write(b_bytes)

if __name__ == "__main__":
    main()
```

```c
/* Hashed n-gram linear model: ./infer weights.bin in.rec N
 * Features: positional unigrams + hashed bigrams for last K positions.
 * Weight format: int32 magic(0x484E4731), d, C, K, B, F; float32 weights[F*C], bias[C]. */
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

static uint32_t hash2(uint8_t a, uint8_t b) {
    uint32_t h = (uint32_t)a * 257 + b;
    h ^= h >> 16;
    h *= 0x7feb352d;
    h ^= h >> 15;
    h *= 0x846ca68b;
    h ^= h >> 16;
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 24 || hd[0] != 0x484E4731) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], K = hd[3], B = hd[4], F = hd[5];
    long expected = 24 + 4L * F * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *weights = (const float *)(hd + 6);
    const float *bias = weights + (long)F * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(scores, bias, sizeof(float) * C);
        for (int j = 0; j < d; j++) {
            int idx = j * 256 + xi[j];
            const float *w = weights + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        for (int bp = 0; bp < K; bp++) {
            int pos_idx = d - 2 - bp;
            uint32_t h = hash2(xi[pos_idx], xi[pos_idx + 1]);
            int idx = d * 256 + bp * B + (h % B);
            const float *w = weights + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```