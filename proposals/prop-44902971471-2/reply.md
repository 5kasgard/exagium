```hypothesis
Family: label-hashngram — linear model on position unigrams + hashed suffix n-grams (bi/tri/4-gram).
Why: next-character prediction is dominated by local suffix statistics; label-linear only sees position-wise unigrams.
Hashed n-grams add 3×4096 sparse features capturing 2–4 byte suffixes with minimal collisions.
Inference: ~19 table lookups + adds/item (vs 16 for label-linear), ~4 MB weights.
Failure mode: if hash collisions drown signal or linear weighting can't exploit n-gram features, accuracy ≤ label-linear.
```
```python
import json, math, struct, sys
import numpy as np

HASH_SIZE = 4096
NGRAM_ORDERS = (2, 3, 4)
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.05, 1e-5, 256, 500000, 50


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, 16)
    n, d = X.shape
    assert d == 16
    C = card["classes"]

    UNIGRAM_SIZE = d * 256
    NGRAM_FEAT_SIZE = len(NGRAM_ORDERS) * HASH_SIZE
    TOTAL_FEATS = UNIGRAM_SIZE + NGRAM_FEAT_SIZE

    rng = np.random.default_rng(SEED)
    W = (rng.normal(0, 0.01, (TOTAL_FEATS, C))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    idx_all = np.arange(n)

    for _ in range(epochs):
        rng.shuffle(idx_all)
        for s in range(0, n, BATCH):
            idx = idx_all[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            bs = len(idx)

            scores = np.tile(b, (bs, 1))
            for j in range(d):
                scores += W[j * 256 + Xb[:, j]]

            for o_idx, order in enumerate(NGRAM_ORDERS):
                base = UNIGRAM_SIZE + o_idx * HASH_SIZE
                h = np.zeros(bs, dtype=np.int32)
                for i in range(bs):
                    hv = 0
                    for k in range(order):
                        hv = (hv * 31 + Xb[i, d - order + k]) & (HASH_SIZE - 1)
                    h[i] = hv
                scores += W[base + h]

            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad = (probs / bs).astype(np.float32)

            grad_W = np.zeros_like(W)
            grad_b = grad.sum(axis=0)

            for j in range(d):
                np.add.at(grad_W, j * 256 + Xb[:, j], grad)

            for o_idx, order in enumerate(NGRAM_ORDERS):
                base = UNIGRAM_SIZE + o_idx * HASH_SIZE
                h = np.zeros(bs, dtype=np.int32)
                for i in range(bs):
                    hv = 0
                    for k in range(order):
                        hv = (hv * 31 + Xb[i, d - order + k]) & (HASH_SIZE - 1)
                    h[i] = hv
                np.add.at(grad_W, base + h, grad)

            grad_W += L2 * W
            grad_b += L2 * b

            t += 1
            mW = beta1 * mW + (1 - beta1) * grad_W
            vW = beta2 * vW + (1 - beta2) * grad_W * grad_W
            W -= LR * (mW / (1 - beta1 ** t)) / (np.sqrt(vW / (1 - beta2 ** t)) + eps)

            mb = beta1 * mb + (1 - beta1) * grad_b
            vb = beta2 * vb + (1 - beta2) * grad_b * grad_b
            b -= LR * (mb / (1 - beta1 ** t)) / (np.sqrt(vb / (1 - beta2 ** t)) + eps)

    magic = 0x484E474C
    header = struct.pack("<5i", magic, d, HASH_SIZE, C, len(NGRAM_ORDERS))
    open(sys.argv[4], "wb").write(header + W.astype("<f4").tobytes() + b.astype("<f4").tobytes())


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

static uint32_t hash_bytes(const unsigned char *bytes, int start, int len, uint32_t mask) {
    uint32_t h = 0;
    for (int i = 0; i < len; i++) h = (h * 31 + bytes[start + i]) & mask;
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x484E474C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], hash_size = hd[2], C = hd[3], ngram_orders = hd[4];
    if (d != 16 || C < 1 || C > 256 || ngram_orders != 3 || hash_size != 4096) {
        fprintf(stderr, "bad weights params\n"); return 3; }
    int unigram_size = d * 256;
    int total_feats = unigram_size + ngram_orders * hash_size;
    long expected = 20 + 4L * total_feats * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W = (const float *)(hd + 5);
    const float *b = W + (long)total_feats * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(scores, b, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *w = W + (long)(j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        int orders[3] = {2, 3, 4};
        for (int o = 0; o < 3; o++) {
            int base = unigram_size + o * hash_size;
            uint32_t h = hash_bytes(xi, d - orders[o], orders[o], hash_size - 1);
            const float *w = W + (long)(base + h) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```