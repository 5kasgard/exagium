```hypothesis
Family: label-ngram (character n-gram logistic regression with exact unigrams + hashed bigrams/trigrams).
Why: TinyShakespeare has strong local sequential structure; bigrams/trigrams capture context that independent per-position one-hots miss. Hash tables (2^16) bound memory while sharing stats across positions. Joint Adam training on all n-gram orders should beat label-linear's 0.3375 and approach label-gbdt's 0.3923 at lower inference cost (31 table lookups vs tree walks).
Failure: If dev accuracy ≤ 0.34 or inference ns/item > 2000, the hash collisions or capacity are insufficient.
```
```python
import json, math, struct, sys

import numpy as np

TABLE_BITS = 16
TABLE_SIZE = 1 << TABLE_BITS
MASK = TABLE_SIZE - 1
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def mix32(h):
    h ^= h >> 16
    h *= 0x45d9f3b
    h ^= h >> 16
    h *= 0x45d9f3b
    h ^= h >> 16
    return h & 0xFFFFFFFF


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = b"".join(records(sys.argv[2]))
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n, d = X.shape
    assert d == 16

    n_uni = d * 256
    n_bi = d - 1
    n_tri = d - 2

    bi_hashes = np.zeros((n, n_bi), dtype=np.uint32)
    tri_hashes = np.zeros((n, n_tri), dtype=np.uint32)

    for pos in range(n_bi):
        h = mix32((((pos + 1) * 257 + X[:, pos]) * 257 + X[:, pos + 1]))
        bi_hashes[:, pos] = h & MASK

    for pos in range(n_tri):
        h = mix32(((((pos + 1) * 257 + X[:, pos]) * 257 + X[:, pos + 1]) * 257 + X[:, pos + 2]))
        tri_hashes[:, pos] = h & MASK

    W_uni = np.zeros((n_uni, C), dtype=np.float32)
    W_bi = np.zeros((TABLE_SIZE, C), dtype=np.float32)
    W_tri = np.zeros((TABLE_SIZE, C), dtype=np.float32)

    m_uni, v_uni = np.zeros_like(W_uni), np.zeros_like(W_uni)
    m_bi, v_bi = np.zeros_like(W_bi), np.zeros_like(W_bi)
    m_tri, v_tri = np.zeros_like(W_tri), np.zeros_like(W_tri)

    rng = np.random.default_rng(SEED)
    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            bsz = len(idx)

            uni_idx = X[idx] + (np.arange(d, dtype=np.uint16) * 256)
            bi_idx = bi_hashes[idx]
            tri_idx = tri_hashes[idx]

            logits = np.zeros((bsz, C), dtype=np.float32)
            logits += W_uni[uni_idx].sum(axis=1)
            logits += W_bi[bi_idx].sum(axis=1)
            logits += W_tri[tri_idx].sum(axis=1)

            yy = y[idx]
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yy] -= 1.0
            grad = (probs / bsz).astype(np.float32)

            g_uni = np.zeros_like(W_uni)
            g_bi = np.zeros_like(W_bi)
            g_tri = np.zeros_like(W_tri)

            for p in range(d):
                np.add.at(g_uni, uni_idx[:, p], grad)
            for p in range(n_bi):
                np.add.at(g_bi, bi_idx[:, p], grad)
            for p in range(n_tri):
                np.add.at(g_tri, tri_idx[:, p], grad)

            g_uni += L2 * W_uni
            g_bi += L2 * W_bi
            g_tri += L2 * W_tri

            t += 1
            for W, m, v, g in [(W_uni, m_uni, v_uni, g_uni),
                               (W_bi, m_bi, v_bi, g_bi),
                               (W_tri, m_tri, v_tri, g_tri)]:
                m *= 0.9; m += 0.1 * g
                v *= 0.999; v += 0.001 * g * g
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                W -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    out = struct.pack("<5i", 0x4E47524D, d, n_uni, TABLE_SIZE, C)
    out += W_uni.astype("<f4").tobytes()
    out += W_bi.astype("<f4").tobytes()
    out += W_tri.astype("<f4").tobytes()
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

static uint32_t mix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x4E47524D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], n_uni = hd[2], table_size = hd[3], C = hd[4];
    if (d != 16 || n_uni != 4096 || C < 1 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20 + 4L * ((long)n_uni * C + 2L * table_size * C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *W_uni = (const float *)(hd + 5);
    const float *W_bi = W_uni + (long)n_uni * C;
    const float *W_tri = W_bi + (long)table_size * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = calloc(C, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) logits[c] = 0.0f;

        for (int p = 0; p < d; p++) {
            int idx = p * 256 + xi[p];
            const float *w = W_uni + (long)idx * C;
            for (int c = 0; c < C; c++) logits[c] += w[c];
        }

        for (int p = 0; p < d - 1; p++) {
            uint32_t h = mix32((((p + 1) * 257 + xi[p]) * 257 + xi[p + 1]));
            h &= table_size - 1;
            const float *w = W_bi + (long)h * C;
            for (int c = 0; c < C; c++) logits[c] += w[c];
        }

        for (int p = 0; p < d - 2; p++) {
            uint32_t h = mix32(((((p + 1) * 257 + xi[p]) * 257 + xi[p + 1]) * 257 + xi[p + 2]));
            h &= table_size - 1;
            const float *w = W_tri + (long)h * C;
            for (int c = 0; c < C; c++) logits[c] += w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```