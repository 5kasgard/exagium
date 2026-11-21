```hypothesis
Family: label-ngram (linear model with hashed bigram features). The unigram-only linear model (0.3375) misses local character dependencies like "th"→"e" or "qu"→"u" that are crucial for character-level language modeling. Adding 15 bigram features (adjacent pairs from the 16-byte context) hashed to 2048 buckets captures these n-gram statistics while keeping inference fast (~31 table lookups per item). If dev accuracy ≤ 0.345, the bigram features don't help enough to justify the extra cost.
```
```python
"""label-ngram: linear model with unigram + hashed bigram features for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Weights format: magic 0x4E475231, d, H=0, C, K; float32 T[d][256][C], B[K][C], bias[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
K = 2048  # bigram hash buckets (power of 2)


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
    C = card["classes"]
    d = X.shape[1]          # 16
    rng = np.random.default_rng(SEED)

    # Build feature encodings: unigram (categorical per position) + bigram hashes
    enc = []
    feat_dim = 0
    # Unigram: one-hot per position over observed values
    for j in range(d):
        vals = np.unique(X[:, j])
        E = np.zeros((256, len(vals)), dtype=np.float32)
        E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
        feat_dim += len(vals)
    # Bigram: 15 adjacent pairs, each hashed to K buckets (one-hot in hash space)
    for _ in range(d - 1):
        E = np.zeros((K, K), dtype=np.float32)
        E[np.arange(K), np.arange(K)] = 1.0
        enc.append(E)
        feat_dim += K

    # Construct design matrix Z
    Z_parts = []
    for j, E in enumerate(enc):
        if j < d:
            Z_parts.append(E[X[:, j]])
        else:
            bp = j - d
            c1 = X[:, bp].astype(np.uint32)
            c2 = X[:, bp + 1].astype(np.uint32)
            h = (c1 * 65 + c2) & (K - 1)  # hash to [0, K)
            Z_parts.append(E[h])
    Z = np.concatenate(Z_parts, axis=1).astype(np.float32)
    n, k = Z.shape
    assert k == feat_dim

    # Linear weights: W1[k, C], b1[C]
    W1 = np.zeros((k, C), dtype=np.float32)
    b1 = np.zeros(C, dtype=np.float32)
    params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def sg(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            g = sg(z @ W1 + b1, yy)
            grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Fold unigram weights into per-position per-byte tables T[d][256][C]
    T = np.zeros((d, 256, C), dtype=np.float32)
    off = 0
    for j in range(d):
        E = enc[j]
        w = W1[off:off + E.shape[1]]
        T[j] = E @ w
        off += E.shape[1]
    # Fold bigram weights into hash tables B[K][C]
    B = np.zeros((K, C), dtype=np.float32)
    for _ in range(d - 1):
        E = enc[off // K]  # all bigram encodings are identity in hash space
        w = W1[off:off + K]
        B += E @ w  # == w since E is identity
        off += K
    assert off == k

    # Write weights
    magic = 0x4E475231  # 'NGR1'
    hdr = struct.pack("<5i", magic, d, 0, C, K)
    out = hdr + T.astype("<f4").tobytes() + B.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-ngram inference: ./infer weights.bin in.rec N
 * Input: 16-byte records. Output: 1 byte class index per record.
 * Weights: magic 0x4E475231, d, H=0, C, K; T[d][256][C], B[K][C], bias[C] (all float32 LE). */
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
    if (nw < 20 || hd[0] != 0x4E475231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], K = hd[4];
    if (H != 0 || C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad header\n"); return 3; }
    long expect = 20L + 4L * ((long)d * 256 * C + (long)K * C + C);
    if (nw != expect) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *B = T + (long)d * 256 * C;
    const float *bias = B + (long)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) acc[c] = bias[c];

        /* unigrams */
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) acc[c] += t[c];
        }

        /* bigrams: 15 adjacent pairs */
        for (int j = 0; j < d - 1; j++) {
            uint32_t h = (xi[j] * 65u + xi[j + 1]) & (K - 1);
            const float *b = B + (long)h * C;
            for (int c = 0; c < C; c++) acc[c] += b[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```