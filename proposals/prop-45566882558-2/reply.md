```hypothesis
Family: label-hashngram. A linear model on FNV-1a hashed n-gram features (unigrams, bigrams, trigrams) from the 16-byte context. Captures local sequential dependencies that independent-byte models miss, with a fixed 8192-bucket feature space for fast inference. If dev accuracy ≤ 0.3375 (label-linear), the n-gram features don't help for this task.
```
```python
import json, math, struct, sys

import numpy as np

BUCKETS = 8192
MAX_N = 3
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fnv1a(data):
    h = 0x811c9dc5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xffffffff
    return h


def context_hashes(ctx):
    h = []
    d = len(ctx)
    for i in range(d):
        h.append(fnv1a(bytes([1, i, ctx[i]])) & (BUCKETS - 1))
    for i in range(d - 1):
        h.append(fnv1a(bytes([2, i, ctx[i], ctx[i + 1]])) & (BUCKETS - 1))
    for i in range(d - 2):
        h.append(fnv1a(bytes([3, i, ctx[i], ctx[i + 1], ctx[i + 2]])) & (BUCKETS - 1))
    return h


def main():
    card = json.load(open(sys.argv[1]))
    assert card["input"]["bytes"] == 16
    C = card["classes"]
    assert C == 65

    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    n = len(X_recs)
    assert len(y_recs) == n
    assert all(len(r) == 16 for r in X_recs)
    assert all(len(r) == 1 for r in y_recs)

    X = [bytes(r) for r in X_recs]
    y = np.array([r[0] for r in y_recs], dtype=np.int64)

    all_hashes = [context_hashes(ctx) for ctx in X]
    n_features = len(all_hashes[0])
    assert all(len(h) == n_features for h in all_hashes)

    rng = np.random.default_rng(SEED)

    W = np.zeros((BUCKETS, C), dtype=np.float32)
    b = np.zeros(C, dtype=np.float32)
    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    mb = np.zeros_like(b)
    vb = np.zeros_like(b)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            bs = len(idx)

            scores = np.tile(b, (bs, 1))
            for i, ex_idx in enumerate(idx):
                for bucket in all_hashes[ex_idx]:
                    scores[i] += W[bucket]

            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            yy = y[idx]
            probs[np.arange(bs), yy] -= 1.0
            grads = probs / bs

            grad_W = np.zeros_like(W)
            grad_b = np.zeros_like(b)
            touched = np.zeros(BUCKETS, dtype=bool)

            for i, ex_idx in enumerate(idx):
                g = grads[i]
                grad_b += g
                for bucket in all_hashes[ex_idx]:
                    grad_W[bucket] += g
                    touched[bucket] = True

            touched_idx = np.where(touched)[0]
            grad_W[touched_idx] += L2 * W[touched_idx]
            grad_b += L2 * b

            t += 1
            mW *= BETA1
            mW += (1 - BETA1) * grad_W
            vW *= BETA2
            vW += (1 - BETA2) * grad_W * grad_W
            mW_hat = mW / (1 - BETA1 ** t)
            vW_hat = vW / (1 - BETA2 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + EPS)

            mb *= BETA1
            mb += (1 - BETA1) * grad_b
            vb *= BETA2
            vb += (1 - BETA2) * grad_b * grad_b
            mb_hat = mb / (1 - BETA1 ** t)
            vb_hat = vb / (1 - BETA2 ** t)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + EPS)

    magic = 0x48534148
    header = struct.pack("<5i", magic, 16, BUCKETS, C, MAX_N)
    out = header + b.astype("<f4").tobytes() + W.astype("<f4").tobytes()
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

static uint32_t fnv1a(const unsigned char *data, int len) {
    uint32_t h = 0x811c9dc5;
    for (int i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x01000193;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x48534148) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], buckets = hd[2], C = hd[3], max_n = hd[4];
    if (d != 16 || buckets != 8192 || C != 65 || max_n != 3) {
        fprintf(stderr, "bad weights config\n"); return 3; }
    long expected = 20 + 4L * C + 4L * buckets * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *b = (const float *)(hd + 5);
    const float *W = b + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    unsigned char hash_buf[5];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        for (int c = 0; c < C; c++) scores[c] = b[c];

        for (int p = 0; p < 16; p++) {
            hash_buf[0] = 1; hash_buf[1] = p; hash_buf[2] = xi[p];
            uint32_t h = fnv1a(hash_buf, 3) & (buckets - 1);
            const float *wrow = W + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += wrow[c];
        }
        for (int p = 0; p < 15; p++) {
            hash_buf[0] = 2; hash_buf[1] = p; hash_buf[2] = xi[p]; hash_buf[3] = xi[p + 1];
            uint32_t h = fnv1a(hash_buf, 4) & (buckets - 1);
            const float *wrow = W + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += wrow[c];
        }
        for (int p = 0; p < 14; p++) {
            hash_buf[0] = 3; hash_buf[1] = p; hash_buf[2] = xi[p]; hash_buf[3] = xi[p + 1]; hash_buf[4] = xi[p + 2];
            uint32_t h = fnv1a(hash_buf, 5) & (buckets - 1);
            const float *wrow = W + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += wrow[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```