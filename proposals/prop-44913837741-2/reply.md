```hypothesis
Family: label-hashngram (hashed n-gram logistic regression).
Why: Next-character prediction is driven by local sequential patterns (n-grams). Label-linear uses position-specific one-hot vectors, missing cross-position interactions like "t" followed by "h". Hashed n-grams (orders 1–3, position-tagged) capture these Markovian dependencies in fixed memory (D=65536) with sparse Adam updates.
Failure signal: Dev accuracy ≤ 0.3375 (label-linear) or CPU cost > 2× label-linear, indicating hash collisions or insufficient n-gram order outweigh interaction gains.
```
```python
"""Hashed n-gram logistic regression for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Features: all n-grams of length 1..3 from the 16-byte context, tagged by ending position.
Hashed to D=65536 buckets with FNV-1a. Trained with Adam (lr=3e-3, L2=1e-4, batch=64).
Deterministic seed 1. Exports: magic 0x484E4731, d=16, D, C, max_n=3; float32 W[D][C], b[C]."""
import json, math, struct, sys

import numpy as np

D = 65536          # hash table size (power of 2)
MAX_N = 3          # maximum n-gram length
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
FNV_OFFSET, FNV_PRIME = 0x811c9dc5, 0x01000193
MASK = D - 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def hash_ngram(pos, *bytes_):
    """FNV-1a hash of (pos, b1, b2, ...) modulo D."""
    h = FNV_OFFSET
    h ^= pos; h = (h * FNV_PRIME) & 0xffffffff
    for b in bytes_:
        h ^= b; h = (h * FNV_PRIME) & 0xffffffff
    return h & MASK

def featurize(x):  # x: (16,) uint8 -> list of feature indices
    idxs = []
    for i in range(16):
        idxs.append(hash_ngram(i, x[i]))
        if i >= 1:
            idxs.append(hash_ngram(i, x[i-1], x[i]))
        if i >= 2:
            idxs.append(hash_ngram(i, x[i-2], x[i-1], x[i]))
    return idxs

def main():
    card = json.load(open(sys.argv[1]))
    X_raw = [r for r in records(sys.argv[2])]
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = len(X_raw)
    assert all(len(r) == 16 for r in X_raw)

    # Precompute feature indices for all training samples
    X_feat = [featurize(np.frombuffer(r, dtype=np.uint8)) for r in X_raw]

    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            batch_sz = len(idx)
            # Accumulate gradients for active features only
            gW = np.zeros_like(W)
            gb = np.zeros_like(b)
            for i in idx:
                feat = X_feat[i]
                # logits = sum(W[f]) + b
                logits = W[feat].sum(axis=0) + b
                logits -= logits.max()
                exp_logits = np.exp(logits, dtype=np.float32)
                probs = exp_logits / exp_logits.sum()
                probs[y[i]] -= 1.0
                probs /= batch_sz
                gW[feat] += probs
                gb += probs
            # L2 on all weights (dense) - approximate by scaling active, but simpler: skip L2 on inactive
            # Actually apply L2 to all W for correctness (dense update)
            gW += L2 * W
            gb += L2 * b
            t += 1
            # Adam update
            mW *= 0.9; mW += 0.1 * gW
            vW *= 0.999; vW += 0.001 * gW * gW
            mb *= 0.9; mb += 0.1 * gb
            vb *= 0.999; vb += 0.001 * gb * gb
            mW_hat = mW / (1 - 0.9 ** t)
            vW_hat = vW / (1 - 0.999 ** t)
            mb_hat = mb / (1 - 0.9 ** t)
            vb_hat = vb / (1 - 0.999 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + 1e-8)

    # Write weights
    hdr = struct.pack("<4i", 0x484E4731, 16, D, C, MAX_N)  # magic, d, D, C, max_n
    open(sys.argv[4], "wb").write(hdr + W.astype("<f4").tobytes() + b.astype("<f4").tobytes())

if __name__ == "__main__":
    main()
```
```c
/* Hashed n-gram logistic regression: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x484E4731, int32 d=16, D, C, max_n=3; float32 W[D][C], b[C]. */
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

static uint32_t fnv1a(uint32_t h, unsigned char b) {
    h ^= b; h *= 0x01000193u;
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x484E4731) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3], max_n = hd[4];
    if (d != 16 || max_n != 3 || C < 1 || C > 256 ||
        nw != 20 + 4L * ((long)D * C + C)) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(hd + 5), *b = W + (long)D * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        memcpy(scores, b, C * sizeof(float));
        for (int p = 0; p < 16; p++) {
            uint32_t h = 0x811c9dc5u;
            h = fnv1a(h, p);
            h = fnv1a(h, xi[p]);
            scores[0] += W[(h & (D-1)) * C];
            // unroll for speed
            for (int c = 1; c < C; c++) scores[c] += W[(h & (D-1)) * C + c];
            if (p >= 1) {
                h = 0x811c9dc5u;
                h = fnv1a(h, p);
                h = fnv1a(h, xi[p-1]);
                h = fnv1a(h, xi[p]);
                for (int c = 0; c < C; c++) scores[c] += W[(h & (D-1)) * C + c];
            }
            if (p >= 2) {
                h = 0x811c9dc5u;
                h = fnv1a(h, p);
                h = fnv1a(h, xi[p-2]);
                h = fnv1a(h, xi[p-1]);
                h = fnv1a(h, xi[p]);
                for (int c = 0; c < C; c++) scores[c] += W[(h & (D-1)) * C + c];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```