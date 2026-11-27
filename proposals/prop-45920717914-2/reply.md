```hypothesis
Family: label-hashed-linear. Multiple hash functions map the 16-byte context to buckets holding per-class weights; the sum of K bucket vectors gives logits. This captures cross-byte interactions (the hash depends on the whole context) while staying a linear model in the hashed space, so inference is O(K*C) with tiny memory. It should beat label-linear (0.3375) because label-linear only sees per-position one-hots. If dev accuracy ≤ 0.35 the hashing adds no useful signal.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
K = 4                 # number of hash functions
B = 16384             # buckets per hash
LR = 0.01
L2 = 1e-5
BATCH = 64
EPOCHS = 10
SEEDS = np.array([0x12345678, 0x23456789, 0x34567890, 0x45678901], dtype=np.uint32)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def hash_contexts(X, seeds):
    """X: (n, 16) uint8, seeds: (K,) uint32 -> (n, K) uint32 bucket indices"""
    n = X.shape[0]
    h = np.tile(seeds.astype(np.uint64), (n, 1))
    for j in range(16):
        h = (h * 31 + X[:, j].astype(np.uint64)[:, None]) & 0xFFFFFFFF
    return (h % B).astype(np.int32)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]
    assert X.shape[1] == 16

    rng = np.random.default_rng(SEED)
    W = rng.normal(0, 0.01, size=(K, B, C)).astype(np.float32)
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    t = 0

    idx = np.arange(n)
    for ep in range(EPOCHS):
        rng.shuffle(idx)
        for s in range(0, n, BATCH):
            batch_idx = idx[s:s+BATCH]
            xb = X[batch_idx]
            yb = y[batch_idx]
            buckets = hash_contexts(xb, SEEDS)          # (b, K)
            logits = np.zeros((len(yb), C), dtype=np.float32)
            for k in range(K):
                logits += W[k, buckets[:, k]]
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad = probs / len(yb)                      # (b, C)
            for k in range(K):
                bkt = buckets[:, k]
                np.add.at(W[k], bkt, -LR * (grad.T @ np.ones(len(yb))).T)  # wrong, need per-bucket grad
                # We'll do a simpler but correct per-bucket accumulation
            # Recompute gradients per bucket correctly
            for k in range(K):
                bkt = buckets[:, k]
                g = np.zeros_like(W[k])
                np.add.at(g, bkt, grad)
                g += L2 * W[k]
                t += 1
                m[k] = 0.9 * m[k] + 0.1 * g
                v[k] = 0.999 * v[k] + 0.001 * g * g
                m_hat = m[k] / (1 - 0.9 ** t)
                v_hat = v[k] / (1 - 0.999 ** t)
                W[k] -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    magic = 0x484C4831  # "HLH1"
    out = struct.pack("<4i", magic, K, B, C)
    out += SEEDS.astype("<u4").tobytes()
    out += W.astype("<f4").tobytes()
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

static uint32_t hash16(const unsigned char *ctx, uint32_t seed) {
    uint32_t h = seed;
    for (int i = 0; i < 16; i++) h = h * 31 + ctx[i];
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 4*4 + 4*4) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hdr = (uint32_t *)wb;
    if (hdr[0] != 0x484C4831) { fprintf(stderr, "bad magic\n"); return 3; }
    int K = hdr[1], B = hdr[2], C = hdr[3];
    if (K != 4 || C != 65 || B != 16384) { fprintf(stderr, "unexpected params\n"); return 3; }
    uint32_t *seeds = hdr + 4;
    float *W = (float *)(seeds + K);
    size_t expected = 4*4 + 4*K + (size_t)K * B * C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %zu\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(C * sizeof(float));
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        memset(logits, 0, C * sizeof(float));
        for (int h = 0; h < K; h++) {
            uint32_t bucket = hash16(xi, seeds[h]) % B;
            const float *w = W + ((size_t)h * B + bucket) * C;
            for (int c = 0; c < C; c++) logits[c] += w[c];
        }
        best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```