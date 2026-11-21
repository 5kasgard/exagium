```hypothesis
Family: hash-ngram (hashed n-gram logistic regression). Why: next-character prediction relies on recent character sequences; label-linear uses independent per-position one-hots (4096 features) missing n-gram interactions. This model hashes suffixes of length 1..5 from the 16-byte context into a 65536-bin feature space via FNV-1a, then trains multinomial logistic regression with Adam. Inference: 5 hash computations + 5×65 adds per item, similar CPU cost to label-linear. Expected to beat label-linear (0.3375) and approach label-gbdt (0.3923). Failure: dev accuracy ≤ 0.34 or inference cost > 2× label-linear.
```
```python
"""Hashed n-gram logistic regression for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Extracts suffixes of length 1..MAX_N from 16-byte context, hashes to FEAT_SIZE bins,
trains multinomial logistic regression with Adam. Exports: magic, d, feat_size, C, max_n, W[feat_size, C], bias[C]."""
import json, math, struct, sys

import numpy as np

FEAT_SIZE = 65536
MAX_N = 5
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x484E4731  # 'HNG1'

def fnv1a_64_arr(arr):
    h = np.uint64(0xcbf29ce484222325)
    for b in arr:
        h ^= np.uint64(b)
        h *= np.uint64(0x100000001b3)
    return h

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def compute_hashes(X, max_n, feat_size):
    n, d = X.shape
    hashes = np.zeros((n, max_n), dtype=np.int32)
    mask = feat_size - 1
    for i in range(n):
        for L in range(1, max_n + 1):
            suffix = X[i, -L:]
            h = fnv1a_64_arr(suffix)
            hashes[i, L-1] = int(h & mask)
    return hashes

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 16
    
    rng = np.random.default_rng(SEED)
    
    # Precompute n-gram hashes
    hashes = compute_hashes(X, MAX_N, FEAT_SIZE)
    n = X.shape[0]
    
    # Initialize weights
    W = (rng.uniform(-1, 1, (FEAT_SIZE, C)) * math.sqrt(6.0 / FEAT_SIZE)).astype(np.float32)
    bias = np.zeros(C, np.float32)
    params = [W, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    
    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            h_b = hashes[idx]  # (B, MAX_N)
            yy = y[idx]
            B = len(yy)
            
            # Forward: scores = bias + sum_k W[h_b[:, k]]
            scores = bias + np.sum(W[h_b], axis=1)  # (B, C)
            
            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores, dtype=np.float32)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            grad_scores = (probs / B).astype(np.float32)  # (B, C)
            
            # Gradients
            grad_bias = grad_scores.sum(axis=0)
            grad_W = np.zeros_like(W)
            for k in range(MAX_N):
                np.add.at(grad_W, h_b[:, k], grad_scores)
            grad_W += L2 * W
            
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, [grad_W, grad_bias])):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    
    # Write weights
    out = struct.pack("<5i", MAGIC, d, FEAT_SIZE, C, MAX_N)
    out += W.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Hashed n-gram logistic regression: ./infer weights.bin in.rec N
 * Input: 16-byte context records. Output: predicted class byte (0..64).
 * Features: suffixes of length 1..MAX_N hashed with FNV-1a to FEAT_SIZE bins. */
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

static uint64_t fnv1a_64(const unsigned char *data, int len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (int i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x484E4731) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], feat_size = hd[2], C = hd[3], max_n = hd[4];
    if (d != 16 || C < 1 || C > 256 || max_n < 1 || max_n > 16 ||
        nw != 20 + 4L * ((long)feat_size * C + C)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *W = (const float *)(hd + 5);
    const float *bias = W + (long)feat_size * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int L = 1; L <= max_n; L++) {
            uint64_t h = fnv1a_64(xi + d - L, L);
            int idx = (int)(h & (feat_size - 1));
            const float *wrow = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += wrow[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```