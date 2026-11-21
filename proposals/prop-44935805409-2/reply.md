```hypothesis
Family: label-hashngram — linear model on hashed n-gram features (n=1,2,3) over the 16-byte context.
Why: Character-level language modeling benefits from local interactions (bigrams/trigrams) that a unigram linear model misses. Feature hashing (D=65536) compresses the vast n-gram space into a fixed-size weight matrix, keeping inference fast (O(features) per item) and training simple (Adam on sparse features). Deterministic custom hash ensures reproducibility.
Failure signal: If dev accuracy ≤ 0.3375 (label-linear) or inference cost ≥ 10k Ir/item, the hashing collisions or feature set hurt more than help.
```
```python
"""Hashed n-gram linear model for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Features: unigrams (pos, byte), bigrams (pos, byte, byte), trigrams (pos, byte, byte, byte) for all valid positions.
Hashed to D=65536 buckets with a deterministic polynomial hash. Multi-class logistic regression with Adam.
Exported weights: int32 magic 0x484E4731, D, C, max_n; float32 bias[C]; float32 W[D][C] (row-major)."""
import json, math, struct, sys

import numpy as np

D = 65536          # hash table size (power of two)
MAX_N = 3          # max n-gram order
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def hash_ngram(n, pos, bytes_seq, D_mask):
    """Deterministic polynomial hash for (n, pos, byte sequence)."""
    h = (n * 1315423911) ^ (pos * 2654435761)
    for b in bytes_seq:
        h = ((h * 31) ^ b) & D_mask
    return h


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X_recs = records(sys.argv[2])       # list of 16-byte contexts
    y_recs = records(sys.argv[3])       # list of 1-byte targets
    n = len(X_recs)
    X = np.frombuffer(b"".join(X_recs), dtype=np.uint8).reshape(n, d)
    y = np.array([r[0] for r in y_recs], dtype=np.int64)

    # Precompute feature hashes for all positions and n-gram orders
    D_mask = D - 1
    feat_hashes = []  # list of (sample_idx, hash) for each feature occurrence
    # We'll build a CSR-like structure: for each sample, list of hash indices
    sample_hashes = [[] for _ in range(n)]
    for i in range(n):
        ctx = X[i]
        for order in range(1, MAX_N + 1):
            for pos in range(d - order + 1):
                h = hash_ngram(order, pos, ctx[pos:pos+order], D_mask)
                sample_hashes[i].append(h)

    # Convert to dense index arrays for vectorized minibatch access
    max_feats = max(len(hs) for hs in sample_hashes)
    H_idx = np.full((n, max_feats), -1, dtype=np.int32)
    H_mask = np.zeros((n, max_feats), dtype=np.float32)
    for i, hs in enumerate(sample_hashes):
        H_idx[i, :len(hs)] = hs
        H_mask[i, :len(hs)] = 1.0

    rng = np.random.default_rng(SEED)
    W = np.zeros((D, C), dtype=np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            batch_sz = len(idx)
            h_idx = H_idx[idx]          # (B, F)
            h_msk = H_mask[idx]         # (B, F)
            yy = y[idx]

            # Accumulate scores: bias + sum_{f} W[h_idx, :]
            scores = b[None, :] + np.sum(W[h_idx] * h_msk[:, :, None], axis=1)  # (B, C)
            # Softmax + gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(batch_sz), yy] -= 1.0
            grad_scores = (probs / batch_sz).astype(np.float32)   # (B, C)

            # Gradients: W[h] += grad_scores[b], b += grad_scores[b]
            # Use scatter-add via numpy's advanced indexing (dense for simplicity)
            gW = np.zeros_like(W)
            np.add.at(gW, h_idx.reshape(-1), (grad_scores[:, None, :] * h_msk[:, :, None]).reshape(-1, C))
            gb = grad_scores.sum(axis=0)

            grads = [gW + L2 * W, gb + L2 * b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x484E4731  # "HNG1"
    out = struct.pack("<4i", magic, D, C, MAX_N)
    out += b.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Hashed n-gram linear model inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format: int32 magic(0x484E4731), D, C, max_n; float32 bias[C]; float32 W[D][C] row-major. */
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

/* Deterministic polynomial hash matching train.py */
static uint32_t hash_ngram(int n, int pos, const unsigned char *bytes, int order, uint32_t mask) {
    uint32_t h = (n * 1315423911u) ^ ((uint32_t)pos * 2654435761u);
    for (int i = 0; i < order; i++) {
        h = ((h * 31u) ^ bytes[i]) & mask;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x484E4731) { fprintf(stderr, "bad weights\n"); return 3; }
    int D = hd[1], C = hd[2], max_n = hd[3];
    if (C < 1 || C > 256 || D != 65536 || max_n != 3 ||
        nw != 16 + 4L * (C + (long)D * C)) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *bias = (const float *)(hd + 4);
    const float *W = bias + C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    uint32_t mask = D - 1;
    const int d = 16;  // fixed by task

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(scores, bias, C * sizeof(float));

        for (int order = 1; order <= max_n; order++) {
            for (int p = 0; p <= d - order; p++) {
                uint32_t h = hash_ngram(3, p, xi + p, order, mask);  // n=3 matches train.py's n=order
                const float *wrow = W + (long)h * C;
                for (int c = 0; c < C; c++) scores[c] += wrow[c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```