```hypothesis
Propose label-ngram-linear: a linear model on hashed n-gram features (unigrams, bigrams, trigrams) from the 16-byte context.
It captures local sequential dependencies that independent byte features miss, while feature hashing keeps the model size small and inference fast.
If dev accuracy does not exceed label-linear's 0.3375, the n-gram features do not help for this task.
```

```python
"""label-ngram-linear: linear model on hashed n-gram features (unigrams, bigrams, trigrams).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: magic 0x4E474C31, M, C, d; float32 W[M][C] (row-major), float32 b[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters
M = 16384          # hash table size (power of two)
LR = 0.01          # learning rate
L2 = 1e-4          # L2 regularization
BATCH = 64         # minibatch size
MAX_EPOCHS = 100   # maximum epochs
SAMPLES = 600000   # max samples for epoch calculation
SEED = 1           # fixed seed

# FNV-1a 32-bit hash
FNV_OFFSET = 0x811c9dc5
FNV_PRIME = 0x01000193
HASH_SEED = 0x12345678

def fnv1a_bytes(data: bytes) -> int:
    h = HASH_SEED
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFF
    return h

def record_hashes(seq: bytes) -> list:
    """Return list of 45 hash indices for a 16-byte sequence."""
    hashes = []
    # unigrams: (pos, byte)
    for p in range(16):
        h = fnv1a_bytes(bytes([p, seq[p]]))
        hashes.append(h & (M - 1))
    # bigrams: (pos, byte1, byte2)
    for p in range(15):
        h = fnv1a_bytes(bytes([p, seq[p], seq[p+1]]))
        hashes.append(h & (M - 1))
    # trigrams: (pos, byte1, byte2, byte3)
    for p in range(14):
        h = fnv1a_bytes(bytes([p, seq[p], seq[p+1], seq[p+2]]))
        hashes.append(h & (M - 1))
    return hashes

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    C = card["classes"]
    d = card["input"]["bytes"]  # 16

    # Precompute hashes for all training examples
    X_hashes = [record_hashes(rec) for rec in X_recs]
    y = np.array([r[0] for r in y_recs], dtype=np.int64)
    n = len(y)

    # Initialize weights
    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (M, C)) * math.sqrt(6.0 / M)).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    idx = np.arange(n)

    for epoch in range(max_epochs):
        rng.shuffle(idx)
        for s in range(0, n, BATCH):
            batch_idx = idx[s:s + BATCH]
            bs = len(batch_idx)
            # Accumulate gradients
            gW = np.zeros_like(W)
            gb = np.zeros_like(b)
            for i in batch_idx:
                hashes = X_hashes[i]
                target = y[i]
                # Logits: b + sum W[h]
                logits = b.copy()
                for h in hashes:
                    logits += W[h]
                # Softmax
                logits -= logits.max()
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum()
                # Gradient: probs - one_hot
                probs[target] -= 1.0
                # Accumulate
                for h in hashes:
                    gW[h] += probs
                gb += probs
            # Average over batch
            gW /= bs
            gb /= bs
            # L2 regularization
            gW += L2 * W
            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * (gW * gW)
            mb = beta1 * mb + (1 - beta1) * gb
            vb = beta2 * vb + (1 - beta2) * (gb * gb)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            mb_hat = mb / (1 - beta1 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + eps)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + eps)

    # Write weights.bin
    magic = 0x4E474C31  # "NGL1"
    header = struct.pack("<4I", magic, M, C, d)
    weight_bytes = W.astype("<f4").tobytes()
    bias_bytes = b.astype("<f4").tobytes()
    with open(sys.argv[4], "wb") as f:
        f.write(header)
        f.write(weight_bytes)
        f.write(bias_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-ngram-linear inference: ./infer weights.bin in.rec N
 * Input records: 16 bytes each. Output: one byte class index per record. */
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

/* FNV-1a 32-bit, same as train.py */
static uint32_t fnv1a(const unsigned char *data, int len) {
    uint32_t h = 0x12345678;
    for (int i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x01000193;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4E474C31) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t M = hd[1], C = hd[2], d = hd[3];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16 + (long)M * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(hd + 4);
    const float *b = (const float *)(W + (long)M * C);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    unsigned char tmp[4];  // for hashing input bytes

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        // Initialize scores with bias
        for (int c = 0; c < (int)C; c++) scores[c] = b[c];
        // Unigrams
        for (int p = 0; p < 16; p++) {
            tmp[0] = p; tmp[1] = xi[p];
            uint32_t h = fnv1a(tmp, 2) & (M - 1);
            const float *w = W + (long)h * C;
            for (int c = 0; c < (int)C; c++) scores[c] += w[c];
        }
        // Bigrams
        for (int p = 0; p < 15; p++) {
            tmp[0] = p; tmp[1] = xi[p]; tmp[2] = xi[p+1];
            uint32_t h = fnv1a(tmp, 3) & (M - 1);
            const float *w = W + (long)h * C;
            for (int c = 0; c < (int)C; c++) scores[c] += w[c];
        }
        // Trigrams
        for (int p = 0; p < 14; p++) {
            tmp[0] = p; tmp[1] = xi[p]; tmp[2] = xi[p+1]; tmp[3] = xi[p+2];
            uint32_t h = fnv1a(tmp, 4) & (M - 1);
            const float *w = W + (long)h * C;
            for (int c = 0; c < (int)C; c++) scores[c] += w[c];
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < (int)C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```