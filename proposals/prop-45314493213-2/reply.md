```hypothesis
We propose "label-hashngram": a linear model over hashed n-gram features (unigrams, bigrams, trigrams) with a fixed-size weight matrix. This captures local character dependencies crucial for next-character prediction, while feature hashing keeps the model size and inference cost bounded. The model should beat label-linear (0.3375) because n-grams provide higher-order context. If dev accuracy does not exceed 0.34, the hypothesis fails.
```

```python
"""Hash n-gram linear model for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Features: hashed unigrams, bigrams, trigrams over 16-byte context.
Weight matrix: hash_size x C (float32), bias: C.
Exported weights: magic 0x484E4731, hash_size, C, ngram_max, byte_to_class[256], W[hash_size][C], b[C]."""
import json, struct, sys, math

import numpy as np

HASH_SIZE = 1 << 16          # 65536
NGRAM_MAX = 3                # unigrams, bigrams, trigrams
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-5, 64, 600000, 100
FNV_OFFSET = 2166136261
FNV_PRIME  = 16777619

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def fnv1a_bytes(arr):
    h = FNV_OFFSET
    for b in arr:
        h = (h ^ b) * FNV_PRIME
        h &= 0xFFFFFFFF
    return h

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    class_chars = card["class_chars"]   # list of 65 single-character strings

    # Build byte -> class index mapping (0..64), unknown -> 255
    byte_to_class = np.full(256, 255, dtype=np.uint8)
    for idx, ch in enumerate(class_chars):
        byte_to_class[ord(ch)] = idx

    # Read training data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([byte_to_class[r[0]] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    # Map input bytes to class indices (0..64)
    X = byte_to_class[X_bytes]          # (n, 16) uint8

    # Precompute n-gram hashes for each example
    # For each example, we will have a list of hash indices (with multiplicities)
    # We store as a list of arrays for memory efficiency.
    example_hashes = []
    for i in range(n):
        hashes = []
        seq = X[i]
        for start in range(d):
            # include start position as first byte to distinguish same n-gram at different positions
            h = FNV_OFFSET
            h = (h ^ start) * FNV_PRIME & 0xFFFFFFFF
            for length in range(1, NGRAM_MAX + 1):
                if start + length > d:
                    break
                h = (h ^ seq[start + length - 1]) * FNV_PRIME & 0xFFFFFFFF
                hashes.append(h % HASH_SIZE)
        example_hashes.append(np.array(hashes, dtype=np.int32))

    # Model parameters
    rng = np.random.default_rng(SEED)
    W = rng.normal(0, 0.01, size=(HASH_SIZE, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)

    # Softmax gradient helper
    def softmax_grad(scores, targets):
        # scores: (batch, C)
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        probs[np.arange(len(targets)), targets] -= 1.0
        return probs.astype(np.float32) / len(targets)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    indices = np.arange(n)
    for epoch in range(max_epochs):
        rng.shuffle(indices)
        for s in range(0, n, BATCH):
            batch_idx = indices[s:s + BATCH]
            batch_size = len(batch_idx)
            # Compute scores for batch
            scores = np.tile(b, (batch_size, 1))
            for bi, ex_idx in enumerate(batch_idx):
                hashes = example_hashes[ex_idx]
                scores[bi] += W[hashes].sum(axis=0)
            # Gradient
            g = softmax_grad(scores, y[batch_idx])   # (batch, C)
            # Accumulate gradients for W and b
            grad_W = np.zeros_like(W)
            grad_b = g.sum(axis=0)
            for bi, ex_idx in enumerate(batch_idx):
                hashes = example_hashes[ex_idx]
                np.add.at(grad_W, hashes, g[bi])
            grad_W += L2 * W
            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            mW = beta1 * mW + (1 - beta1) * grad_W
            vW = beta2 * vW + (1 - beta2) * grad_W * grad_W
            mb = beta1 * mb + (1 - beta1) * grad_b
            vb = beta2 * vb + (1 - beta2) * grad_b * grad_b
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            mb_hat = mb / (1 - beta1 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + 1e-8)

    # Write weights.bin
    magic = 0x484E4731  # "HNG1"
    header = struct.pack("<4i", magic, HASH_SIZE, C, NGRAM_MAX)
    mapping_bytes = byte_to_class.tobytes()
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + mapping_bytes + W_bytes + b_bytes)

if __name__ == "__main__":
    main()
```

```c
/* Hash n-gram linear model inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte class index (0..64).
 * Weights format: magic(0x484E4731), hash_size, C, ngram_max, byte_to_class[256],
 *                 W[hash_size][C] (float32 row-major), b[C] (float32). */
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

/* FNV-1a 32-bit */
static uint32_t fnv1a(const unsigned char *data, int len, uint32_t h) {
    for (int i = 0; i < len; i++) {
        h ^= data[i];
        h *= 16777619;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wbuf = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }

    int32_t *hdr = (int32_t *)wbuf;
    if (hdr[0] != 0x484E4731) { fprintf(stderr, "bad magic\n"); return 3; }
    int hash_size = hdr[1];
    int C = hdr[2];
    int ngram_max = hdr[3];
    if (C != 65 || hash_size != (1<<16) || ngram_max != 3) {
        fprintf(stderr, "unexpected header values\n"); return 3;
    }

    const unsigned char *byte_to_class = wbuf + 16;
    const float *W = (const float *)(byte_to_class + 256);
    const float *b = W + (long)hash_size * C;

    long expected = 16 + 256 + (long)hash_size * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        // Map 16 bytes to class indices (0..64)
        unsigned char seq[16];
        for (int j = 0; j < 16; j++) {
            unsigned char mapped = byte_to_class[xi[j]];
            if (mapped == 255) mapped = 0; // fallback, should not happen
            seq[j] = mapped;
        }

        // Initialize scores with bias
        for (int c = 0; c < C; c++) scores[c] = b[c];

        // Unigrams, bigrams, trigrams
        for (int start = 0; start < 16; start++) {
            uint32_t h = 2166136261;
            h = fnv1a((unsigned char *)&start, 1, h);
            for (int len = 1; len <= 3; len++) {
                if (start + len > 16) break;
                h = fnv1a(&seq[start + len - 1], 1, h);
                int idx = h % hash_size;
                const float *wrow = W + (long)idx * C;
                for (int c = 0; c < C; c++) scores[c] += wrow[c];
            }
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```