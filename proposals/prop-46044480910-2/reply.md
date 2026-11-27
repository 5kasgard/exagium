```hypothesis
Family: label-hashngram (hashed n-gram linear model). Why: character-level LM benefits from local n-gram interactions (1-4 grams); hashing captures these in a fixed-size linear model, faster than trees and more expressive than per-position one-hot. Failure: if dev accuracy ≤0.34 (label-linear) or inference cost >2× label-linear, the gain isn't worth it.
```

```python
"""Hashed n-gram linear model for next-character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Features: all n-grams up to length 4 from 16-byte context, hashed to 8192 bins.
Linear classifier with Adam (lr=3e-3, L2=1e-4), batch 64, up to 25 epochs.
Weights format: magic 0x484E4731, d, H, C, D, NMAX, seed; float32 W[D][C], b[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters
D = 8192          # hash table size
NMAX = 4          # max n-gram length
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
HASH_SEED = 123456789

MAGIC = 0x484E4731  # "HNG1"


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    class_chars = card["class_chars"]
    vocab_size = len(class_chars)  # 65
    char_to_idx = {ord(c): i for i, c in enumerate(class_chars)}

    X_raw = records(sys.argv[2])   # list of 16-byte contexts
    y_raw = records(sys.argv[3])   # list of 1-byte labels

    n = len(X_raw)
    # Convert to indices (0..64), unknown -> 0
    X = np.zeros((n, 16), dtype=np.int32)
    for i, seq in enumerate(X_raw):
        for j, b in enumerate(seq):
            X[i, j] = char_to_idx.get(b, 0)
    y = np.array([char_to_idx.get(seq[0], 0) for seq in y_raw], dtype=np.int64)

    # Precompute feature indices for each example: 58 n-grams per example
    # n-grams: for each pos 0..15, lengths 1..min(pos+1, NMAX)
    feats_per_ex = 0
    for pos in range(16):
        feats_per_ex += min(pos + 1, NMAX)  # 58
    features = np.zeros((n, feats_per_ex), dtype=np.int32)

    def hash_ngram(ngram, pos, length):
        h = HASH_SEED
        for idx in ngram:
            h = (h * 1315423911 + idx) & 0xFFFFFFFF
        h = (h * 1315423911 + pos) & 0xFFFFFFFF
        h = (h * 1315423911 + length) & 0xFFFFFFFF
        return h % D

    for i in range(n):
        fidx = 0
        for pos in range(16):
            for length in range(1, NMAX + 1):
                if pos >= length - 1:
                    ngram = tuple(X[i, pos - length + 1:pos + 1])
                    features[i, fidx] = hash_ngram(ngram, pos, length)
                    fidx += 1

    # Model parameters
    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (D, vocab_size)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(vocab_size, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, y_idx):
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(y_idx)), y_idx] -= 1.0
        return (probs / len(y_idx)).astype(np.float32)

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            batch_feats = features[idx]          # (batch, 58)
            yy = y[idx]
            batch_size = len(idx)

            # Forward: logits = sum W[feat] + b
            logits = W[batch_feats].sum(axis=1) + b  # (batch, C)

            grad = softmax_grad(logits, yy)          # (batch, C)

            # Gradients
            dW = np.zeros_like(W)
            grad_expanded = np.repeat(grad[:, np.newaxis, :], feats_per_ex, axis=1)  # (batch, 58, C)
            np.add.at(dW, batch_feats.ravel(), grad_expanded.reshape(-1, vocab_size))
            dW += L2 * W
            db = grad.sum(axis=0) + L2 * b

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, [dW, db])):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights
    out = struct.pack("<7i", MAGIC, 16, 0, vocab_size, D, NMAX, HASH_SEED)
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Hashed n-gram linear model inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte (class index 0..64). */
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

static uint32_t hash_ngram(const int8_t *ngram, int len, int pos, int length,
                           uint32_t seed, int D) {
    uint32_t h = seed;
    for (int i = 0; i < len; i++) {
        h = h * 1315423911u + (uint32_t)ngram[i];
    }
    h = h * 1315423911u + (uint32_t)pos;
    h = h * 1315423911u + (uint32_t)length;
    return h % (uint32_t)D;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 28 || hd[0] != 0x484E4731) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], D = hd[4], NMAX = hd[5];
    uint32_t hash_seed = (uint32_t)hd[6];
    if (d != 16 || H != 0 || C != 65 || D != 8192 || NMAX != 4) {
        fprintf(stderr, "bad weights header: d=%d H=%d C=%d D=%d NMAX=%d\n", d, H, C, D, NMAX); return 3;
    }
    long expected = 28L + 4L * D * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *W = (const float *)(hd + 7);
    const float *b = W + (long)D * C;

    // Build byte->index mapping (matches train.py)
    int8_t char_to_idx[256];
    for (int i = 0; i < 256; i++) char_to_idx[i] = 0;
    const char *chars = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    for (int i = 0; chars[i]; i++) char_to_idx[(unsigned char)chars[i]] = (int8_t)i;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[65];
    int8_t idx_buf[16];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        for (int j = 0; j < 16; j++) idx_buf[j] = char_to_idx[xi[j]];

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int pos_i = 0; pos_i < 16; pos_i++) {
            for (int len = 1; len <= NMAX; len++) {
                if (pos_i >= len - 1) {
                    uint32_t h = hash_ngram(&idx_buf[pos_i - len + 1], len, pos_i, len, hash_seed, D);
                    const float *w = W + (long)h * C;
                    for (int c = 0; c < C; c++) scores[c] += w[c];
                }
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```