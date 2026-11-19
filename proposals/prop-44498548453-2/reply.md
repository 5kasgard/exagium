```hypothesis
Family: label-hashngram (hashed n-gram linear model). Uses unigram (position, byte) and bigram (position, byte, next_byte) features hashed to a fixed table (65536 entries), with a linear output layer per class. Captures local sequential structure critical for character prediction while staying purely linear (fast inference). Expected to beat label-linear (0.3375) by modeling bigram statistics; if dev accuracy < 0.35, the hashing capacity or feature set is insufficient.
```
```python
"""Hashed n-gram linear model for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Features: 16 unigrams (pos, byte) + 15 bigrams (pos, byte, next_byte) = 31 features/item.
Hashed to table_size=65536 with a fast multiplicative hash. Linear output (table_size x classes).
Adam lr=0.01, L2=1e-5, batch=64, epochs=min(50, ceil(600000/n)). Deterministic seed=1."""
import json, math, struct, sys

import numpy as np

TABLE_SIZE = 65536
MASK = TABLE_SIZE - 1
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-5, 64, 600000, 50
MULT = 1315423911  # 32-bit multiplicative constant


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def hash_features(x, d):
    """x: (n, d) uint8. Returns (n, 31) int32 feature indices."""
    n = x.shape[0]
    feats = np.empty((n, 31), dtype=np.uint32)
    # unigrams: positions 0..15
    for j in range(d):
        h = (j * MULT + x[:, j].astype(np.uint32)) & 0xFFFFFFFF
        feats[:, j] = h & MASK
    # bigrams: positions 0..14
    for j in range(d - 1):
        h = (j * MULT + x[:, j].astype(np.uint32)) & 0xFFFFFFFF
        h = (h * MULT + x[:, j + 1].astype(np.uint32)) & 0xFFFFFFFF
        feats[:, 16 + j] = h & MASK
    return feats


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # Precompute feature indices for all training data
    feat_idx = hash_features(X, d)  # (n, 31)
    flat_idx = feat_idx.ravel()     # (n*31,)

    rng = np.random.default_rng(SEED)
    W = rng.normal(0, 0.01, (TABLE_SIZE, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            idx_flat = (idx[:, None] * 31 + np.arange(31)).ravel()
            yy = y[idx]
            bs = len(yy)

            # Forward: scores = b + sum_{features} W[feat]
            scores = b + W[flat_idx[idx_flat]].reshape(bs, 31, C).sum(axis=1)
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores, dtype=np.float32)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yy] -= 1.0
            grad = (probs / bs).astype(np.float32)  # (bs, C)

            # Gradients: dW[feat] += grad[sample] for each active feature
            grad_rep = np.repeat(grad, 31, axis=0)  # (bs*31, C)
            np.add.at(W, flat_idx[idx_flat], grad_rep)
            db = grad.sum(axis=0)

            # L2 on W only (not bias)
            grad_W = W[flat_idx[idx_flat]].reshape(bs, 31, C) * 0 + L2 * W[flat_idx[idx_flat]].reshape(bs, 31, C)
            # Actually simpler: add L2 to the accumulated gradient via np.add.at
            np.add.at(W, flat_idx[idx_flat], L2 * W[flat_idx[idx_flat]].reshape(bs, 31, C).reshape(-1, C))
            # But W is being updated in place, so use current W for L2:
            # We'll apply L2 as weight decay on the parameter directly in the update step (Adam with weight decay).
            # Instead, just add L2*W to the gradient buffer before Adam step.
            # Let's do proper AdamW: decoupled weight decay.
            # For simplicity, use classic L2 on gradient:
            gW = np.zeros_like(W)
            np.add.at(gW, flat_idx[idx_flat], grad_rep)
            gW += L2 * W
            gb = db  # no L2 on bias

            t += 1
            # Adam update
            mW = 0.9 * mW + 0.1 * gW
            vW = 0.999 * vW + 0.001 * gW * gW
            mW_hat = mW / (1 - 0.9 ** t)
            vW_hat = vW / (1 - 0.999 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + 1e-8)

            mb = 0.9 * mb + 0.1 * gb
            vb = 0.999 * vb + 0.001 * gb * gb
            mb_hat = mb / (1 - 0.9 ** t)
            vb_hat = vb / (1 - 0.999 ** t)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + 1e-8)

    # Save weights: magic, d, H=0, C, table_size; then W (table_size x C), b (C)
    magic = 0x484E474C  # 'LGNH'
    hdr = struct.pack("<5i", magic, d, 0, C, TABLE_SIZE)
    out = hdr + W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Hashed n-gram linear model: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Features: 16 unigrams + 15 bigrams hashed to 65536-table.
 * Output: 1 byte class (argmax). Weights: magic, d, H=0, C, table_size; float32 W[table_size][C], b[C]. */
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

static inline uint32_t hash1(uint32_t pos, uint32_t b1) {
    uint32_t h = pos * 1315423911u + b1;
    return h & 0xFFFFu;
}
static inline uint32_t hash2(uint32_t pos, uint32_t b1, uint32_t b2) {
    uint32_t h = pos * 1315423911u + b1;
    h = h * 1315423911u + b2;
    return h & 0xFFFFu;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x484E474C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], table_size = hd[4];
    if (H != 0 || C < 1 || C > 256 || d != 16 || table_size != 65536) {
        fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 20L + 4L * table_size * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W = (const float *)(hd + 5);
    const float *b = W + (long)table_size * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(scores, b, sizeof(float) * C);

        for (int j = 0; j < d; j++) {
            uint32_t idx = hash1(j, xi[j]);
            const float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        for (int j = 0; j < d - 1; j++) {
            uint32_t idx = hash2(j, xi[j], xi[j + 1]);
            const float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```