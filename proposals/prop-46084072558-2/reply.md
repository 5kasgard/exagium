```hypothesis
We propose a "label-ngram" family: a backoff n-gram language model with interpolation weights. It captures sequential dependencies in the text better than a linear model, and is computationally cheap at inference (only a few hash lookups per prediction). Since the training data is small (24k sequences), the n-gram counts fit in a small hash table. If it does not beat label-linear (0.3375), it would show that even simple n-gram models cannot capture the structure better than a linear model on this task.
```

```python
import json, math, struct, sys

import numpy as np

# deterministic, single-threaded, stdlib + numpy only
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def pack_ngram(ngram):
    """ngram: tuple of ints (class indices 0..64), length 1..max_order.
       Returns uint64 key: (length << 56) | (b0 << 48) | (b1 << 40) | ..."""
    L = len(ngram)
    key = (L & 0xFF) << 56
    for idx, b in enumerate(ngram):
        key |= (b & 0xFF) << (48 - 8 * idx)
    return key

def main():
    card = json.load(open(sys.argv[1]))
    C = card["classes"]          # 65
    d = card["input"]["bytes"]   # 16
    class_chars = card["class_chars"]
    # map raw byte -> class index
    byte_to_class = [-1] * 256
    for idx, ch in enumerate(class_chars):
        byte_to_class[ord(ch)] = idx

    # read training data
    X_raw = records(sys.argv[2])   # list of 16 raw bytes
    y_raw = records(sys.argv[3])   # list of 1 byte (class index)
    assert all(len(x) == d for x in X_raw)
    assert all(len(yy) == 1 for yy in y_raw)

    # convert contexts to class indices
    sequences = []
    for x, yy in zip(X_raw, y_raw):
        ctx = tuple(byte_to_class[b] for b in x)
        target = yy[0]
        sequences.append(ctx + (target,))

    n = len(sequences)
    split = int(0.8 * n)
    train_seqs = sequences[:split]
    tune_seqs  = sequences[split:]

    # hyperparameter grid
    best_acc = -1.0
    best_params = None

    for max_order in (3, 4, 5):
        for alpha in (0.1, 0.5, 1.0):
            # equal interpolation weights
            lambdas = [1.0 / max_order] * max_order

            # count n-grams on train_seqs
            counts = {}
            total_chars = 0
            for seq in train_seqs:
                L = len(seq)
                for i in range(L):
                    for k in range(1, max_order + 1):
                        if i + k <= L:
                            ngram = seq[i:i+k]
                            key = pack_ngram(ngram)
                            counts[key] = counts.get(key, 0) + 1
                            if k == 1:
                                total_chars += 1

            # evaluate on tune_seqs
            correct = 0
            for seq in tune_seqs:
                ctx = seq[:d]
                target = seq[d]
                scores = np.zeros(C, dtype=np.float32)
                for order in range(1, max_order + 1):
                    lam = lambdas[order - 1]
                    if order == 1:
                        count_ctx = total_chars
                    else:
                        suffix = ctx[-(order-1):]
                        ctx_key = pack_ngram(suffix)
                        count_ctx = counts.get(ctx_key, 0)
                        if count_ctx == 0:
                            continue
                    for c in range(C):
                        if order == 1:
                            ngram_key = pack_ngram((c,))
                        else:
                            ngram_key = pack_ngram(suffix + (c,))
                        count_ngram = counts.get(ngram_key, 0)
                        prob = (count_ngram + alpha) / (count_ctx + alpha * C)
                        scores[c] += lam * prob
                pred = int(np.argmax(scores))
                if pred == target:
                    correct += 1
            acc = correct / len(tune_seqs)
            if acc > best_acc:
                best_acc = acc
                best_params = (max_order, alpha, lambdas)

    # retrain on full data with best params
    max_order, alpha, lambdas = best_params
    counts = {}
    total_chars = 0
    for seq in sequences:
        L = len(seq)
        for i in range(L):
            for k in range(1, max_order + 1):
                if i + k <= L:
                    ngram = seq[i:i+k]
                    key = pack_ngram(ngram)
                    counts[key] = counts.get(key, 0) + 1
                    if k == 1:
                        total_chars += 1

    # prepare entries for orders >=2
    entries = []
    for key, cnt in counts.items():
        L = key >> 56
        if L >= 2:
            entries.append((key, cnt))
    entries.sort(key=lambda x: x[0])

    # write weights.bin
    # header: magic(u32), max_order(u32), alpha(f32), lambdas[max_order](f32),
    #         total_chars(u64), num_entries(u32), one_gram_counts[65](u32),
    #         then entries: key(u64), count(u32)
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<II", 0x4E47524D, max_order))   # 'NGRM'
        f.write(struct.pack("<f", alpha))
        for lam in lambdas:
            f.write(struct.pack("<f", lam))
        f.write(struct.pack("<Q", total_chars))
        f.write(struct.pack("<I", len(entries)))
        # one-gram counts in class order 0..64
        one_gram = [counts.get(pack_ngram((c,)), 0) for c in range(C)]
        f.write(struct.pack("<" + "I"*C, *one_gram))
        for key, cnt in entries:
            f.write(struct.pack("<QI", key, cnt))

if __name__ == "__main__":
    main()
```

```c
/* label-ngram inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes (ASCII). Output record: 1 byte (class index 0..64). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

/* mapping from raw ASCII byte to class index (0..64), 255 = unknown */
static const uint8_t byte_to_class[256] = {
    ['\n'] = 0, [' '] = 1, ['!'] = 2, ['$'] = 3, ['&'] = 4, ['\''] = 5,
    [','] = 6, ['-'] = 7, ['.'] = 8, ['3'] = 9, [':'] = 10, [';'] = 11,
    ['?'] = 12, ['A'] = 13, ['B'] = 14, ['C'] = 15, ['D'] = 16, ['E'] = 17,
    ['F'] = 18, ['G'] = 19, ['H'] = 20, ['I'] = 21, ['J'] = 22, ['K'] = 23,
    ['L'] = 24, ['M'] = 25, ['N'] = 26, ['O'] = 27, ['P'] = 28, ['Q'] = 29,
    ['R'] = 30, ['S'] = 31, ['T'] = 32, ['U'] = 33, ['V'] = 34, ['W'] = 35,
    ['X'] = 36, ['Y'] = 37, ['Z'] = 38, ['a'] = 39, ['b'] = 40, ['c'] = 41,
    ['d'] = 42, ['e'] = 43, ['f'] = 44, ['g'] = 45, ['h'] = 46, ['i'] = 47,
    ['j'] = 48, ['k'] = 49, ['l'] = 50, ['m'] = 51, ['n'] = 52, ['o'] = 53,
    ['p'] = 54, ['q'] = 55, ['r'] = 56, ['s'] = 57, ['t'] = 58, ['u'] = 59,
    ['v'] = 60, ['w'] = 61, ['x'] = 62, ['y'] = 63, ['z'] = 64
};

static uint64_t pack_ngram(const uint8_t *ngram, int len) {
    uint64_t key = ((uint64_t)len & 0xFF) << 56;
    for (int i = 0; i < len; i++) {
        key |= ((uint64_t)ngram[i] & 0xFF) << (48 - 8 * i);
    }
    return key;
}

/* binary search in sorted entries array */
static uint32_t find_count(const uint64_t *keys, const uint32_t *counts, uint32_t n, uint64_t key) {
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        if (keys[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    if (lo < n && keys[lo] == key) return counts[lo];
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    const unsigned char *p = w;
    if (nw < 4) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = p[0] | p[1]<<8 | p[2]<<16 | p[3]<<24; p += 4;
    if (magic != 0x4E47524D) { fprintf(stderr, "bad magic\n"); return 3; }
    if (nw < (long)(p - w) + 4) { fprintf(stderr, "truncated weights\n"); return 3; }
    uint32_t max_order = p[0] | p[1]<<8 | p[2]<<16 | p[3]<<24; p += 4;
    if (max_order == 0 || max_order > 16) { fprintf(stderr, "bad max_order\n"); return 3; }
    if (nw < (long)(p - w) + 4) { fprintf(stderr, "truncated weights\n"); return 3; }
    float alpha; memcpy(&alpha, p, 4); p += 4;
    float *lambdas = (float*)malloc(max_order * sizeof(float));
    if (nw < (long)(p - w) + 4*max_order) { fprintf(stderr, "truncated weights\n"); return 3; }
    for (uint32_t i = 0; i < max_order; i++) { memcpy(&lambdas[i], p, 4); p += 4; }
    uint64_t total_chars; memcpy(&total_chars, p, 8); p += 8;
    if (nw < (long)(p - w) + 4) { fprintf(stderr, "truncated weights\n"); return 3; }
    uint32_t num_entries = p[0] | p[1]<<8 | p[2]<<16 | p[3]<<24; p += 4;
    const uint32_t C = 65;
    if (nw < (long)(p - w) + 4*C) { fprintf(stderr, "truncated weights\n"); return 3; }
    uint32_t *one_gram = (uint32_t*)malloc(C * sizeof(uint32_t));
    for (uint32_t i = 0; i < C; i++) {
        one_gram[i] = p[0] | p[1]<<8 | p[2]<<16 | p[3]<<24; p += 4;
    }
    uint64_t *keys = (uint64_t*)malloc(num_entries * sizeof(uint64_t));
    uint32_t *counts = (uint32_t*)malloc(num_entries * sizeof(uint32_t));
    if (nw < (long)(p - w) + 12*num_entries) { fprintf(stderr, "truncated weights\n"); return 3; }
    for (uint32_t i = 0; i < num_entries; i++) {
        memcpy(&keys[i], p, 8); p += 8;
        counts[i] = p[0] | p[1]<<8 | p[2]<<16 | p[3]<<24; p += 4;
    }
    if ((long)(p - w) != nw) { fprintf(stderr, "weights size mismatch\n"); return 3; }

    float *scores = (float*)malloc(C * sizeof(float));
    uint8_t ngram_buf[16];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        /* convert raw bytes to class indices */
        uint8_t ctx[16];
        for (int j = 0; j < 16; j++) {
            uint8_t c = byte_to_class[xi[j]];
            if (c == 255) { fprintf(stderr, "unknown byte %d at pos %d\n", xi[j], j); return 3; }
            ctx[j] = c;
        }
        memset(scores, 0, C * sizeof(float));
        for (uint32_t order = 1; order <= max_order; order++) {
            float lam = lambdas[order - 1];
            uint32_t count_ctx;
            if (order == 1) {
                count_ctx = (uint32_t)total_chars;
            } else {
                int start = 16 - (order - 1);
                for (int j = 0; j < order - 1; j++) ngram_buf[j] = ctx[start + j];
                uint64_t ctx_key = pack_ngram(ngram_buf, order - 1);
                count_ctx = find_count(keys, counts, num_entries, ctx_key);
                if (count_ctx == 0) continue;
            }
            for (uint32_t c = 0; c < C; c++) {
                uint32_t count_ngram;
                if (order == 1) {
                    count_ngram = one_gram[c];
                } else {
                    ngram_buf[order - 1] = c;
                    uint64_t ngram_key = pack_ngram(ngram_buf, order);
                    count_ngram = find_count(keys, counts, num_entries, ngram_key);
                }
                float prob = (count_ngram + alpha) / (count_ctx + alpha * C);
                scores[c] += lam * prob;
            }
        }
        uint8_t best = 0;
        for (uint32_t c = 1; c < C; c++) if (scores[c] > scores[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```