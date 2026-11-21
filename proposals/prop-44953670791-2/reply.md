```hypothesis
Family: label-ngram (4-gram with Laplace smoothing and backoff).  
Why: Character-level language modeling benefits greatly from local n-gram statistics; a 4-gram captures frequent sequences like "the ", "ing", etc., which the linear model misses because it treats each position independently.  
Failure signal: If dev accuracy ≤ 0.3375 (label-linear), then local n-grams are insufficient and longer-range or more complex interactions are needed.
```

```python
import json, struct, sys
import numpy as np

MAX_N = 4
ALPHA = 1.0

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = records(sys.argv[2])
    y = [r[0] for r in records(sys.argv[3])]

    counts = [{} for _ in range(MAX_N+1)]
    unigram = np.zeros(C, dtype=np.uint32)

    for ctx, target in zip(X, y):
        unigram[target] += 1
        for n in range(1, MAX_N+1):
            key = ctx[-n:]
            key_int = int.from_bytes(key, 'little')
            if key_int not in counts[n]:
                counts[n][key_int] = np.zeros(C, dtype=np.uint32)
            counts[n][key_int][target] += 1

    tables = []
    for n in range(1, MAX_N+1):
        items = list(counts[n].items())
        size = 1
        while size < 2 * len(items):
            size <<= 1
        keys = np.full(size, 0xFFFFFFFF, dtype=np.uint32)
        totals = np.zeros(size, dtype=np.uint32)
        cnts = np.zeros((size, C), dtype=np.uint16)
        occupied = np.zeros(size, dtype=np.uint8)
        for key_int, cnt_arr in items:
            h = key_int & (size - 1)
            while occupied[h]:
                h = (h + 1) & (size - 1)
            keys[h] = key_int
            totals[h] = cnt_arr.sum()
            cnts[h] = cnt_arr.astype(np.uint16)
            occupied[h] = 1
        tables.append((size, keys, totals, cnts, occupied))

    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<I", 0x4E47524D))
        f.write(struct.pack("<BB", 1, MAX_N))
        for n in range(MAX_N):
            size, keys, totals, cnts, occupied = tables[n]
            f.write(struct.pack("<I", size))
            f.write(keys.tobytes())
            f.write(totals.tobytes())
            f.write(cnts.tobytes())
            f.write(occupied.tobytes())
        f.write(unigram.astype(np.uint32).tobytes())

if __name__ == "__main__":
    main()
```

```c
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

#define ALPHA 1.0f
#define C 65
#define MAX_N 4

typedef struct {
    uint32_t size;
    uint32_t *keys;
    uint32_t *totals;
    uint16_t *counts;
    uint8_t *occupied;
} HashTable;

static int lookup(const HashTable *ht, uint32_t key, uint32_t *total, const uint16_t **counts) {
    uint32_t mask = ht->size - 1;
    uint32_t h = key & mask;
    while (ht->occupied[h]) {
        if (ht->keys[h] == key) {
            *total = ht->totals[h];
            *counts = &ht->counts[h * C];
            return 1;
        }
        h = (h + 1) & mask;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *weights = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    uint32_t *p32 = (uint32_t *)weights;
    if (nw < 8 || p32[0] != 0x4E47524D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    uint8_t *p8 = (uint8_t *)(weights + 4);
    uint8_t version = p8[0];
    uint8_t max_n = p8[1];
    if (version != 1 || max_n != MAX_N) { fprintf(stderr, "bad version or max_n\n"); return 3; }

    HashTable tables[MAX_N];
    uint8_t *cur = weights + 6;
    for (int n = 0; n < MAX_N; n++) {
        if (cur + 4 > weights + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint32_t size = *(uint32_t *)cur; cur += 4;
        tables[n].size = size;
        tables[n].keys = (uint32_t *)cur; cur += size * 4;
        tables[n].totals = (uint32_t *)cur; cur += size * 4;
        tables[n].counts = (uint16_t *)cur; cur += size * C * 2;
        tables[n].occupied = cur; cur += size;
    }
    if (cur + C * 4 > weights + nw) { fprintf(stderr, "truncated weights unigram\n"); return 3; }
    uint32_t *unigram = (uint32_t *)cur; cur += C * 4;

    float unigram_prob[C];
    uint32_t unigram_total = 0;
    for (int c = 0; c < C; c++) unigram_total += unigram[c];
    for (int c = 0; c < C; c++) unigram_prob[c] = (unigram[c] + ALPHA) / (unigram_total + ALPHA * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        unsigned char best = 0;
        float best_prob = -1.0f;
        int found = 0;

        for (int n = MAX_N; n >= 1; n--) {
            uint32_t key = 0;
            for (int b = 0; b < n; b++) {
                key |= (uint32_t)xi[16 - n + b] << (8 * b);
            }
            uint32_t total;
            const uint16_t *counts;
            if (lookup(&tables[n-1], key, &total, &counts)) {
                float denom = total + ALPHA * C;
                for (int c = 0; c < C; c++) {
                    float prob = (counts[c] + ALPHA) / denom;
                    if (prob > best_prob) {
                        best_prob = prob;
                        best = c;
                    }
                }
                found = 1;
                break;
            }
        }
        if (!found) {
            for (int c = 0; c < C; c++) {
                if (unigram_prob[c] > best_prob) {
                    best_prob = unigram_prob[c];
                    best = c;
                }
            }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```