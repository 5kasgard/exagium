```hypothesis
Family: label-ngram (hash-based n-gram with stupid backoff, orders 1–5).
Why: Next-character prediction is inherently local; n-grams model P(c|context) directly without
  the independence assumption of linear models. 400K training chars support orders 1–5 well.
  Hash tables (65K buckets) make it memory-efficient (~86 MB) and inference fast (≤5 hashes/item).
Failure: Dev accuracy ≤ 0.3375 (label-linear) shows hashing collisions or data sparsity hurt more
  than the sequential inductive bias helps.
```
```python
import json, struct, sys
import numpy as np

MAGIC = 0x4E47524D  # 'NGRM'
VERSION = 1
MAX_ORDER = 5
BUCKETS = 65536
CLASSES = 65

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert d == 16 and C == 65
    
    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    n = len(X_recs)
    assert len(y_recs) == n
    
    totals = np.zeros((MAX_ORDER, BUCKETS), dtype=np.uint32)
    counts = np.zeros((MAX_ORDER, BUCKETS, CLASSES), dtype=np.uint32)
    
    for i in range(n):
        ctx = X_recs[i]
        tgt = y_recs[i][0]
        for k in range(1, MAX_ORDER + 1):
            h = 2166136261
            for j in range(d - k, d):
                h ^= ctx[j]
                h = (h * 16777619) & 0xFFFFFFFF
            h &= (BUCKETS - 1)
            totals[k-1, h] += 1
            counts[k-1, h, tgt] += 1
    
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<5I", MAGIC, VERSION, MAX_ORDER, BUCKETS, CLASSES))
        for k in range(MAX_ORDER):
            f.write(totals[k].astype("<u4").tobytes())
            f.write(counts[k].astype("<u4").tobytes())

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
    uint32_t k = buf[*pos] | (buf[*pos + 1] << 8) | (buf[*pos + 2] << 16) | ((uint32_t)buf[*pos + 3] << 24);
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

static uint32_t hash_bytes(const unsigned char *buf, int len, uint32_t buckets_mask) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= buf[i];
        h *= 16777619u;
    }
    return h & buckets_mask;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t *hdr = (uint32_t *)wb;
    if (hdr[0] != 0x4E47524D) { fprintf(stderr, "bad weights: magic\n"); return 3; }
    uint32_t version = hdr[1];
    uint32_t max_order = hdr[2];
    uint32_t buckets = hdr[3];
    uint32_t classes = hdr[4];
    if (version != 1 || max_order != 5 || buckets != 65536 || classes != 65) {
        fprintf(stderr, "bad weights: params\n"); return 3; }
    uint32_t buckets_mask = buckets - 1;
    uint32_t min_count = 2;
    long expected = 20 + (long)max_order * (buckets + (long)buckets * classes) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights: size %ld != %ld\n", nw, expected); return 3; }
    uint32_t *data = (uint32_t *)(wb + 20);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        unsigned char best = 0;
        int found = 0;
        for (int order = max_order; order >= 1; order--) {
            uint32_t h = hash_bytes(xi + 16 - order, order, buckets_mask);
            uint32_t *order_data = data + (long)(order - 1) * (buckets + (long)buckets * classes);
            uint32_t *totals = order_data;
            uint32_t *counts = order_data + buckets;
            uint32_t total = totals[h];
            if (total >= min_count) {
                uint32_t *bucket_counts = counts + (long)h * classes;
                uint32_t best_count = bucket_counts[0];
                best = 0;
                for (uint32_t c = 1; c < classes; c++) {
                    if (bucket_counts[c] > best_count) {
                        best_count = bucket_counts[c];
                        best = (unsigned char)c;
                    }
                }
                found = 1;
                break;
            }
        }
        if (!found) {
            uint32_t h = hash_bytes(xi + 15, 1, buckets_mask);
            uint32_t *order_data = data;
            uint32_t *counts = order_data + buckets;
            uint32_t *bucket_counts = counts + (long)h * classes;
            uint32_t best_count = bucket_counts[0];
            best = 0;
            for (uint32_t c = 1; c < classes; c++) {
                if (bucket_counts[c] > best_count) {
                    best_count = bucket_counts[c];
                    best = (unsigned char)c;
                }
            }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```