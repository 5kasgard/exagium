```hypothesis
Family: label-ngram (n-gram backoff model). It captures exact sequential patterns in text by storing the most frequent next character for every observed suffix of the 16-byte context (lengths 16 down to 1). At inference it tries the longest suffix first, falling back to shorter ones, which is a classic strong baseline for character-level language modeling. It should beat label-linear (0.3375) because it models interactions between adjacent bytes that the per-position linear model misses, while remaining extremely fast (a few binary searches per item). If it fails to beat label-linear, it would indicate that the training set is too sparse for reliable n-gram counts and the linear model's smoothing via weight sharing generalizes better.
```
```python
import json, struct, sys
import numpy as np

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65

    # counts[L][suffix_bytes] = array of 65 counts
    counts = [None] + [{} for _ in range(d)]

    for ctx, tgt in zip(X, y):
        ctx_bytes = ctx.tobytes()
        for L in range(1, d + 1):
            suffix = ctx_bytes[-L:]
            arr = counts[L].get(suffix)
            if arr is None:
                arr = np.zeros(C, dtype=np.int32)
                counts[L][suffix] = arr
            arr[tgt] += 1

    # Build sorted tables: for each L, list of (suffix, best_class)
    tables = [None] * (d + 1)
    for L in range(1, d + 1):
        items = []
        for suffix, arr in counts[L].items():
            best = int(np.argmax(arr))          # tie-break by smallest class (argmax does this)
            items.append((suffix, best))
        items.sort(key=lambda x: x[0])          # deterministic lexicographic order
        tables[L] = items

    # Overall majority class for fallback
    majority = int(np.bincount(y).argmax())

    # Write weights.bin
    # Format: magic(4) version(1) max_L(1)  then for L=max_L..1: count(4) [suffix(L) class(1)]*count  finally majority(1)
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<I B B", 0x4E47524D, 1, d))   # 'NGRM'
        for L in range(d, 0, -1):
            items = tables[L]
            f.write(struct.pack("<I", len(items)))
            for suffix, cls in items:
                f.write(suffix)
                f.write(struct.pack("B", cls))
        f.write(struct.pack("B", majority))

if __name__ == "__main__":
    main()
```
```c
/* label-ngram inference: ./infer weights.bin in.rec N
 * Weights format: magic 'NGRM'(0x4E47524D), version(1), max_L(1),
 *   for L=max_L..1: uint32 n, then n records of (L bytes suffix, uint8 class),
 *   finally uint8 majority_class.
 * Input records: 16 bytes each. Output: one byte class per record. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

static unsigned char *slurp(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(3); }
    fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*len ? *len : 1);
    if (fread(buf, 1, *len, f) != (size_t)*len) { fprintf(stderr, "short read %s\n", path); exit(3); }
    fclose(f);
    return buf;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *out_len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = (uint32_t)buf[*pos] | (uint32_t)buf[*pos+1]<<8 | (uint32_t)buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *out_len = k;
    const unsigned char *rec = buf + *pos + 4;
    *pos += 4 + k;
    return rec;
}

static void put_rec(FILE *f, const unsigned char *rec, uint32_t k) {
    unsigned char hdr[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(hdr, 1, 4, f); fwrite(rec, 1, k, f);
}

typedef struct { int L, n; unsigned char *suffixes; unsigned char *classes; } NGramTable;

static int binsearch(const NGramTable *t, const unsigned char *key, unsigned char *out_class) {
    int lo = 0, hi = t->n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        const unsigned char *mid_key = t->suffixes + (long)mid * t->L;
        int cmp = memcmp(key, mid_key, t->L);
        if (cmp == 0) { *out_class = t->classes[mid]; return 1; }
        else if (cmp < 0) hi = mid - 1;
        else lo = mid + 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long wlen, xlen = 0; long N = atol(argv[3]);
    unsigned char *wbuf = slurp(argv[1], &wlen);
    unsigned char *xbuf = N ? slurp(argv[2], &xlen) : NULL;

    const unsigned char *p = wbuf;
    if (wlen < 6) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = p[0] | p[1]<<8 | p[2]<<16 | p[3]<<24;
    if (magic != 0x4E47524D) { fprintf(stderr, "bad magic\n"); return 3; }
    int version = p[4]; (void)version;
    int max_L = p[5];
    if (max_L < 1 || max_L > 16) { fprintf(stderr, "bad max_L\n"); return 3; }
    p += 6;

    NGramTable tables[17] = {0};   // index by L
    for (int L = max_L; L >= 1; L--) {
        if (p + 4 > wbuf + wlen) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint32_t n = p[0] | p[1]<<8 | p[2]<<16 | p[3]<<24; p += 4;
        tables[L].L = L; tables[L].n = n;
        if (n) {
            tables[L].suffixes = malloc((size_t)n * L);
            tables[L].classes  = malloc((size_t)n);
            if (!tables[L].suffixes || !tables[L].classes) { fprintf(stderr, "oom\n"); return 3; }
            if (p + (long)n * L > wbuf + wlen) { fprintf(stderr, "truncated weights\n"); return 3; }
            memcpy(tables[L].suffixes, p, (size_t)n * L); p += (long)n * L;
            if (p + (long)n > wbuf + wlen) { fprintf(stderr, "truncated weights\n"); return 3; }
            memcpy(tables[L].classes, p, (size_t)n); p += n;
        }
    }
    unsigned char majority = 0;
    if (p < wbuf + wlen) majority = *p;

    unsigned char out_class;
    long pos = 0;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *rec = next_rec(xbuf, xlen, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        int found = 0;
        for (int L = max_L; L >= 1; L--) {
            if (tables[L].n == 0) continue;
            const unsigned char *suffix = rec + 16 - L;
            if (binsearch(&tables[L], suffix, &out_class)) { found = 1; break; }
        }
        if (!found) out_class = majority;
        put_rec(stdout, &out_class, 1);
    }
    return 0;
}
```