```hypothesis
Family: label-ngram. A variable-length n-gram model with backoff that stores the most frequent next character for every observed context suffix (lengths 1–16) in sorted arrays for binary search. It should beat label-linear because character-level language modeling benefits from exact context statistics; the linear model only learns a smoothed average. If it does not exceed 0.3375 accuracy, the training data is too sparse for n-gram counts to generalize better than a parametric model.
```

```python
import json, struct, sys

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
    X = records(sys.argv[2])
    y = [r[0] for r in records(sys.argv[3])]
    n = len(X)

    # global majority
    freq = [0] * C
    for c in y:
        freq[c] += 1
    global_maj = max(range(C), key=lambda c: (freq[c], -c))

    # tables for each length L=1..d
    tables = [[] for _ in range(d + 1)]  # index by L
    for L in range(1, d + 1):
        counts = {}
        for ctx, target in zip(X, y):
            key = ctx[-L:]
            if key not in counts:
                counts[key] = [0] * C
            counts[key][target] += 1
        # determine best class for each context
        for ctx_bytes, cnt in counts.items():
            best = max(range(C), key=lambda c: (cnt[c], -c))
            tables[L].append((ctx_bytes, best))
        # sort by context bytes for binary search
        tables[L].sort(key=lambda x: x[0])

    # write weights.bin
    magic = 0x4E47524D  # 'NGRM' little-endian
    header = struct.pack("<4I", magic, d, C, global_maj)
    # write N_L for each L
    nls = [len(tables[L]) for L in range(1, d + 1)]
    header += struct.pack("<" + "I" * d, *nls)
    body = bytearray()
    for L in range(1, d + 1):
        for ctx_bytes, best in tables[L]:
            body += ctx_bytes
            body.append(best)
    open(sys.argv[4], "wb").write(header + body)

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
    uint32_t k = buf[*pos] | buf[*pos + 1] << 8 | buf[*pos + 2] << 16 | (uint32_t)buf[*pos + 3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

/* binary search in a table of fixed-length entries.
 * table: pointer to first entry, each entry is L bytes context + 1 byte class.
 * n: number of entries. L: context length.
 * key: L bytes to search for.
 * returns class byte if found, else -1. */
static int bsearch_table(const unsigned char *table, uint32_t n, int L, const unsigned char *key) {
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        const unsigned char *entry = table + (uint64_t)mid * (L + 1);
        int cmp = memcmp(entry, key, L);
        if (cmp == 0) return entry[L];
        if (cmp < 0) lo = mid + 1;
        else hi = mid;
    }
    return -1;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = w[0] | w[1] << 8 | w[2] << 16 | (uint32_t)w[3] << 24;
    if (magic != 0x4E47524D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = w[4] | w[5] << 8 | w[6] << 16 | (uint32_t)w[7] << 24;
    int C = w[8] | w[9] << 8 | w[10] << 16 | (uint32_t)w[11] << 24;
    int global_maj = w[12] | w[13] << 8 | w[14] << 16 | (uint32_t)w[15] << 24;
    if (d != 16 || C != 65) { fprintf(stderr, "bad weights dims\n"); return 3; }

    uint32_t nL[17] = {0};
    const unsigned char *p = w + 16;
    for (int L = 1; L <= d; L++) {
        if (p + 4 > w + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        nL[L] = p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24;
        p += 4;
    }
    const unsigned char *tables[17] = {NULL};
    for (int L = 1; L <= d; L++) {
        tables[L] = p;
        uint64_t sz = (uint64_t)nL[L] * (L + 1);
        if (p + sz > w + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        p += sz;
    }
    if (p != w + nw) { fprintf(stderr, "weights size mismatch\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        unsigned char pred = global_maj;
        for (int L = d; L >= 1; L--) {
            if (nL[L] == 0) continue;
            int res = bsearch_table(tables[L], nL[L], L, xi + d - L);
            if (res >= 0) { pred = (unsigned char)res; break; }
        }
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```