```hypothesis
The label-ngram family uses a variable-order n-gram model (orders 1–5) with longest-match backoff, storing only the most frequent next character for each observed n-gram. Character-level language modeling has strong short-range dependencies, so a 5-gram should capture the bulk of predictive power and beat the linear model's 0.3375 dev accuracy. If it fails to beat linear, it indicates that the 16-byte context requires longer-range or non-local patterns that n-grams cannot capture, or that the training set is too sparse for reliable n-gram estimates.
```
```python
import json, struct, sys
from collections import defaultdict

MAX_ORDER = 5

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X = records(sys.argv[2])            # list of 16-byte contexts
    y = [r[0] for r in records(sys.argv[3])]  # list of target bytes (0..64)

    # count[order][ngram_bytes] -> array of 65 counts
    count = [defaultdict(lambda: [0]*C) for _ in range(MAX_ORDER+1)]  # 1..MAX_ORDER
    for ctx, tgt in zip(X, y):
        for order in range(1, MAX_ORDER+1):
            ngram = ctx[-order:]                     # last 'order' bytes
            count[order][ngram][tgt] += 1

    # best_class[order][ngram_bytes] -> best class (uint8)
    best_class = [{} for _ in range(MAX_ORDER+1)]
    for order in range(1, MAX_ORDER+1):
        for ngram, cnts in count[order].items():
            best = max(range(C), key=lambda c: (cnts[c], -c))  # tie: smallest class
            best_class[order][ngram] = best

    # global majority class
    global_cnt = [0]*C
    for tgt in y:
        global_cnt[tgt] += 1
    global_majority = max(range(C), key=lambda c: (global_cnt[c], -c))

    # write weights.bin
    # header: magic(4), MAX_ORDER(1), global_majority(1), reserved(2)
    # for each order: num_entries(4), then entries: ngram_bytes (order bytes) + best_class (1 byte)
    # entries sorted by ngram_bytes for binary search
    out = bytearray()
    out += struct.pack("<I", 0x4E47524D)  # 'NGRM'
    out += struct.pack("<B", MAX_ORDER)
    out += struct.pack("<B", global_majority)
    out += struct.pack("<H", 0)           # reserved
    for order in range(1, MAX_ORDER+1):
        items = sorted(best_class[order].items())  # sort by ngram bytes
        out += struct.pack("<I", len(items))
        for ngram, cls in items:
            out += ngram
            out += struct.pack("<B", cls)
    open(sys.argv[4], "wb").write(out)

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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

/* binary search in a sorted table of entries: each entry is (order bytes key + 1 byte class) */
static int find_class(const unsigned char *table, uint32_t n, int order, const unsigned char *key, unsigned char *cls) {
    int lo = 0, hi = (int)n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        const unsigned char *entry = table + mid * (order + 1);
        int cmp = memcmp(entry, key, order);
        if (cmp == 0) { *cls = entry[order]; return 1; }
        if (cmp < 0) lo = mid + 1; else hi = mid - 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 8) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = w[0] | w[1]<<8 | w[2]<<16 | (uint32_t)w[3]<<24;
    if (magic != 0x4E47524D) { fprintf(stderr, "bad magic\n"); return 3; }
    int MAX_ORDER = w[4];
    unsigned char global_majority = w[5];
    if (MAX_ORDER < 1 || MAX_ORDER > 16) { fprintf(stderr, "bad MAX_ORDER\n"); return 3; }
    const unsigned char *p = w + 8;  /* after header */
    const unsigned char *order_table[17] = {0};
    uint32_t order_count[17] = {0};
    int order_size[17] = {0};
    for (int order = 1; order <= MAX_ORDER; order++) {
        if (p + 4 > w + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint32_t cnt = p[0] | p[1]<<8 | p[2]<<16 | (uint32_t)p[3]<<24;
        p += 4;
        order_table[order] = p;
        order_count[order] = cnt;
        order_size[order] = order + 1;
        p += (long)cnt * order_size[order];
    }
    if (p != w + nw) { fprintf(stderr, "weights size mismatch\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char key[16];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        memcpy(key, xi, 16);
        unsigned char cls = global_majority;
        for (int order = MAX_ORDER; order >= 1; order--) {
            if (find_class(order_table[order], order_count[order], order, key + 16 - order, &cls))
                break;
        }
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```