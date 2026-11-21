```hypothesis
label-ngram: variable-order Markov model with stupid backoff over suffixes of the input bytes.
Text is highly predictive from recent characters; this captures variable-length patterns with minimal
parameters and O(log n) lookup per order. If dev accuracy ≤ 0.3375 (label-linear), the suffix backoff
fails to beat fixed-context linear on this task.
```
```python
import json, struct, sys
from collections import defaultdict, Counter

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
    contexts = records(sys.argv[2])
    targets = [r[0] for r in records(sys.argv[3])]

    max_order = d
    tables = [defaultdict(Counter) for _ in range(max_order + 1)]

    for ctx, tgt in zip(contexts, targets):
        for order in range(1, max_order + 1):
            suffix = ctx[-order:]
            tables[order][suffix][tgt] += 1

    order_data = []
    for order in range(1, max_order + 1):
        items = []
        for suffix, cnt in tables[order].items():
            best_class = max(range(C), key=lambda c: cnt.get(c, 0) + 1)
            items.append((suffix, best_class))
        items.sort(key=lambda x: x[0])
        order_data.append(items)

    global_cnt = Counter(targets)
    global_best = max(range(C), key=lambda c: global_cnt.get(c, 0))

    MAGIC = 0x4E47524D
    out = bytearray()
    out += struct.pack("<IIIIB", MAGIC, d, C, max_order, global_best)

    for order in range(1, max_order + 1):
        items = order_data[order - 1]
        n = len(items)
        out += struct.pack("<I", n)
        for suffix, _ in items:
            out += suffix
        for _, cls in items:
            out.append(cls)

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
    uint32_t k = buf[*pos] | buf[*pos + 1] << 8 | buf[*pos + 2] << 16 | (uint32_t)buf[*pos + 3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

static int binsearch(const unsigned char *keys, int n, int order, const unsigned char *target) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        const unsigned char *mid_key = keys + (long)mid * order;
        int cmp = memcmp(mid_key, target, order);
        if (cmp == 0) return mid;
        if (cmp < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);

    if (nw < 21) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = w[0] | w[1] << 8 | w[2] << 16 | w[3] << 24;
    if (magic != 0x4E47524D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    uint32_t d = w[4] | w[5] << 8 | w[6] << 16 | w[7] << 24;
    uint32_t C = w[8] | w[9] << 8 | w[10] << 16 | w[11] << 24;
    uint32_t max_order = w[12] | w[13] << 8 | w[14] << 16 | w[15] << 24;
    uint8_t global_best = w[16];

    const unsigned char *p = w + 17;
    const unsigned char **order_keys = malloc((max_order + 1) * sizeof(unsigned char*));
    const unsigned char **order_vals = malloc((max_order + 1) * sizeof(unsigned char*));
    int *order_n = malloc((max_order + 1) * sizeof(int));

    for (int order = 1; order <= max_order; order++) {
        if (p + 4 > w + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint32_t n = p[0] | p[1] << 8 | p[2] << 16 | p[3] << 24;
        p += 4;
        order_n[order] = n;
        order_keys[order] = p;
        p += (long)n * order;
        order_vals[order] = p;
        p += n;
        if (p > w + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_rec[1];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        uint8_t pred = global_best;
        for (int order = max_order; order >= 1; order--) {
            if (order_n[order] == 0) continue;
            const unsigned char *suffix = xi + d - order;
            int idx = binsearch(order_keys[order], order_n[order], order, suffix);
            if (idx >= 0) {
                pred = order_vals[order][idx];
                break;
            }
        }
        out_rec[0] = pred;
        put_rec(stdout, out_rec, 1);
    }
    return 0;
}
```