```hypothesis
The label-ngram family uses variable-length n-grams (up to order 8) with backoff, capturing local sequential dependencies that linear models miss. For character-level language modeling, n-grams are strong baselines; they should beat label-linear (0.3375) on dev accuracy while being extremely fast at inference (binary search on sorted arrays). If dev accuracy does not exceed 0.34, the approach fails.
```

```python
import json, struct, sys
from collections import defaultdict

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def pack_key(bytes_seq):
    """Pack a byte sequence (len <= 8) into a uint64, first byte in highest bits."""
    val = 0
    for b in bytes_seq:
        val = (val << 8) | b
    return val

def build_model(contexts, targets, max_order):
    """Build n-gram model: for each order 1..max_order, map packed key -> best class."""
    # counts[order][key] = array of 65 counts
    counts = [defaultdict(lambda: [0]*65) for _ in range(max_order + 1)]
    global_counts = [0]*65
    for ctx, tgt in zip(contexts, targets):
        tgt = tgt[0] if isinstance(tgt, (bytes, bytearray)) else tgt
        global_counts[tgt] += 1
        for order in range(1, max_order + 1):
            if order <= len(ctx):
                key = pack_key(ctx[-order:])
                counts[order][key][tgt] += 1
    # Determine best class for each key (tie-break smallest class)
    model = {}
    for order in range(1, max_order + 1):
        keys = []
        vals = []
        for key, cnts in counts[order].items():
            best = max(range(65), key=lambda c: (cnts[c], -c))
            keys.append(key)
            vals.append(best)
        # Sort by key for binary search
        paired = sorted(zip(keys, vals))
        model[order] = ( [p[0] for p in paired], [p[1] for p in paired] )
    # Global fallback
    global_best = max(range(65), key=lambda c: (global_counts[c], -c))
    return model, global_best

def evaluate(model, global_best, contexts, targets, max_order):
    correct = 0
    for ctx, tgt in zip(contexts, targets):
        tgt = tgt[0] if isinstance(tgt, (bytes, bytearray)) else tgt
        pred = global_best
        for order in range(max_order, 0, -1):
            if order <= len(ctx):
                key = pack_key(ctx[-order:])
                keys, vals = model[order]
                # binary search
                lo, hi = 0, len(keys)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if keys[mid] < key:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < len(keys) and keys[lo] == key:
                    pred = vals[lo]
                    break
        if pred == tgt:
            correct += 1
    return correct / len(contexts)

def main():
    card = json.load(open(sys.argv[1]))
    ctxs = records(sys.argv[2])  # each 16 bytes
    tgts = records(sys.argv[3])  # each 1 byte
    n = len(ctxs)
    tune_size = n // 5
    train_ctxs, train_tgts = ctxs[:-tune_size], tgts[:-tune_size]
    tune_ctxs, tune_tgts = ctxs[-tune_size:], tgts[-tune_size:]

    best_order = 4
    best_acc = -1.0
    for order in range(4, 9):  # try 4..8
        model, gb = build_model(train_ctxs, train_tgts, order)
        acc = evaluate(model, gb, tune_ctxs, tune_tgts, order)
        if acc > best_acc:
            best_acc = acc
            best_order = order

    # Retrain on all data
    model, global_best = build_model(ctxs, tgts, best_order)

    # Write weights.bin
    # Format: magic(4), max_order(4), then for each order 1..max_order: count(4), keys(uint64[count]), vals(uint8[count]), global_best(1)
    magic = 0x4E47524D  # 'NGRM'
    out = bytearray()
    out += struct.pack("<II", magic, best_order)
    for order in range(1, best_order + 1):
        keys, vals = model[order]
        cnt = len(keys)
        out += struct.pack("<I", cnt)
        for k in keys:
            out += struct.pack("<Q", k)
        out.extend(bytes(vals))
    out.append(global_best)
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

static int binary_search(const uint64_t *keys, int cnt, uint64_t key) {
    int lo = 0, hi = cnt;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (keys[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    return (lo < cnt && keys[lo] == key) ? lo : -1;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | (uint32_t)wb[3] << 24;
    if (magic != 0x4E47524D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    uint32_t max_order = wb[4] | wb[5] << 8 | wb[6] << 16 | (uint32_t)wb[7] << 24;
    if (max_order < 1 || max_order > 8) { fprintf(stderr, "bad max_order\n"); return 3; }

    const unsigned char *ptr = wb + 8;
    uint64_t **order_keys = calloc(max_order + 1, sizeof(uint64_t*));
    unsigned char **order_vals = calloc(max_order + 1, sizeof(unsigned char*));
    uint32_t *order_cnt = calloc(max_order + 1, sizeof(uint32_t));

    for (uint32_t o = 1; o <= max_order; o++) {
        if (ptr + 4 > wb + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint32_t cnt = ptr[0] | ptr[1] << 8 | ptr[2] << 16 | (uint32_t)ptr[3] << 24;
        ptr += 4;
        if (ptr + cnt * 8 > wb + nw) { fprintf(stderr, "truncated weights keys\n"); return 3; }
        order_keys[o] = (uint64_t*)ptr;
        ptr += cnt * 8;
        if (ptr + cnt > wb + nw) { fprintf(stderr, "truncated weights vals\n"); return 3; }
        order_vals[o] = ptr;
        ptr += cnt;
        order_cnt[o] = cnt;
    }
    if (ptr + 1 != wb + nw) { fprintf(stderr, "weights size mismatch\n"); return 3; }
    unsigned char global_best = *ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        unsigned char pred = global_best;
        for (int o = max_order; o >= 1; o--) {
            uint64_t key = 0;
            for (int j = 16 - o; j < 16; j++) key = (key << 8) | xi[j];
            int idx = binary_search(order_keys[o], order_cnt[o], key);
            if (idx >= 0) { pred = order_vals[o][idx]; break; }
        }
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```