```hypothesis
Family: label-ngram (variable-order Markov with deterministic backoff). Character-level LM has strong local sequential structure; n-gram backoff captures variable-length interactions (like "the " → ' ') more parameter-efficiently than linear one-hot (which needs explicit crosses) or small MLPs. GBDT wins at 0.392 by capturing interactions; this does so with tiny lookup tables. Tuning max_order on the held-out 20% selects the sweet spot. Failure: dev accuracy ≤ 0.34 or Ir/item > 15000.
```
```python
"""label-ngram: variable-order Markov with backoff for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy. Builds n-gram count tables on
train split, chooses max_order (1..8) on last 20% tuning split, then rebuilds on all
train. Exports sorted (context -> top_pred) tables per order for binary search."""
import json, struct, sys
from collections import defaultdict

import numpy as np

SEED = 1
MAX_ORDER = 8  # will be tuned down to 1..8


def read_recs(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_tables(X, y, max_order):
    """X: (n, 16) uint8, y: (n,) uint8. Returns list of (order, contexts, preds) sorted."""
    tables = []
    for order in range(1, max_order + 1):
        counts = defaultdict(lambda: np.zeros(65, dtype=np.int32))
        for ctx, nxt in zip(X, y):
            key = bytes(ctx[-order:])
            counts[key][nxt] += 1
        # sort contexts lexicographically, store top prediction (tie -> smallest class)
        items = sorted(counts.items())
        contexts = bytearray()
        preds = bytearray()
        for ctx, cnt in items:
            contexts += ctx
            preds.append(int(cnt.argmax()))
        tables.append((order, len(items), contexts, preds))
    return tables


def predict_one(ctx, tables):
    """ctx: 16 bytes. Return predicted class using backoff."""
    for order, n_entries, contexts, preds in reversed(tables):
        if n_entries == 0:
            continue
        key = ctx[-order:]
        # binary search in contexts (each entry is `order` bytes)
        lo, hi = 0, n_entries
        while lo < hi:
            mid = (lo + hi) // 2
            start = mid * order
            mid_ctx = contexts[start:start + order]
            if mid_ctx < key:
                lo = mid + 1
            else:
                hi = mid
        if lo < n_entries:
            start = lo * order
            if contexts[start:start + order] == key:
                return preds[lo]
    # ultimate fallback: global majority (should not happen if order=1 exists)
    return 0


def evaluate(X, y, tables):
    correct = 0
    for ctx, nxt in zip(X, y):
        if predict_one(ctx, tables) == nxt:
            correct += 1
    return correct / len(y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(read_recs(sys.argv[2])), dtype=np.uint8).reshape(-1, 16)
    y = np.array([r[0] for r in read_recs(sys.argv[3])], dtype=np.uint8)
    n = len(X)
    assert card["input"]["bytes"] == 16 and card["classes"] == 65

    # split: last 20% for tuning (by file order)
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    best_order, best_acc = 1, -1.0
    for order in range(1, MAX_ORDER + 1):
        tables = build_tables(X_tr, y_tr, order)
        acc = evaluate(X_tune, y_tune, tables)
        if acc > best_acc:
            best_acc, best_order = acc, order

    # rebuild on all data with best_order
    tables = build_tables(X, y, best_order)

    # write weights.bin
    # magic: 0x4E47524D ('M' 'R' 'G' 'N' little-endian = "NGRM")
    # int32: max_order
    # per order: int32 num_entries, then contexts (num_entries * order bytes), preds (num_entries bytes)
    out = bytearray()
    out += struct.pack("<Ii", 0x4E47524D, best_order)
    for order, n_entries, contexts, preds in tables:
        if order > best_order:
            continue
        out += struct.pack("<i", n_entries)
        out += contexts
        out += preds
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-ngram inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes context. Output record: 1 byte predicted class.
 * Weights format: magic(0x4E47524D), max_order(i32), then per order 1..max_order:
 *   num_entries(i32), contexts[num_entries][order](bytes), preds[num_entries](bytes).
 * Contexts sorted lexicographically. Backoff from max_order down to 1. */
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

/* binary search in sorted fixed-width contexts. Returns index or -1. */
static int bsearch_ctx(const unsigned char *contexts, int n_entries, int order, const unsigned char *key) {
    int lo = 0, hi = n_entries;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        const unsigned char *mid_ctx = contexts + (size_t)mid * order;
        int cmp = memcmp(mid_ctx, key, order);
        if (cmp < 0) lo = mid + 1;
        else hi = mid;
    }
    if (lo < n_entries && memcmp(contexts + (size_t)lo * order, key, order) == 0)
        return lo;
    return -1;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 8) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = w[0] | w[1] << 8 | w[2] << 16 | w[3] << 24;
    if (magic != 0x4E47524D) { fprintf(stderr, "bad magic\n"); return 3; }
    int max_order = (int32_t)(w[4] | w[5] << 8 | w[6] << 16 | w[7] << 24);
    if (max_order < 1 || max_order > 16) { fprintf(stderr, "bad max_order\n"); return 3; }

    size_t offset = 8;
    struct OrderTable { int order, n_entries; const unsigned char *contexts, *preds; } tables[16];
    for (int o = 1; o <= max_order; o++) {
        if (offset + 4 > (size_t)nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        int n_entries = (int32_t)(w[offset] | w[offset+1]<<8 | w[offset+2]<<16 | w[offset+3]<<24);
        offset += 4;
        if (n_entries < 0) { fprintf(stderr, "bad n_entries\n"); return 3; }
        size_t need = (size_t)n_entries * o + n_entries;
        if (offset + need > (size_t)nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        tables[o].order = o;
        tables[o].n_entries = n_entries;
        tables[o].contexts = w + offset;
        tables[o].preds = w + offset + (size_t)n_entries * o;
        offset += need;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        unsigned char pred = 0;
        for (int o = max_order; o >= 1; o--) {
            if (tables[o].n_entries == 0) continue;
            int idx = bsearch_ctx(tables[o].contexts, tables[o].n_entries, o, xi + 16 - o);
            if (idx >= 0) { pred = tables[o].preds[idx]; break; }
        }
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```