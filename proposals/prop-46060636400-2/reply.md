```hypothesis
We propose "label-ngram": a stupid-backoff n-gram model (orders 1–5) that predicts the next character from the previous 16 bytes. Character-level language modeling benefits strongly from local n-gram statistics; a backoff model captures these dependencies with minimal parameters. Inference requires only a few hash-table lookups (one per order), so CPU cost should be far below label-linear. We expect dev accuracy > 0.38 (beating label-linear's 0.3375 and approaching label-gbdt's 0.3923). If dev accuracy is ≤ 0.34, the hypothesis fails.
```

```python
"""label-ngram: stupid-backoff n-gram model for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (fixed seeds, sorted insertion). Builds hash tables for orders 1..5.
Weights format: magic 0x4E47524D, max_order=5, then per-order (capacity, context_len),
followed by keys (capacity*context_len bytes) and values (capacity bytes, 0xFF=empty)."""
import json, struct, sys
from collections import defaultdict

MAGIC = 0x4E47524D  # "NGRM"
MAX_ORDER = 5
SEED = 42  # not used but kept for determinism


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fnv1a_32(data: bytes) -> int:
    h = 0x811c9dc5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def build_hash_table(entries, context_len):
    """entries: list of (context_bytes, predicted_class). Returns (capacity, keys_bytes, values_bytes)."""
    n = len(entries)
    capacity = next_pow2(max(2 * n, 1))
    keys = bytearray(capacity * context_len)
    values = bytearray([0xFF] * capacity)
    # deterministic insertion: sort by context bytes
    entries.sort(key=lambda x: x[0])
    for ctx, val in entries:
        h = fnv1a_32(ctx)
        idx = h & (capacity - 1)
        while True:
            if values[idx] == 0xFF:
                # empty slot
                keys[idx * context_len:(idx + 1) * context_len] = ctx
                values[idx] = val
                break
            # check if same key (should not happen with unique entries)
            if keys[idx * context_len:(idx + 1) * context_len] == ctx:
                values[idx] = val
                break
            idx = (idx + 1) & (capacity - 1)
    return capacity, keys, values


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65

    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    assert len(X_recs) == len(y_recs)
    n = len(X_recs)

    # Collect training examples: each X is 16 bytes, y is 1 byte (0..64)
    examples = [(X_recs[i], y_recs[i][0]) for i in range(n)]

    # Count n-grams for orders 1..MAX_ORDER
    # order 1: unigram (no context)
    # order k (2..MAX_ORDER): context length = k-1 (last k-1 bytes of the 16-byte context)
    counts = [defaultdict(lambda: [0] * C) for _ in range(MAX_ORDER + 1)]  # index by order
    for ctx_bytes, target in examples:
        # order 1
        counts[1][()][target] += 1
        # higher orders
        for order in range(2, MAX_ORDER + 1):
            clen = order - 1
            if clen <= d:
                key = ctx_bytes[d - clen:]
                counts[order][key][target] += 1

    # Determine predicted class for each context (argmax, tie-break by smallest class)
    pred = [{} for _ in range(MAX_ORDER + 1)]
    for order in range(1, MAX_ORDER + 1):
        for key, cnts in counts[order].items():
            best = 0
            best_cnt = cnts[0]
            for c in range(1, C):
                if cnts[c] > best_cnt:
                    best_cnt = cnts[c]
                    best = c
            pred[order][key] = best

    # Build hash tables for orders 2..MAX_ORDER
    tables = [None] * (MAX_ORDER + 1)  # each entry: (capacity, context_len, keys, values)
    # order 1: special case, just store the single predicted class
    tables[1] = (1, 0, b"", bytes([pred[1][()]]))

    for order in range(2, MAX_ORDER + 1):
        clen = order - 1
        entries = [(key, pred[order][key]) for key in pred[order]]
        capacity, keys, values = build_hash_table(entries, clen)
        tables[order] = (capacity, clen, keys, values)

    # Write weights file
    with open(sys.argv[4], "wb") as f:
        # Header: magic (4), max_order (1), num_orders (1), reserved (2)
        f.write(struct.pack("<IBBH", MAGIC, MAX_ORDER, MAX_ORDER, 0))
        # Per-order info: capacity (4), context_len (1), reserved (3)
        for order in range(1, MAX_ORDER + 1):
            cap, clen, _, _ = tables[order]
            f.write(struct.pack("<IBBB", cap, clen, 0, 0))
        # Data for each order: keys then values
        for order in range(1, MAX_ORDER + 1):
            cap, clen, keys, values = tables[order]
            f.write(keys)
            f.write(values)


if __name__ == "__main__":
    main()
```

```c
/* label-ngram inference: ./infer weights.bin in.rec N
 * Stupid-backoff n-gram model (orders 1..5). Each input record is 16 bytes.
 * Output record is 1 byte: predicted class (0..64). */
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

static uint32_t fnv1a_32(const unsigned char *data, size_t len) {
    uint32_t h = 0x811c9dc5;
    for (size_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 0x01000193;
    }
    return h;
}

typedef struct {
    uint32_t capacity;
    uint8_t context_len;
    const unsigned char *keys;   // capacity * context_len
    const unsigned char *values; // capacity
} HashTable;

static uint8_t lookup(const HashTable *ht, const unsigned char *ctx) {
    if (ht->capacity == 0) return 0xFF;
    if (ht->context_len == 0) return ht->values[0]; // order 1
    uint32_t h = fnv1a_32(ctx, ht->context_len);
    uint32_t mask = ht->capacity - 1;
    uint32_t idx = h & mask;
    for (uint32_t i = 0; i < ht->capacity; i++) {
        uint32_t probe = (idx + i) & mask;
        uint8_t val = ht->values[probe];
        if (val == 0xFF) break; // empty slot -> not found
        const unsigned char *key = ht->keys + probe * ht->context_len;
        if (memcmp(key, ctx, ht->context_len) == 0) return val;
    }
    return 0xFF; // not found
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Parse header
    if (nw < 8) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = w[0] | w[1] << 8 | w[2] << 16 | w[3] << 24;
    if (magic != 0x4E47524D) { fprintf(stderr, "bad magic\n"); return 3; }
    uint8_t max_order = w[4];
    uint8_t num_orders = w[5];
    if (max_order != 5 || num_orders != 5) { fprintf(stderr, "unexpected order config\n"); return 3; }
    size_t offset = 8;
    HashTable tables[6]; // 1..5
    for (int order = 1; order <= 5; order++) {
        if (offset + 8 > (size_t)nw) { fprintf(stderr, "truncated header\n"); return 3; }
        uint32_t cap = w[offset] | w[offset+1] << 8 | w[offset+2] << 16 | w[offset+3] << 24;
        uint8_t clen = w[offset+4];
        offset += 8;
        tables[order].capacity = cap;
        tables[order].context_len = clen;
        tables[order].keys = (clen == 0) ? NULL : w + offset;
        offset += (size_t)cap * clen;
        tables[order].values = w + offset;
        offset += cap;
    }
    if (offset != (size_t)nw) { fprintf(stderr, "weights size mismatch\n"); return 3; }

    float *dummy = NULL; // not used, but keep for compatibility with example's cleanup
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        uint8_t pred = 0xFF;
        // try orders 5 down to 1
        for (int order = 5; order >= 1; order--) {
            uint8_t clen = tables[order].context_len;
            const unsigned char *ctx = xi + 16 - clen;
            pred = lookup(&tables[order], ctx);
            if (pred != 0xFF) break;
        }
        if (pred == 0xFF) pred = 0; // should never happen (order 1 always has a value)
        put_rec(stdout, &pred, 1);
    }
    free(w); free(x);
    return 0;
}
```