```hypothesis
Family: label-ngram (variable-order Markov with longest-match backoff).
Why: Character-level language modeling benefits from exact context matching; tinyshakespeare has repetitive patterns. A longest-match suffix model captures variable-length dependencies without heavy computation. It should beat linear (0.3375) by leveraging exact n-gram statistics, and be faster than GBDT at inference.
Failure: If dev accuracy <= 0.3375 or inference cost > label-linear (7359 Ir/item), the approach does not improve on the baseline.
```

```python
"""label-ngram: variable-order Markov model with longest-match backoff.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic, single-threaded, stdlib + numpy only."""
import json, struct, sys
import numpy as np

MAGIC = 0x4E47524D  # 'MRGN' little-endian
VERSION = 1
NUM_LENGTHS = 16
EMPTY_BYTE = 0xFF
LOAD_FACTOR = 0.5  # table_size >= num_entries / LOAD_FACTOR

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def fnv1a_32(key_bytes, length):
    h = 0x811c9dc5
    for i in range(length):
        h ^= key_bytes[i]
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

def next_pow2(x):
    return 1 << (x - 1).bit_length() if x > 0 else 1

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65
    contexts = records(sys.argv[2])
    targets = records(sys.argv[3])
    n = len(contexts)
    assert all(len(c) == d for c in contexts)
    assert all(len(t) == 1 for t in targets)
    y = np.array([t[0] for t in targets], dtype=np.int64)

    # Count occurrences for each suffix length
    # counts[L][key_bytes] = array of 65 counts
    counts = [{} for _ in range(NUM_LENGTHS + 1)]  # index by length 1..16
    for ctx, target in zip(contexts, y):
        for L in range(1, NUM_LENGTHS + 1):
            key = ctx[d - L:]  # last L bytes
            arr = counts[L].get(key)
            if arr is None:
                arr = np.zeros(C, dtype=np.int32)
                counts[L][key] = arr
            arr[target] += 1

    # Determine majority class for each key
    tables = []  # list of (length, sorted_keys, majority_classes)
    global_counts = np.zeros(C, dtype=np.int64)
    for target in y:
        global_counts[target] += 1
    global_majority = int(np.argmax(global_counts))

    for L in range(1, NUM_LENGTHS + 1):
        items = list(counts[L].items())
        items.sort(key=lambda kv: kv[0])  # deterministic order
        keys = [kv[0] for kv in items]
        majors = [int(np.argmax(kv[1])) for kv in items]
        tables.append((L, keys, majors))

    # Build hash tables for each length
    table_data = []  # each entry: (table_size, num_entries, keys_flat, values_flat)
    for L, keys, majors in tables:
        num_entries = len(keys)
        table_size = next_pow2(int(num_entries / LOAD_FACTOR) + 1)
        # Initialize with EMPTY_BYTE
        keys_flat = bytearray([EMPTY_BYTE] * (table_size * 16))
        values_flat = bytearray([0] * table_size)
        for key, val in zip(keys, majors):
            h = fnv1a_32(key, L) % table_size
            while True:
                # Check if slot empty (first byte == EMPTY_BYTE)
                if keys_flat[h * 16] == EMPTY_BYTE:
                    # Write key (first L bytes), rest remain EMPTY_BYTE
                    for i in range(L):
                        keys_flat[h * 16 + i] = key[i]
                    values_flat[h] = val
                    break
                # If same key already present (should not happen due to unique keys), skip
                match = True
                for i in range(L):
                    if keys_flat[h * 16 + i] != key[i]:
                        match = False
                        break
                if match:
                    break
                h = (h + 1) % table_size
        table_data.append((table_size, num_entries, keys_flat, values_flat))

    # Write weights.bin
    with open(sys.argv[4], "wb") as f:
        # Header: magic, version, num_lengths, global_majority, reserved
        f.write(struct.pack("<IBBBB", MAGIC, VERSION, NUM_LENGTHS, global_majority, 0))
        # Per-length table_size and num_entries
        for table_size, num_entries, _, _ in table_data:
            f.write(struct.pack("<II", table_size, num_entries))
        # Tables data
        for table_size, _, keys_flat, values_flat in table_data:
            f.write(keys_flat)
            f.write(values_flat)

if __name__ == "__main__":
    main()
```

```c
/* label-ngram inference: ./infer weights.bin in.rec N */
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

static uint32_t fnv1a_32(const unsigned char *key, int len) {
    uint32_t h = 0x811c9dc5u;
    for (int i = 0; i < len; i++) {
        h ^= key[i];
        h *= 0x01000193u;
    }
    return h;
}

#define EMPTY_BYTE 0xFF
#define CONTEXT_LEN 16
#define NUM_LENGTHS 16

typedef struct {
    uint32_t table_size;
    uint32_t num_entries;
    const unsigned char *keys;   // table_size * 16 bytes
    const unsigned char *values; // table_size bytes
} HashTable;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 9) { fprintf(stderr, "weights too small\n"); return 3; }

    // Parse header
    const unsigned char *p = w;
    uint32_t magic = p[0] | p[1] << 8 | p[2] << 16 | p[3] << 24;
    if (magic != 0x4E47524Du) { fprintf(stderr, "bad magic\n"); return 3; }
    uint8_t version = p[4];
    uint8_t num_lengths = p[5];
    uint8_t global_majority = p[6];
    uint8_t reserved = p[7];
    if (version != 1 || num_lengths != NUM_LENGTHS) { fprintf(stderr, "bad version/num_lengths\n"); return 3; }
    p += 8;

    HashTable tables[NUM_LENGTHS];
    for (int i = 0; i < NUM_LENGTHS; i++) {
        if (p + 8 > w + nw) { fprintf(stderr, "truncated header\n"); return 3; }
        tables[i].table_size = p[0] | p[1] << 8 | p[2] << 16 | p[3] << 24;
        tables[i].num_entries = p[4] | p[5] << 8 | p[6] << 16 | p[7] << 24;
        p += 8;
    }

    // Point to tables data
    for (int i = 0; i < NUM_LENGTHS; i++) {
        uint32_t ts = tables[i].table_size;
        if (p + (long)ts * 17 > w + nw) { fprintf(stderr, "truncated table data\n"); return 3; }
        tables[i].keys = p;
        tables[i].values = p + (long)ts * 16;
        p += (long)ts * 17;
    }
    if (p != w + nw) { fprintf(stderr, "weights size mismatch\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != CONTEXT_LEN) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, CONTEXT_LEN); return 3; }

        int found = 0;
        for (int L = NUM_LENGTHS; L >= 1 && !found; L--) {
            const HashTable *ht = &tables[L - 1];
            uint32_t h = fnv1a_32(xi + CONTEXT_LEN - L, L) % ht->table_size;
            for (uint32_t probes = 0; probes < ht->table_size; probes++) {
                const unsigned char *slot_key = ht->keys + (long)h * 16;
                if (slot_key[0] == EMPTY_BYTE) break; // empty
                int match = 1;
                for (int j = 0; j < L; j++) {
                    if (slot_key[j] != xi[CONTEXT_LEN - L + j]) { match = 0; break; }
                }
                if (match) {
                    out_byte = ht->values[h];
                    found = 1;
                    break;
                }
                h = (h + 1) % ht->table_size;
            }
        }
        if (!found) out_byte = global_majority;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```