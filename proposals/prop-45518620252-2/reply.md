```hypothesis
We propose a variable-order Markov model (n-gram with backoff) called "label-ngram".  
It should do well because character-level language modeling benefits from longer contexts when they have been observed, and smoothing with backoff handles unseen contexts gracefully. The model is extremely fast at inference (a few hash lookups per item) and has no hyperparameters beyond the smoothing constant α, which is chosen on the tuning split. If it does not beat label-linear (0.3375), it would indicate that the task requires more generalization than raw n-gram statistics can provide.
```

```python
import json, math, struct, sys
from array import array

# ----------------------------------------------------------------------
# Record reading (same as the example)
# ----------------------------------------------------------------------
def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

# ----------------------------------------------------------------------
# Hash table builder for a given context length
# ----------------------------------------------------------------------
def build_hash_table(contexts, targets, L, alpha, num_classes):
    """
    contexts: list of bytes objects of length 16
    targets:  list of ints (0..num_classes-1)
    L: context length (1..16)
    Returns: (table_size, entries) where entries is a list of (key_bytes, counts_array)
             The list has length table_size; empty slots have key_bytes = None.
    """
    # Count occurrences
    count_dict = {}
    for ctx, tgt in zip(contexts, targets):
        key = ctx[16 - L:]  # suffix of length L
        if key not in count_dict:
            count_dict[key] = array('H', [0]) * num_classes
        count_dict[key][tgt] += 1

    num_entries = len(count_dict)
    # Choose table size as power of two >= 2 * num_entries (load factor <= 0.5)
    table_size = 1
    while table_size < 2 * num_entries:
        table_size <<= 1
    if table_size < 8:
        table_size = 8

    # FNV-1a 32-bit hash
    def fnv1a(key):
        h = 0x811c9dc5
        for b in key:
            h ^= b
            h = (h * 0x01000193) & 0xffffffff
        return h

    # Initialize table
    table = [None] * table_size
    for key, counts in count_dict.items():
        h = fnv1a(key)
        idx = h & (table_size - 1)
        while table[idx] is not None:
            idx = (idx + 1) & (table_size - 1)
        table[idx] = (key, counts)
    return table_size, table

# ----------------------------------------------------------------------
# Evaluation on a split (for tuning alpha)
# ----------------------------------------------------------------------
def evaluate(contexts, targets, tables, unigram_counts, alpha, num_classes):
    correct = 0
    total = len(contexts)
    for ctx, tgt in zip(contexts, targets):
        pred = None
        for L in range(16, 0, -1):
            table_size, table = tables[L]
            key = ctx[16 - L:]
            # FNV-1a hash
            h = 0x811c9dc5
            for b in key:
                h ^= b
                h = (h * 0x01000193) & 0xffffffff
            idx = h & (table_size - 1)
            for _ in range(table_size):
                entry = table[idx]
                if entry is None:
                    break
                if entry[0] == key:
                    counts = entry[1]
                    best = 0
                    best_val = counts[0] + alpha
                    for c in range(1, num_classes):
                        val = counts[c] + alpha
                        if val > best_val:
                            best_val = val
                            best = c
                    pred = best
                    break
                idx = (idx + 1) & (table_size - 1)
            if pred is not None:
                break
        if pred is None:
            # unigram backoff
            best = 0
            best_val = unigram_counts[0] + alpha
            for c in range(1, num_classes):
                val = unigram_counts[c] + alpha
                if val > best_val:
                    best_val = val
                    best = c
            pred = best
        if pred == tgt:
            correct += 1
    return correct / total

# ----------------------------------------------------------------------
def main():
    card = json.load(open(sys.argv[1]))
    num_classes = card["classes"]
    d = card["input"]["bytes"]  # 16

    train_in = records(sys.argv[2])
    train_key = records(sys.argv[3])
    X = train_in
    y = [r[0] for r in train_key]

    # Split: last 20% for tuning (by file order)
    n = len(X)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    # Hyperparameter search for alpha
    alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    best_alpha = 0.1
    best_acc = -1.0

    for alpha in alphas:
        # Build tables on training split
        tables = {}
        unigram = array('H', [0]) * num_classes
        for tgt in y_train:
            unigram[tgt] += 1
        for L in range(1, 17):
            tables[L] = build_hash_table(X_train, y_train, L, alpha, num_classes)
        acc = evaluate(X_tune, y_tune, tables, unigram, alpha, num_classes)
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    # Rebuild on all data with best_alpha
    tables = {}
    unigram = array('H', [0]) * num_classes
    for tgt in y:
        unigram[tgt] += 1
    for L in range(1, 17):
        tables[L] = build_hash_table(X, y, L, best_alpha, num_classes)

    # ------------------------------------------------------------------
    # Write weights.bin
    # ------------------------------------------------------------------
    # Header: magic(4), version(4), max_len(1), num_classes(1), reserved(2), alpha(4) = 16 bytes
    magic = 0x4E47524D  # 'NGRM'
    version = 1
    max_len = 16
    reserved = 0
    header = struct.pack("<IIBBHf", magic, version, max_len, num_classes, reserved, best_alpha)

    out = bytearray()
    out += header

    # Unigram counts (65 uint16)
    out += unigram.tobytes()

    # For each L=1..16: table_size (u32), num_entries (u32), then table_size entries
    for L in range(1, 17):
        table_size, table = tables[L]
        num_entries = sum(1 for e in table if e is not None)
        out += struct.pack("<II", table_size, num_entries)
        for entry in table:
            if entry is None:
                # empty slot: key = 0xFF * L, counts = zeros
                out += bytes([0xFF] * L)
                out += bytes(2 * num_classes)
            else:
                key, counts = entry
                out += key
                out += counts.tobytes()

    open(sys.argv[4], "wb").write(out)

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
    uint32_t k = buf[*pos] | buf[*pos + 1] << 8 | buf[*pos + 2] << 16 | (uint32_t)buf[*pos + 3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

// FNV-1a 32-bit
static uint32_t fnv1a(const unsigned char *key, int len) {
    uint32_t h = 0x811c9dc5;
    for (int i = 0; i < len; i++) {
        h ^= key[i];
        h *= 0x01000193;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Parse header
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = *(uint32_t*)w;
    if (magic != 0x4E47524D) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t version = *(uint32_t*)(w+4);
    if (version != 1) { fprintf(stderr, "bad version\n"); return 3; }
    uint8_t max_len = w[8];
    uint8_t num_classes = w[9];
    // reserved = w[10], w[11]
    float alpha = *(float*)(w+12);
    if (max_len != 16 || num_classes != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }

    const unsigned char *ptr = w + 16;
    long remaining = nw - 16;

    // Unigram counts
    if (remaining < 2 * num_classes) { fprintf(stderr, "truncated unigram\n"); return 3; }
    uint16_t *unigram = (uint16_t*)ptr;
    ptr += 2 * num_classes;
    remaining -= 2 * num_classes;

    // Tables for L=1..16
    struct Table {
        uint32_t size;
        uint32_t entries;
        const unsigned char *data; // start of entries (each entry: key[L] + counts[65*uint16])
        int keylen;
    } tables[17] = {0};

    for (int L = 1; L <= 16; L++) {
        if (remaining < 8) { fprintf(stderr, "truncated table header L=%d\n", L); return 3; }
        uint32_t tsize = *(uint32_t*)ptr; ptr += 4;
        uint32_t tentries = *(uint32_t*)ptr; ptr += 4;
        remaining -= 8;
        size_t entry_size = L + 2 * num_classes;
        if (remaining < (long)(tsize * entry_size)) { fprintf(stderr, "truncated table data L=%d\n", L); return 3; }
        tables[L].size = tsize;
        tables[L].entries = tentries;
        tables[L].data = ptr;
        tables[L].keylen = L;
        ptr += tsize * entry_size;
        remaining -= tsize * entry_size;
    }

    if (remaining != 0) { fprintf(stderr, "extra bytes in weights\n"); return 3; }

    // Inference loop
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        unsigned char best_class = 0;
        int found = 0;

        for (int L = 16; L >= 1; L--) {
            const struct Table *t = &tables[L];
            const unsigned char *key = xi + 16 - L; // suffix of length L
            uint32_t h = fnv1a(key, L);
            uint32_t idx = h & (t->size - 1);
            size_t entry_size = t->keylen + 2 * num_classes;
            for (uint32_t probe = 0; probe < t->size; probe++) {
                const unsigned char *entry = t->data + idx * entry_size;
                // Check if empty (first byte 0xFF)
                if (entry[0] == 0xFF) break;
                // Compare key
                if (memcmp(entry, key, L) == 0) {
                    const uint16_t *counts = (const uint16_t*)(entry + L);
                    float best_val = counts[0] + alpha;
                    best_class = 0;
                    for (int c = 1; c < num_classes; c++) {
                        float val = counts[c] + alpha;
                        if (val > best_val) {
                            best_val = val;
                            best_class = c;
                        }
                    }
                    found = 1;
                    break;
                }
                idx = (idx + 1) & (t->size - 1);
            }
            if (found) break;
        }

        if (!found) {
            float best_val = unigram[0] + alpha;
            best_class = 0;
            for (int c = 1; c < num_classes; c++) {
                float val = unigram[c] + alpha;
                if (val > best_val) {
                    best_val = val;
                    best_class = c;
                }
            }
        }

        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```