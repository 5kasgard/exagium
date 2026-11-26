```hypothesis
Family: label-ngram (variable-order Markov model with Laplace smoothing).
Why: Character-level language modeling benefits from n-gram statistics; a compact suffix-based model captures local context efficiently and is faster than linear models due to sparse lookups.
Failure: If dev accuracy <= 0.3375 (label-linear) or inference cost exceeds label-linear significantly.
```
```python
import json, struct, sys

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    class_chars = card["class_chars"]
    # Build mapping from byte value to class index (0..64)
    mapping = [255] * 256
    for idx, ch in enumerate(class_chars):
        mapping[ord(ch)] = idx

    train_in = records(sys.argv[2])
    train_key = records(sys.argv[3])
    n = len(train_in)
    # Tuning split: last 20%
    split = n * 4 // 5
    train_in_part = train_in[:split]
    train_key_part = train_key[:split]
    valid_in = train_in[split:]
    valid_key = train_key[split:]

    best_acc = -1.0
    best_order = 3
    for max_order in (3, 4, 5):
        counts_dict = {}
        global_counts = [0] * 65
        for ctx_bytes, tgt_bytes in zip(train_in_part, train_key_part):
            ctx = [mapping[b] for b in ctx_bytes]
            tgt = mapping[tgt_bytes[0]]
            global_counts[tgt] += 1
            for L in range(1, max_order + 1):
                suffix = tuple(ctx[-L:])
                key = 0
                for c in suffix:
                    key = key * 65 + c
                key = (key << 3) | L
                if key not in counts_dict:
                    counts_dict[key] = [0] * 65
                counts_dict[key][tgt] += 1

        majority_class = max(range(65), key=lambda c: global_counts[c])
        correct = 0
        total = 0
        for ctx_bytes, tgt_bytes in zip(valid_in, valid_key):
            ctx = [mapping[b] for b in ctx_bytes]
            tgt = mapping[tgt_bytes[0]]
            pred = majority_class
            for L in range(max_order, 0, -1):
                suffix = tuple(ctx[-L:])
                key = 0
                for c in suffix:
                    key = key * 65 + c
                key = (key << 3) | L
                if key in counts_dict:
                    cnts = counts_dict[key]
                    best = max(range(65), key=lambda c: cnts[c] + 1)
                    pred = best
                    break
            if pred == tgt:
                correct += 1
            total += 1
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_order = max_order

    # Retrain on full data with best_order
    counts_dict = {}
    global_counts = [0] * 65
    for ctx_bytes, tgt_bytes in zip(train_in, train_key):
        ctx = [mapping[b] for b in ctx_bytes]
        tgt = mapping[tgt_bytes[0]]
        global_counts[tgt] += 1
        for L in range(1, best_order + 1):
            suffix = tuple(ctx[-L:])
            key = 0
            for c in suffix:
                key = key * 65 + c
            key = (key << 3) | L
            if key not in counts_dict:
                counts_dict[key] = [0] * 65
            counts_dict[key][tgt] += 1
    majority_class = max(range(65), key=lambda c: global_counts[c])

    entries = sorted(counts_dict.items())
    num_entries = len(entries)

    with open(sys.argv[4], "wb") as f:
        # Magic: 'NGRM' = 0x4E47524D
        f.write(struct.pack("<I", 0x4E47524D))
        f.write(struct.pack("<B", best_order))
        f.write(struct.pack("<B", 65))
        f.write(struct.pack("<H", 0))          # reserved
        f.write(struct.pack("<I", num_entries))
        f.write(bytes(mapping))                # 256 bytes
        f.write(struct.pack("<B", majority_class))
        f.write(struct.pack("<BBB", 0, 0, 0))  # padding to 8-byte boundary
        for key, cnts in entries:
            f.write(struct.pack("<Q", key))
            for c in cnts:
                f.write(struct.pack("<H", c))

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

typedef struct {
    uint64_t key;
    uint16_t counts[65];
} Entry;

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 12) { fprintf(stderr, "weights too small\n"); return 3; }

    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | (uint32_t)wb[3] << 24;
    if (magic != 0x4E47524D) { fprintf(stderr, "bad magic\n"); return 3; }
    unsigned char max_order = wb[4];
    unsigned char alphabet_size = wb[5];
    if (alphabet_size != 65) { fprintf(stderr, "alphabet size mismatch\n"); return 3; }
    uint16_t reserved = wb[6] | wb[7] << 8;
    (void)reserved;
    uint32_t num_entries = wb[8] | wb[9] << 8 | wb[10] << 16 | (uint32_t)wb[11] << 24;

    const unsigned char *mapping = wb + 12;
    unsigned char majority_class = wb[12 + 256];
    const Entry *entries = (const Entry *)(wb + 12 + 256 + 4); // 4 bytes padding after majority_class

    long expected_size = 12 + 256 + 4 + (long)num_entries * sizeof(Entry);
    if (nw != expected_size) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        unsigned char ctx_idx[16];
        for (int j = 0; j < 16; j++) {
            unsigned char idx = mapping[xi[j]];
            if (idx == 255) { idx = majority_class; } // fallback for unknown byte
            ctx_idx[j] = idx;
        }

        unsigned char pred = majority_class;
        for (int L = max_order; L >= 1; L--) {
            uint64_t key = 0;
            for (int j = 16 - L; j < 16; j++) {
                key = key * 65 + ctx_idx[j];
            }
            key = (key << 3) | L;

            int lo = 0, hi = num_entries - 1, found = 0;
            while (lo <= hi) {
                int mid = (lo + hi) >> 1;
                if (entries[mid].key == key) {
                    const uint16_t *cnts = entries[mid].counts;
                    unsigned char best = 0;
                    uint16_t best_val = cnts[0] + 1;
                    for (int c = 1; c < 65; c++) {
                        uint16_t val = cnts[c] + 1;
                        if (val > best_val) {
                            best_val = val;
                            best = c;
                        }
                    }
                    pred = best;
                    found = 1;
                    break;
                } else if (entries[mid].key < key) {
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            if (found) break;
        }
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```