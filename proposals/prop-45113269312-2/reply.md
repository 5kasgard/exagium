```hypothesis
label-ngram: Variable-order character n-gram model (orders 1–5) with Laplace smoothing and backoff to longest observed suffix. Character-level language modeling is dominated by local n-gram statistics; the 16-byte context provides ample history for orders up to 5. The model uses compact hash tables (int16 quantized log-probs) for fast inference: only 1–5 hash lookups vs label-linear's 16 table lookups. Failure: dev accuracy ≤ 0.3375 (label-linear) or inference cost ≥ label-linear.
```
```python
import json, struct, sys
import numpy as np

MAX_ORDER = 5
ALPHA = 1.0
SCALE = 1024
CLASSES = 65

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def pack_suffix(suffix_bytes):
    val = 0
    for i, b in enumerate(suffix_bytes):
        val |= (b << (8 * i))
    return val

def main():
    card = json.load(open(sys.argv[1]))
    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    n = len(X_recs)
    d = card["input"]["bytes"]
    C = card["classes"]
    
    X = np.frombuffer(b"".join(X_recs), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in y_recs], dtype=np.int64)
    
    counts = [{} for _ in range(MAX_ORDER + 1)]
    totals = [{} for _ in range(MAX_ORDER + 1)]
    
    for i in range(n):
        ctx = X[i]
        target = y[i]
        for order in range(1, MAX_ORDER + 1):
            suffix = ctx[d - order:]
            key = pack_suffix(suffix)
            if key not in counts[order]:
                counts[order][key] = np.zeros(C, dtype=np.int64)
                totals[order][key] = 0
            counts[order][key][target] += 1
            totals[order][key] += 1
    
    logprobs = [{} for _ in range(MAX_ORDER + 1)]
    for order in range(1, MAX_ORDER + 1):
        for key, cnts in counts[order].items():
            total = totals[order][key]
            probs = (cnts + ALPHA) / (total + ALPHA * C)
            logp = np.log(probs)
            quantized = np.round(logp * SCALE).astype(np.int16)
            logprobs[order][key] = quantized
    
    global_counts = np.bincount(y, minlength=C)
    global_best = int(np.argmax(global_counts))
    
    tables = []
    for order in range(1, MAX_ORDER + 1):
        entries = logprobs[order]
        num_entries = len(entries)
        size = 16
        while size < num_entries * 2:
            size <<= 1
        keys_arr = np.zeros(size, dtype=np.uint64)
        logprobs_arr = np.zeros(size * C, dtype=np.int16)
        
        for key, lp in entries.items():
            h = (key * 11400714819323198485) & (size - 1)
            while keys_arr[h] != 0:
                h = (h + 1) & (size - 1)
            keys_arr[h] = key
            logprobs_arr[h*C:(h+1)*C] = lp
        tables.append((size, keys_arr, logprobs_arr))
    
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<I", 0x4E47524D))
        f.write(struct.pack("<B", MAX_ORDER))
        f.write(struct.pack("<B", global_best))
        f.write(struct.pack("<H", 0))
        for order in range(1, MAX_ORDER + 1):
            size, keys_arr, logprobs_arr = tables[order - 1]
            f.write(struct.pack("<I", size))
            f.write(keys_arr.astype("<Q").tobytes())
            f.write(logprobs_arr.astype("<h").tobytes())

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

#define MAX_ORDER 5
#define CLASSES 65

typedef struct {
    uint32_t size;
    uint64_t *keys;
    int16_t *logprobs;
} HashTable;

static int hash_lookup(const HashTable *ht, uint64_t key, const int16_t **out_logprobs) {
    uint32_t mask = ht->size - 1;
    uint32_t h = (uint32_t)((key * 11400714819323198485ULL) & mask);
    uint32_t start = h;
    do {
        if (ht->keys[h] == 0) return 0;
        if (ht->keys[h] == key) {
            *out_logprobs = ht->logprobs + h * CLASSES;
            return 1;
        }
        h = (h + 1) & mask;
    } while (h != start);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    
    if (nw < 8) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = *(uint32_t*)wb;
    if (magic != 0x4E47524D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    uint8_t max_order = wb[4];
    uint8_t global_best = wb[5];
    if (max_order != MAX_ORDER) { fprintf(stderr, "max_order mismatch\n"); return 3; }
    
    uint32_t offset = 8;
    HashTable tables[MAX_ORDER + 1];
    for (int order = 1; order <= MAX_ORDER; order++) {
        if (offset + 4 > nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint32_t size = *(uint32_t*)(wb + offset);
        offset += 4;
        if (offset + size * 8 > nw) { fprintf(stderr, "truncated weights keys\n"); return 3; }
        tables[order].size = size;
        tables[order].keys = (uint64_t*)(wb + offset);
        offset += size * 8;
        if (offset + size * CLASSES * 2 > nw) { fprintf(stderr, "truncated weights logprobs\n"); return 3; }
        tables[order].logprobs = (int16_t*)(wb + offset);
        offset += size * CLASSES * 2;
    }
    if (offset != nw) { fprintf(stderr, "weights size mismatch %ld != %ld\n", offset, nw); return 3; }
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        
        unsigned char best = global_best;
        for (int order = MAX_ORDER; order >= 1; order--) {
            uint64_t key = 0;
            for (int j = 0; j < order; j++) {
                key |= (uint64_t)xi[16 - order + j] << (8 * j);
            }
            const int16_t *logprobs;
            if (hash_lookup(&tables[order], key, &logprobs)) {
                int best_c = 0;
                int16_t best_val = logprobs[0];
                for (int c = 1; c < CLASSES; c++) {
                    if (logprobs[c] > best_val) {
                        best_val = logprobs[c];
                        best_c = c;
                    }
                }
                best = (unsigned char)best_c;
                break;
            }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```