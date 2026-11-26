```hypothesis
Family: label-ngram (n-gram backoff model with binary search).
Why: Character-level language modeling benefits from exact n-gram statistics; a backoff model captures variable-length dependencies without neural net overhead. The tiny training set (24k) allows storing all observed n-grams up to length 8 in compact sorted arrays, giving O(log n) inference.
Failure mode: If dev accuracy <= 0.34 (label-linear) or inference cost exceeds label-linear, the model does not improve.
```

```python
import json, struct, sys
import numpy as np

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def build_tables(X, y, d, max_order):
    tables = {L: {} for L in range(1, max_order+1)}
    class_counts = np.zeros(256, dtype=np.int32)
    for i in range(len(X)):
        ctx = X[i]
        target = y[i]
        class_counts[target] += 1
        for L in range(1, max_order+1):
            if L > d: break
            key = 0
            for j in range(d-L, d):
                key = (key << 8) | ctx[j]
            if key not in tables[L]:
                tables[L][key] = np.zeros(256, dtype=np.int32)
            tables[L][key][target] += 1
    majority = int(np.argmax(class_counts))
    final_tables = {'majority': majority}
    for L in range(1, max_order+1):
        entries = []
        for key, counts in tables[L].items():
            best_class = int(np.argmax(counts))
            entries.append((key, best_class))
        entries.sort(key=lambda x: x[0])
        final_tables[L] = entries
    return final_tables

def evaluate(X, y, tables, d, max_order):
    correct = 0
    for i in range(len(X)):
        ctx = X[i]
        pred = tables['majority']
        for L in range(max_order, 0, -1):
            if L > d: continue
            key = 0
            for j in range(d-L, d):
                key = (key << 8) | ctx[j]
            entries = tables[L]
            lo, hi = 0, len(entries)
            while lo < hi:
                mid = (lo + hi) // 2
                if entries[mid][0] < key:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < len(entries) and entries[lo][0] == key:
                pred = entries[lo][1]
                break
        if pred == y[i]:
            correct += 1
    return correct / len(X)

def write_weights(path, tables, max_order, d, C):
    majority = tables['majority']
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0x4E47524D))
        f.write(struct.pack("<B", max_order))
        f.write(struct.pack("<B", majority))
        f.write(struct.pack("<H", 0))
        for L in range(1, max_order+1):
            entries = tables[L]
            f.write(struct.pack("<I", len(entries)))
            for key, cls in entries:
                f.write(struct.pack("<Q", key))
                f.write(struct.pack("<B", cls))

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    contexts = records(sys.argv[2])
    keys = records(sys.argv[3])
    X = np.array([list(c) for c in contexts], dtype=np.uint8)
    y = np.array([k[0] for k in keys], dtype=np.uint8)
    n = len(X)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]
    
    best_order = 1
    best_acc = -1.0
    for max_order in range(1, 9):
        tables = build_tables(X_train, y_train, d, max_order)
        acc = evaluate(X_tune, y_tune, tables, d, max_order)
        if acc > best_acc:
            best_acc = acc
            best_order = max_order
    tables = build_tables(X, y, d, best_order)
    write_weights(sys.argv[4], tables, best_order, d, C)

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
    uint32_t k = buf[*pos] | buf[*pos+1] << 8 | buf[*pos+2] << 16 | (uint32_t)buf[*pos+3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

static int binary_search(const uint64_t *keys, const uint8_t *classes, uint32_t n, uint64_t key, uint8_t *out_class) {
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        if (keys[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    if (lo < n && keys[lo] == key) {
        *out_class = classes[lo];
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 8) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = w[0] | w[1] << 8 | w[2] << 16 | (uint32_t)w[3] << 24;
    if (magic != 0x4E47524D) { fprintf(stderr, "bad magic\n"); return 3; }
    uint8_t max_order = w[4];
    uint8_t majority = w[5];
    const unsigned char *ptr = w + 8;
    long remaining = nw - 8;
    uint64_t **keys = malloc((max_order+1) * sizeof(uint64_t*));
    uint8_t **classes = malloc((max_order+1) * sizeof(uint8_t*));
    uint32_t *counts = malloc((max_order+1) * sizeof(uint32_t));
    for (int L = 1; L <= max_order; L++) {
        if (remaining < 4) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint32_t cnt = ptr[0] | ptr[1] << 8 | ptr[2] << 16 | (uint32_t)ptr[3] << 24;
        ptr += 4; remaining -= 4;
        counts[L] = cnt;
        if (remaining < (long)cnt * 9) { fprintf(stderr, "truncated weights\n"); return 3; }
        keys[L] = malloc(cnt * sizeof(uint64_t));
        classes[L] = malloc(cnt * sizeof(uint8_t));
        for (uint32_t i = 0; i < cnt; i++) {
            uint64_t key = 0;
            for (int b = 0; b < 8; b++) key |= (uint64_t)ptr[b] << (8*b);
            keys[L][i] = key;
            classes[L][i] = ptr[8];
            ptr += 9;
            remaining -= 9;
        }
    }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        uint8_t pred = majority;
        for (int L = max_order; L >= 1; L--) {
            if (L > 16) continue;
            uint64_t key = 0;
            for (int j = 16 - L; j < 16; j++) key = (key << 8) | xi[j];
            uint8_t cls;
            if (binary_search(keys[L], classes[L], counts[L], key, &cls)) {
                pred = cls;
                break;
            }
        }
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```