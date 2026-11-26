```hypothesis
Family: label-ngram (n-gram language model with backoff).
Why: The task is next-character prediction from 16 bytes of context. N-gram models capture local sequential dependencies that the linear model misses, and they are extremely fast at inference (binary search in sorted tables, no floating-point). They should exceed label-linear's 0.3375 accuracy while keeping CPU cost very low.
Failure signal: Dev accuracy ≤ 0.34 or instructions per item significantly above label-linear's 7360.
```

```python
import json
import struct
import sys
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
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    d = card["input"]["bytes"]
    C = card["classes"]
    class_chars = card["class_chars"]
    # map byte value -> class index
    byte_to_class = {ord(ch): i for i, ch in enumerate(class_chars)}
    # convert to class indices
    X = []
    for rec in X_bytes:
        arr = [byte_to_class.get(b, 0) for b in rec]
        X.append(bytes(arr))
    y = [byte_to_class.get(rec[0], 0) for rec in y_bytes]

    n = len(X)
    tune_size = n // 5
    train_X, tune_X = X[:-tune_size], X[-tune_size:]
    train_y, tune_y = y[:-tune_size], y[-tune_size:]

    def build_model(data_X, data_y, K):
        counts = [defaultdict(Counter) for _ in range(K + 1)]
        for ctx, tgt in zip(data_X, data_y):
            for L in range(1, K + 1):
                ngram = ctx[-L:]
                counts[L][ngram][tgt] += 1
        tables = []
        for L in range(1, K + 1):
            entries = []
            for ngram, ctr in counts[L].items():
                best = max(ctr.items(), key=lambda kv: (kv[1], -kv[0]))[0]
                entries.append((ngram, best))
            entries.sort(key=lambda x: x[0])
            tables.append(entries)
        # global fallback
        global_ctr = Counter(data_y)
        fallback = max(global_ctr.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        return tables, fallback

    def evaluate(tables, fallback, K, eval_X, eval_y):
        correct = 0
        for ctx, tgt in zip(eval_X, eval_y):
            pred = fallback
            for L in range(K, 0, -1):
                ngram = ctx[-L:]
                entries = tables[L - 1]
                lo, hi = 0, len(entries)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if entries[mid][0] < ngram:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < len(entries) and entries[lo][0] == ngram:
                    pred = entries[lo][1]
                    break
            if pred == tgt:
                correct += 1
        return correct / len(eval_X)

    best_K, best_acc = 3, -1.0
    for K in range(3, 7):
        tables, fallback = build_model(train_X, train_y, K)
        acc = evaluate(tables, fallback, K, tune_X, tune_y)
        if acc > best_acc:
            best_acc, best_K = acc, K

    # retrain on all data
    tables, fallback = build_model(X, y, best_K)

    # write weights.bin
    # header: magic(4), K(1), fallback(1), reserved(2)
    # for each L=1..K: num_entries(4), then entries: L bytes ngram + 1 byte pred
    magic = 0x4E47524D  # 'NGRM'
    out = bytearray()
    out += struct.pack("<I", magic)
    out += struct.pack("<BBH", best_K, fallback, 0)
    for L in range(1, best_K + 1):
        entries = tables[L - 1]
        out += struct.pack("<I", len(entries))
        for ngram, pred in entries:
            out += ngram
            out += bytes([pred])
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 8) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = w[0] | w[1] << 8 | w[2] << 16 | (uint32_t)w[3] << 24;
    if (magic != 0x4E47524D) { fprintf(stderr, "bad magic\n"); return 3; }
    int K = w[4];
    int fallback = w[5];
    if (K < 1 || K > 10) { fprintf(stderr, "bad K\n"); return 3; }
    const unsigned char *p = w + 8;
    struct Table { int L; int n; const unsigned char *data; } tables[10];
    long expected = 8;
    for (int L = 1; L <= K; L++) {
        if (p + 4 > w + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        int nent = p[0] | p[1] << 8 | p[2] << 16 | (int)p[3] << 24;
        p += 4;
        tables[L - 1].L = L;
        tables[L - 1].n = nent;
        tables[L - 1].data = p;
        p += (long)nent * (L + 1);
        expected += 4 + (long)nent * (L + 1);
    }
    if (expected != nw) { fprintf(stderr, "weights size mismatch\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        int pred = fallback;
        for (int L = K; L >= 1; L--) {
            const struct Table *t = &tables[L - 1];
            const unsigned char *key = xi + 16 - L;
            int lo = 0, hi = t->n;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                const unsigned char *entry = t->data + (long)mid * (L + 1);
                int cmp = memcmp(key, entry, L);
                if (cmp < 0) hi = mid;
                else if (cmp > 0) lo = mid + 1;
                else { pred = entry[L]; lo = hi; break; }
            }
            if (lo < t->n && memcmp(key, t->data + (long)lo * (L + 1), L) == 0) {
                pred = t->data[(long)lo * (L + 1) + L];
                break;
            }
        }
        out_byte = (unsigned char)pred;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```