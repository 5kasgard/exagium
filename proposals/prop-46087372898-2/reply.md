```hypothesis
Family: label-ngram (variable-order Markov with backoff).  
Why: Character-level language modeling benefits from high-order n-grams; the 16-byte context is short enough that a compact trie of observed suffixes (orders 1–5) captures strong local patterns that linear models miss. Backoff to shorter contexts handles sparsity.  
Failure signal: If dev accuracy ≤ 0.34 (label-linear) or inference cost exceeds label-linear by >2×, the family does not improve the Pareto frontier.
```

```python
"""label-ngram: variable-order Markov model with backoff for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Builds n-gram counts (orders 1..MAX_ORDER) on training data, selects best order on tuning split (last 20%),
then writes a compact weights file: magic, max_order, byte->class map, global counts, and for each order
a sorted array of (context_code, uint16 counts[65]). Contexts encoded in base 65.
Deterministic, single-threaded, stdlib+numpy only."""
import json, struct, sys
import numpy as np

MAX_ORDER_CANDIDATE = 10
SEED = 1

def read_records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    class_chars = card["class_chars"]
    # mapping from byte value (0..255) to class index 0..64, 255 for unknown
    byte2class = np.full(256, 255, dtype=np.uint8)
    for idx, ch in enumerate(class_chars):
        byte2class[ord(ch)] = idx

    X_recs = read_records(sys.argv[2])
    y_recs = read_records(sys.argv[3])
    n = len(X_recs)
    assert len(y_recs) == n

    # map to class indices
    X = np.zeros((n, d), dtype=np.uint8)
    y = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        xb = X_recs[i]
        yb = y_recs[i]
        assert len(xb) == d and len(yb) == 1
        X[i] = byte2class[np.frombuffer(xb, dtype=np.uint8)]
        y[i] = byte2class[yb[0]]
        assert np.all(X[i] != 255) and y[i] != 255

    # split: last 20% for tuning
    split = n - n // 5
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # precompute powers of 65 for encoding
    pow65 = [1]
    for _ in range(MAX_ORDER_CANDIDATE):
        pow65.append(pow65[-1] * 65)

    def build_counts(X_sub, y_sub, max_order):
        """return list of dicts: counts[order][code] = array of 65 uint16"""
        counts = [{} for _ in range(max_order + 1)]
        # order 0
        global_cnt = np.zeros(65, dtype=np.uint32)
        for i in range(len(X_sub)):
            global_cnt[y_sub[i]] += 1
            for L in range(1, max_order + 1):
                code = 0
                for j in range(d - L, d):
                    code = code * 65 + X_sub[i, j]
                if code not in counts[L]:
                    counts[L][code] = np.zeros(65, dtype=np.uint16)
                counts[L][code][y_sub[i]] += 1
        counts[0] = global_cnt
        return counts

    def evaluate(counts, max_order, X_sub, y_sub):
        correct = 0
        for i in range(len(X_sub)):
            pred = None
            for L in range(max_order, 0, -1):
                code = 0
                for j in range(d - L, d):
                    code = code * 65 + X_sub[i, j]
                if code in counts[L]:
                    cnt = counts[L][code]
                    pred = int(np.argmax(cnt))
                    break
            if pred is None:
                pred = int(np.argmax(counts[0]))
            if pred == y_sub[i]:
                correct += 1
        return correct / len(X_sub)

    best_order, best_acc = 1, -1.0
    for order in range(1, MAX_ORDER_CANDIDATE + 1):
        cnt = build_counts(X_tr, y_tr, order)
        acc = evaluate(cnt, order, X_te, y_te)
        if acc > best_acc:
            best_acc, best_order = acc, order

    # rebuild on full data with best_order
    final_counts = build_counts(X, y, best_order)

    # write weights.bin
    with open(sys.argv[4], "wb") as f:
        # magic: 'NGRM' = 0x4D52474E little-endian
        f.write(struct.pack("<I", 0x4D52474E))
        f.write(struct.pack("<B", best_order))
        f.write(byte2class.tobytes())  # 256 bytes
        # order 0: global counts (uint32)
        f.write(final_counts[0].astype("<u4").tobytes())
        # orders 1..best_order
        for L in range(1, best_order + 1):
            items = sorted(final_counts[L].items())
            f.write(struct.pack("<I", len(items)))
            for code, cnt_arr in items:
                f.write(struct.pack("<Q", code))  # uint64 context code
                f.write(cnt_arr.astype("<u2").tobytes())  # 65 x uint16

if __name__ == "__main__":
    main()
```

```c
/* label-ngram inference: ./infer weights.bin in.rec N
 * Reads n-gram model with backoff, predicts next class for each 16-byte context.
 * Weights format: magic(4), max_order(1), map[256], global_counts[65](u32),
 *   for each order: num_entries(u32), then entries: code(u64), counts[65](u16). */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

typedef struct { uint64_t code; uint16_t counts[65]; } Entry;

static int cmp_entry(const void *a, const void *b) {
    uint64_t ca = ((const Entry*)a)->code, cb = ((const Entry*)b)->code;
    return (ca > cb) - (ca < cb);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    if (nw < 5 + 256 + 65*4) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = w[0] | w[1]<<8 | w[2]<<16 | (uint32_t)w[3]<<24;
    if (magic != 0x4D52474E) { fprintf(stderr, "bad magic\n"); return 3; }
    uint8_t max_order = w[4];
    const uint8_t *map = w + 5;
    const uint32_t *global_counts = (const uint32_t*)(w + 5 + 256);
    const unsigned char *p = w + 5 + 256 + 65*4;

    Entry **orders = calloc(max_order + 1, sizeof(Entry*));
    uint32_t *order_sizes = calloc(max_order + 1, sizeof(uint32_t));
    for (int L = 1; L <= max_order; L++) {
        if (p + 4 > w + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint32_t sz = p[0] | p[1]<<8 | p[2]<<16 | (uint32_t)p[3]<<24;
        p += 4;
        if (p + (long)sz * (8 + 65*2) > w + nw) { fprintf(stderr, "truncated weights\n"); return 3; }
        orders[L] = malloc(sz * sizeof(Entry));
        order_sizes[L] = sz;
        for (uint32_t i = 0; i < sz; i++) {
            orders[L][i].code = *(const uint64_t*)p; p += 8;
            memcpy(orders[L][i].counts, p, 65*2); p += 65*2;
        }
        qsort(orders[L], sz, sizeof(Entry), cmp_entry);
    }

    uint8_t mapped[16];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        for (int j = 0; j < 16; j++) {
            uint8_t c = map[xi[j]];
            if (c == 255) { fprintf(stderr, "unknown byte %d at pos %d\n", xi[j], j); return 3; }
            mapped[j] = c;
        }
        int pred = -1;
        for (int L = max_order; L >= 1; L--) {
            uint64_t code = 0;
            for (int j = 16 - L; j < 16; j++) code = code * 65 + mapped[j];
            Entry key = {.code = code};
            Entry *found = bsearch(&key, orders[L], order_sizes[L], sizeof(Entry), cmp_entry);
            if (found) {
                uint16_t best_cnt = 0; int best_cls = 0;
                for (int c = 0; c < 65; c++) {
                    if (found->counts[c] > best_cnt) { best_cnt = found->counts[c]; best_cls = c; }
                }
                pred = best_cls; break;
            }
        }
        if (pred == -1) {
            uint32_t best_cnt = 0; int best_cls = 0;
            for (int c = 0; c < 65; c++) {
                if (global_counts[c] > best_cnt) { best_cnt = global_counts[c]; best_cls = c; }
            }
            pred = best_cls;
        }
        put_rec(stdout, (unsigned char*)&pred, 1);
    }
    return 0;
}
```