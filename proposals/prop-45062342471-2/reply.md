```hypothesis
We propose a "label-ngram" family: a variable-order n-gram model with add-1 smoothing and backoff. It stores counts for all suffixes (lengths 0..16) of the 16-byte context seen in training. At inference, it uses the longest matching context to predict the next character. This captures exact n-gram statistics which are highly predictive for text, likely beating the linear model's 0.3375 accuracy. If it fails to beat linear, it would show that the training data is too sparse for n-grams to generalize better than a linear model.
```

```python
"""label-ngram: variable-order n-gram with add-1 smoothing and backoff.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic, single-threaded, stdlib + numpy only."""
import json, struct, sys
import numpy as np

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    contexts = records(sys.argv[2])     # list of 16-byte contexts
    targets = [r[0] for r in records(sys.argv[3])]  # class indices

    # dict: (length, context_bytes) -> np.array of 65 int64 counts
    counts = {}
    # unigram (length 0)
    counts[(0, b"")] = np.zeros(C, dtype=np.int64)

    for ctx, tgt in zip(contexts, targets):
        # all suffixes including full context and empty
        for L in range(d + 1):
            suffix = ctx[d - L:]  # last L bytes
            key = (L, suffix)
            if key not in counts:
                counts[key] = np.zeros(C, dtype=np.int64)
            counts[key][tgt] += 1
        # also update unigram (already done when L=0)

    # write weights.bin
    # magic "NGRM", version 1, max_length 16
    out = bytearray()
    out += b"NGRM"
    out += struct.pack("<BB", 1, d)  # version, max_length

    # for each length 0..d, write entries sorted by context
    for L in range(d + 1):
        entries = [(ctx, cnt) for (ln, ctx), cnt in counts.items() if ln == L]
        entries.sort(key=lambda x: x[0])  # sort by context bytes
        out += struct.pack("<I", len(entries))
        for ctx, cnt in entries:
            out += ctx                     # L bytes
            out += cnt.astype("<i4").tobytes()  # 65 int32

    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-ngram inference: ./infer weights.bin in.rec N
 * Input records: 16 bytes each. Output: 1 byte class index.
 * Uses longest matching suffix with add-1 smoothing (argmax of count+1). */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    const unsigned char *ctx;   // pointer into weights buffer
    const int32_t *counts;      // pointer into weights buffer
} Entry;

static int cmp_ctx(const void *a, const void *b, void *keylen) {
    int L = *(int *)keylen;
    return memcmp(((const Entry *)a)->ctx, ((const Entry *)b)->ctx, L);
}

static int cmp_key_ctx(const void *key, const void *elem, void *keylen) {
    int L = *(int *)keylen;
    return memcmp(key, ((const Entry *)elem)->ctx, L);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wbuf = slurp(argv[1], &nw);
    if (nw < 6 || memcmp(wbuf, "NGRM", 4) != 0) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int version = wbuf[4];
    int max_len = wbuf[5];
    if (version != 1 || max_len != 16) { fprintf(stderr, "bad version/max_len\n"); return 3; }

    const unsigned char *p = wbuf + 6;
    const unsigned char *end = wbuf + nw;

    // arrays for each length 0..16
    Entry *entries[17] = {0};
    int n_entries[17] = {0};

    for (int L = 0; L <= max_len; L++) {
        if (p + 4 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint32_t cnt = p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24;
        p += 4;
        n_entries[L] = cnt;
        if (cnt == 0) continue;
        entries[L] = malloc(cnt * sizeof(Entry));
        for (uint32_t i = 0; i < cnt; i++) {
            if (p + L > end) { fprintf(stderr, "truncated weights context\n"); return 3; }
            entries[L][i].ctx = p;
            p += L;
            if (p + 65 * 4 > end) { fprintf(stderr, "truncated weights counts\n"); return 3; }
            entries[L][i].counts = (const int32_t *)p;
            p += 65 * 4;
        }
        // entries are already sorted by context from training
    }

    unsigned char *xbuf = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(xbuf, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        int best_class = 0;
        // search from longest suffix to shortest
        for (int L = 16; L >= 0; L--) {
            if (n_entries[L] == 0) continue;
            const unsigned char *suffix = xi + (16 - L);
            int keylen = L;
            Entry *found = bsearch(suffix, entries[L], n_entries[L], sizeof(Entry), cmp_key_ctx, &keylen);
            if (found) {
                // add-1 smoothing: maximize count+1
                int32_t best_cnt = -1;
                for (int c = 0; c < 65; c++) {
                    int32_t cnt = found->counts[c] + 1;
                    if (cnt > best_cnt) { best_cnt = cnt; best_class = c; }
                }
                break;
            }
        }
        out_byte = (unsigned char)best_class;
        put_rec(stdout, &out_byte, 1);
    }

    // free entries arrays
    for (int L = 0; L <= max_len; L++) free(entries[L]);
    free(wbuf);
    if (xbuf) free(xbuf);
    return 0;
}
```