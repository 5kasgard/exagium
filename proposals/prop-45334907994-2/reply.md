```hypothesis
Family: label-suffix (variable-order Markov model). For each training example, all suffixes of the 16-byte context (lengths 1–16) are counted with the next character. At inference, the longest matching suffix determines the prediction (backing off to shorter suffixes, then global majority). This captures n-gram statistics directly, which are known to be strong for character-level language modeling. It has no hyperparameters, trains in one pass, and inference is ≤16 binary searches over compact sorted tables. It should beat label-linear (0.3375) because text has nonlinear context interactions; if it doesn't, the signal is mostly per-position linear.
```

```python
import json, struct, sys
from array import array
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
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    global_counts = np.zeros(C, dtype=np.uint32)
    suffix_counts = {}  # key: bytes, value: array('H', C)

    for i in range(n):
        ctx = X[i].tobytes()
        nxt = int(y[i])
        global_counts[nxt] += 1
        for L in range(1, d + 1):
            suf = ctx[d - L:]
            arr = suffix_counts.get(suf)
            if arr is None:
                arr = array('H', [0]) * C
                suffix_counts[suf] = arr
            arr[nxt] += 1

    global_majority = int(np.argmax(global_counts))

    buckets = [[] for _ in range(d + 1)]
    for suf, arr in suffix_counts.items():
        L = len(suf)
        best = 0
        best_cnt = arr[0]
        for c in range(1, C):
            cnt = arr[c]
            if cnt > best_cnt:
                best_cnt = cnt
                best = c
        buckets[L].append((suf, best))

    for L in range(1, d + 1):
        buckets[L].sort(key=lambda x: x[0])

    magic = 0x46465553  # 'SUFF' little-endian
    counts = [len(buckets[L]) for L in range(1, d + 1)]
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<I B 3x 16I", magic, global_majority, *counts))
        for L in range(1, d + 1):
            for suf, best in buckets[L]:
                f.write(suf)
                f.write(bytes([best]))

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
    if (nw < 4) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t magic = w[0] | w[1] << 8 | w[2] << 16 | (uint32_t)w[3] << 24;
    if (magic != 0x46465553) { fprintf(stderr, "bad weights: magic\n"); return 3; }
    if (nw < 4 + 1 + 3 + 16 * 4) { fprintf(stderr, "bad weights: header too small\n"); return 3; }
    unsigned char global_majority = w[4];
    uint32_t counts[16];
    for (int i = 0; i < 16; i++) {
        int off = 4 + 1 + 3 + i * 4;
        counts[i] = w[off] | w[off+1] << 8 | w[off+2] << 16 | (uint32_t)w[off+3] << 24;
    }
    const unsigned char *data[17];
    size_t header_size = 4 + 1 + 3 + 16 * 4;
    const unsigned char *ptr = w + header_size;
    for (int L = 1; L <= 16; L++) {
        data[L] = ptr;
        ptr += (size_t)counts[L - 1] * (L + 1);
    }
    if (ptr - w != nw) { fprintf(stderr, "bad weights: size mismatch\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        unsigned char best = global_majority;
        int found = 0;
        for (int L = 16; L >= 1; L--) {
            const unsigned char *suffix = xi + 16 - L;
            int lo = 0, hi = counts[L - 1];
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                const unsigned char *entry = data[L] + (size_t)mid * (L + 1);
                int cmp = memcmp(suffix, entry, L);
                if (cmp == 0) {
                    best = entry[L];
                    found = 1;
                    break;
                } else if (cmp < 0) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            if (found) break;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```