```hypothesis
Family: label-ngram (variable-order Markov with backoff).  
Why: Text has strong local dependencies; exact n-gram matches with backoff to shorter contexts capture sequential structure better than independent position features.  
Failure: If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds label-linear significantly.
```

```python
import json, struct, sys
from collections import Counter

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = records(sys.argv[2])
    y = [r[0] for r in records(sys.argv[3])]
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65

    # Overall majority class for fallback
    overall_counter = Counter(y)
    majority_class = min(overall_counter.items(), key=lambda x: (-x[1], x[0]))[0]

    # Build context -> Counter for each length 1..16
    context_counters = [Counter() for _ in range(d + 1)]
    for ctx_bytes, target in zip(X, y):
        for L in range(1, d + 1):
            context = ctx_bytes[-L:]
            context_counters[L][context][target] += 1

    # For each length, determine majority class per context and sort
    entries_by_len = []
    for L in range(1, d + 1):
        entries = []
        for context, counter in context_counters[L].items():
            best_class = min(counter.items(), key=lambda x: (-x[1], x[0]))[0]
            entries.append((context, best_class))
        entries.sort(key=lambda x: x[0])
        entries_by_len.append(entries)

    # Write weights.bin
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<I", 0x4D52474E))  # 'NGRM' little-endian
        f.write(bytes([majority_class]))
        for L in range(1, d + 1):
            entries = entries_by_len[L - 1]
            f.write(struct.pack("<I", len(entries)))
            for context, cls in entries:
                f.write(context)
                f.write(bytes([cls]))

if __name__ == "__main__":
    main()
```

```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static int cmp_context(const unsigned char *a, const unsigned char *b, int len) {
    return memcmp(a, b, len);
}

static int binary_search(const unsigned char *data, int entry_size, int count,
                         const unsigned char *key, int key_len) {
    int lo = 0, hi = count - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        const unsigned char *entry = data + mid * entry_size;
        int c = cmp_context(entry, key, key_len);
        if (c == 0) return entry[key_len];
        if (c < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

static void put_rec(FILE *f, unsigned char byte) {
    unsigned char h[4] = {1, 0, 0, 0};
    fwrite(h, 1, 4, f);
    fwrite(&byte, 1, 1, f);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: infer weights.bin in.rec N\n");
        return 2;
    }
    long N = atol(argv[3]);
    if (N == 0) return 0;

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("weights.bin"); return 3; }
    struct stat st;
    fstat(fd, &st);
    size_t wsize = st.st_size;
    unsigned char *wdata = mmap(NULL, wsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (wdata == MAP_FAILED) { perror("mmap"); return 3; }
    close(fd);

    if (wsize < 5) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = *(uint32_t*)wdata;
    if (magic != 0x4D52474E) { fprintf(stderr, "bad magic\n"); return 3; }
    unsigned char majority = wdata[4];
    unsigned char *ptr = wdata + 5;
    unsigned char *end = wdata + wsize;

    struct { unsigned char *data; int count; int entry_size; } lengths[16];
    for (int L = 1; L <= 16; L++) {
        if (ptr + 4 > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        uint32_t count = *(uint32_t*)ptr;
        ptr += 4;
        int entry_size = L + 1;
        if (ptr + (long)count * entry_size > end) { fprintf(stderr, "truncated weights\n"); return 3; }
        lengths[L-1].data = ptr;
        lengths[L-1].count = count;
        lengths[L-1].entry_size = entry_size;
        ptr += (long)count * entry_size;
    }
    if (ptr != end) { fprintf(stderr, "extra data in weights\n"); return 3; }

    fd = open(argv[2], O_RDONLY);
    if (fd < 0) { perror("in.rec"); return 3; }
    fstat(fd, &st);
    size_t isize = st.st_size;
    unsigned char *idata = mmap(NULL, isize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (idata == MAP_FAILED) { perror("mmap"); return 3; }
    close(fd);

    unsigned char *ipos = idata, *iend = idata + isize;
    for (long n = 0; n < N; n++) {
        if (ipos + 4 > iend) { fprintf(stderr, "truncated input\n"); return 3; }
        uint32_t k = ipos[0] | (ipos[1]<<8) | (ipos[2]<<16) | ((uint32_t)ipos[3]<<24);
        ipos += 4;
        if (ipos + k > iend) { fprintf(stderr, "truncated input\n"); return 3; }
        if (k != 16) { fprintf(stderr, "record %ld is %u bytes, not 16\n", n, k); return 3; }
        unsigned char *record = ipos;
        ipos += k;

        unsigned char predicted = majority;
        for (int L = 16; L >= 1; L--) {
            int cls = binary_search(lengths[L-1].data, lengths[L-1].entry_size,
                                    lengths[L-1].count, record + 16 - L, L);
            if (cls >= 0) { predicted = (unsigned char)cls; break; }
        }
        put_rec(stdout, predicted);
    }

    munmap(wdata, wsize);
    munmap(idata, isize);
    return 0;
}
```