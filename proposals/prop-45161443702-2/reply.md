```hypothesis
Family: label-ngram (variable-order n-gram with binary-search lookup).
Why: Character-level language modeling benefits from exact substring statistics; a longest-suffix match with backoff captures context dependencies that linear models miss. The model is compact (only observed n-grams stored) and inference is a few binary searches per item, so it should exceed label-linear's 0.3375 accuracy at lower CPU cost.
Failure signal: If dev accuracy ≤ 0.34 or instructions/item ≥ 7000, the approach does not dominate label-linear.
```

```python
"""label-ngram: variable-order n-gram with longest-suffix backoff.
train.py card.json train.in train.key weights.bin
Deterministic: no randomness, single thread, stdlib+numpy only.
Hyperparameter MIN_COUNT chosen on last 20% of train.in (tuning split)."""
import json, struct, sys
import numpy as np

# candidate MIN_COUNT values to try on the tuning split
MIN_COUNT_CANDIDATES = [1, 2, 3, 5, 10, 20]

MAGIC = b'NGRM'          # 4 bytes
VERSION = 1             # 1 byte
MAX_LEN = 16            # 1 byte (context length)
NUM_CLASSES = 65        # 1 byte (from card, but fixed for this task)


def read_records(path):
    """Return list of bytes objects from a record file (4-byte LE length + data)."""
    data = open(path, "rb").read()
    recs, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        recs.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return recs


def build_ngram_counts(X, y, min_count):
    """Build dict: suffix_bytes -> counts[65] (uint32). Only keep suffixes with total >= min_count."""
    counts_dict = {}
    n = X.shape[0]
    for i in range(n):
        ctx = X[i]           # 16 bytes
        target = int(y[i])   # 0..64
        # all suffixes of the context (length 1..16)
        for L in range(1, 17):
            suffix = bytes(ctx[16 - L:])   # last L bytes
            arr = counts_dict.get(suffix)
            if arr is None:
                arr = np.zeros(NUM_CLASSES, dtype=np.uint32)
                counts_dict[suffix] = arr
            arr[target] += 1
    # filter by min_count
    filtered = {}
    for suf, arr in counts_dict.items():
        if arr.sum() >= min_count:
            filtered[suf] = arr
    return filtered


def evaluate_ngram(ngram_dict, X, y):
    """Predict using longest suffix in ngram_dict; return accuracy."""
    correct = 0
    n = X.shape[0]
    for i in range(n):
        ctx = X[i]
        target = int(y[i])
        pred = -1
        for L in range(16, 0, -1):
            suffix = bytes(ctx[16 - L:])
            arr = ngram_dict.get(suffix)
            if arr is not None:
                pred = int(arr.argmax())
                break
        if pred == -1:   # should not happen if global unigram is present (L=1 always present if min_count=1)
            pred = 0
        if pred == target:
            correct += 1
    return correct / n


def write_weights(ngram_dict, global_counts, min_count, out_path):
    """Serialize ngram_dict to binary format."""
    # Group by length
    by_len = {L: [] for L in range(1, MAX_LEN + 1)}
    for suf, arr in ngram_dict.items():
        by_len[len(suf)].append((suf, arr))
    # Sort each length lexicographically by suffix bytes
    for L in range(1, MAX_LEN + 1):
        by_len[L].sort(key=lambda x: x[0])

    # Compute section sizes
    section_counts = [len(by_len[L]) for L in range(1, MAX_LEN + 1)]
    entry_sizes = [L + 2 * NUM_CLASSES for L in range(1, MAX_LEN + 1)]  # bytes + uint16[65]

    # Header: magic(4), version(1), num_classes(1), max_len(1), min_count(2), 16*uint32 section_counts
    header = bytearray()
    header += MAGIC
    header += struct.pack("<B", VERSION)
    header += struct.pack("<B", NUM_CLASSES)
    header += struct.pack("<B", MAX_LEN)
    header += struct.pack("<H", min_count)
    for cnt in section_counts:
        header += struct.pack("<I", cnt)

    # Body: for each length, each entry: suffix bytes (L) + counts[65] as uint16
    body = bytearray()
    for L in range(1, MAX_LEN + 1):
        for suf, arr in by_len[L]:
            body += suf
            # clamp to uint16 max (65535) - training set size 24000 so safe
            body += arr.astype(np.uint16).tobytes()

    # Global counts (unigram) as uint16[65]
    body += global_counts.astype(np.uint16).tobytes()

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(body)


def main():
    card_path, train_in, train_key, weights_out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    card = json.load(open(card_path))
    # Sanity checks
    assert card["input"]["bytes"] == MAX_LEN
    assert card["classes"] == NUM_CLASSES
    for t in card["input"]["types"]:
        assert t == "categorical"

    # Read data
    X_recs = read_records(train_in)
    y_recs = read_records(train_key)
    N = len(X_recs)
    assert len(y_recs) == N
    X = np.frombuffer(b"".join(X_recs), dtype=np.uint8).reshape(N, MAX_LEN)
    y = np.frombuffer(b"".join(y_recs), dtype=np.uint8).reshape(N)

    # Global unigram counts (used as final backoff and for L=1 section)
    global_counts = np.zeros(NUM_CLASSES, dtype=np.uint32)
    for target in y:
        global_counts[target] += 1

    # Split for tuning: last 20% in file order
    split = int(0.8 * N)
    X_tr, y_tr = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    best_min_count = 1
    best_acc = -1.0
    for mc in MIN_COUNT_CANDIDATES:
        ngram_dict = build_ngram_counts(X_tr, y_tr, mc)
        # Ensure global unigram is present for L=1 (it will be if min_count=1, but for safety)
        # Actually build_ngram_counts includes all suffixes, so L=1 suffixes are there if they meet min_count.
        # If min_count > 1, some length-1 suffixes might be filtered out. That's okay; we fall back to global_counts in inference.
        acc = evaluate_ngram(ngram_dict, X_tune, y_tune)
        if acc > best_acc:
            best_acc = acc
            best_min_count = mc

    # Rebuild on full data with best_min_count
    final_ngram_dict = build_ngram_counts(X, y, best_min_count)
    write_weights(final_ngram_dict, global_counts, best_min_count, weights_out)


if __name__ == "__main__":
    main()
```

```c
/* label-ngram inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes context. Output record: 1 byte class index (0..64).
 * Weights format: see train.py. Binary search per length (16 down to 1). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(3);
}

static void die_errno(const char *msg) {
    perror(msg);
    exit(3);
}

static unsigned char *slurp(const char *path, long *size) {
    FILE *f = fopen(path, "rb");
    if (!f) die_errno(path);
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*size ? *size : 1);
    if (fread(buf, 1, *size, f) != (size_t)*size) die_errno("fread weights");
    fclose(f);
    return buf;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) die("truncated input");
    uint32_t k = buf[*pos] | (buf[*pos+1] << 8) | (buf[*pos+2] << 16) | ((uint32_t)buf[*pos+3] << 24);
    if (*pos + 4 + (long)k > n) die("truncated input");
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(h, 1, 4, f);
    fwrite(r, 1, k, f);
}

/* Binary search in a sorted array of n-grams for a given length.
 * base: pointer to first entry (each entry = L bytes key + 130 bytes counts)
 * count: number of entries
 * entry_size: L + 130
 * key: L bytes to search for
 * Returns pointer to counts (uint16_t[65]) if found, else NULL. */
static const uint16_t *bsearch_ngram(const unsigned char *base, uint32_t count,
                                     uint32_t entry_size, int L,
                                     const unsigned char *key) {
    uint32_t lo = 0, hi = count;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        const unsigned char *entry_key = base + mid * entry_size;
        int cmp = memcmp(key, entry_key, L);
        if (cmp == 0) {
            return (const uint16_t *)(entry_key + L);
        } else if (cmp < 0) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: infer weights.bin in.rec N\n");
        return 2;
    }
    long wsize, isize = 0;
    unsigned char *wbuf = slurp(argv[1], &wsize);
    unsigned char *ibuf = argv[3][0] == '0' ? NULL : slurp(argv[2], &isize);
    long N = atol(argv[3]);
    long ipos = 0;

    /* Parse header */
    const unsigned char *p = wbuf;
    if (wsize < 4 + 1 + 1 + 1 + 2 + 16*4) die("weights too small for header");
    if (memcmp(p, "NGRM", 4) != 0) die("bad magic");
    p += 4;
    uint8_t version = *p++;  (void)version;
    uint8_t num_classes = *p++;  if (num_classes != 65) die("num_classes != 65");
    uint8_t max_len = *p++;      if (max_len != 16) die("max_len != 16");
    uint16_t min_count = p[0] | (p[1] << 8); p += 2; (void)min_count;

    uint32_t section_counts[16];
    for (int i = 0; i < 16; i++) {
        section_counts[i] = p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24);
        p += 4;
    }

    /* Compute base pointers for each length */
    const unsigned char *section_base[16];
    uint32_t entry_size[16];
    for (int L = 1; L <= 16; L++) {
        section_base[L-1] = p;
        entry_size[L-1] = L + 2 * 65;  // key bytes + uint16[65]
        p += (long)section_counts[L-1] * entry_size[L-1];
    }
    const uint16_t *global_counts = (const uint16_t *)p;
    /* Optional: validate we consumed exactly wsize */

    unsigned char ctx[16];
    unsigned char pred_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k;
        const unsigned char *rec = next_rec(ibuf, isize, &ipos, &k);
        if (k != 16) die("input record not 16 bytes");
        memcpy(ctx, rec, 16);

        int found = 0;
        for (int L = 16; L >= 1; L--) {
            uint32_t cnt = section_counts[L-1];
            if (cnt == 0) continue;
            const unsigned char *key = ctx + 16 - L;  // last L bytes
            const uint16_t *counts = bsearch_ngram(section_base[L-1], cnt, entry_size[L-1], L, key);
            if (counts) {
                /* argmax over 65 classes */
                unsigned best = 0;
                uint16_t best_val = counts[0];
                for (int c = 1; c < 65; c++) {
                    if (counts[c] > best_val) {
                        best_val = counts[c];
                        best = c;
                    }
                }
                pred_byte = (unsigned char)best;
                found = 1;
                break;
            }
        }
        if (!found) {
            /* fallback to global unigram */
            unsigned best = 0;
            uint16_t best_val = global_counts[0];
            for (int c = 1; c < 65; c++) {
                if (global_counts[c] > best_val) {
                    best_val = global_counts[c];
                    best = c;
                }
            }
            pred_byte = (unsigned char)best;
        }
        put_rec(stdout, &pred_byte, 1);
    }
    return 0;
}
```