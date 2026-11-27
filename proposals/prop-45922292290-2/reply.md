```hypothesis
Family: label-trigram. A trigram language model with add-k smoothing and linear interpolation over unigram, bigram, and trigram probabilities. It should excel because character-level next-char prediction is dominated by local n-gram statistics; the model is tiny (~0.6 MB), deterministic, and inference requires only a few array lookups and arithmetic per item. Failure would be dev accuracy ≤ label-linear (0.3375) or inference cost ≥ label-linear (7359 Ir/item).
```

```python
import json, math, struct, sys
import numpy as np

# deterministic grid search
SEED = 1
LAMBDA_GRID = []  # (l1, l2, l3)
for l3 in np.arange(0.0, 1.01, 0.1):
    for l2 in np.arange(0.0, 1.01 - l3, 0.1):
        l1 = 1.0 - l2 - l3
        LAMBDA_GRID.append((float(l1), float(l2), float(l3)))
K_GRID = [0.01, 0.1, 0.5, 1.0, 2.0]

MAGIC = 0x4E47524D  # 'NGRM'
VOCAB_SIZE = 65
CONTEXT_LEN = 2  # we use last 2 bytes of the 16-byte context

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    class_chars = card["class_chars"]
    # mapping from byte value (0-255) to class index 0-64
    byte_to_class = np.zeros(256, dtype=np.uint8)
    for idx, ch in enumerate(class_chars):
        byte_to_class[ord(ch)] = idx

    # read all training data
    X_bytes = records(sys.argv[2])  # list of 16-byte sequences
    y_bytes = records(sys.argv[3])  # list of 1-byte targets
    n = len(X_bytes)
    assert n == len(y_bytes) == 24000

    # convert to class indices
    X = np.zeros((n, 16), dtype=np.uint8)
    y = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        X[i] = byte_to_class[np.frombuffer(X_bytes[i], dtype=np.uint8)]
        y[i] = byte_to_class[y_bytes[i][0]]

    # split: last 20% for tuning
    tune_size = n // 5
    train_size = n - tune_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_tune, y_tune = X[train_size:], y[train_size:]

    # count n-grams on training portion
    unigram = np.zeros(VOCAB_SIZE, dtype=np.uint32)
    bigram = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.uint32)
    trigram = np.zeros((VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE), dtype=np.uint32)

    for i in range(train_size):
        c1, c2 = X_train[i, 14], X_train[i, 15]
        t = y_train[i]
        unigram[t] += 1
        bigram[c2, t] += 1
        trigram[c1, c2, t] += 1

    unigram_total = unigram.sum()
    bigram_totals = bigram.sum(axis=1)
    trigram_totals = trigram.sum(axis=2)

    # evaluate on tuning set
    best_acc = -1.0
    best_params = None
    for l1, l2, l3 in LAMBDA_GRID:
        for k in K_GRID:
            correct = 0
            for i in range(tune_size):
                c1, c2 = X_tune[i, 14], X_tune[i, 15]
                # compute scores for all classes
                scores = np.zeros(VOCAB_SIZE, dtype=np.float32)
                # trigram
                tri_counts = trigram[c1, c2]
                tri_total = trigram_totals[c1, c2]
                if tri_total > 0:
                    scores += l3 * (tri_counts + k) / (tri_total + VOCAB_SIZE * k)
                else:
                    scores += l3 / VOCAB_SIZE
                # bigram
                bi_counts = bigram[c2]
                bi_total = bigram_totals[c2]
                if bi_total > 0:
                    scores += l2 * (bi_counts + k) / (bi_total + VOCAB_SIZE * k)
                else:
                    scores += l2 / VOCAB_SIZE
                # unigram
                if unigram_total > 0:
                    scores += l1 * (unigram + k) / (unigram_total + VOCAB_SIZE * k)
                else:
                    scores += l1 / VOCAB_SIZE
                pred = np.argmax(scores)
                if pred == y_tune[i]:
                    correct += 1
            acc = correct / tune_size
            if acc > best_acc:
                best_acc = acc
                best_params = (l1, l2, l3, k)

    l1, l2, l3, k = best_params
    sys.stderr.write(f"Best params: l1={l1:.3f} l2={l2:.3f} l3={l3:.3f} k={k:.3f} tune_acc={best_acc:.4f}\n")

    # retrain on all data
    unigram = np.zeros(VOCAB_SIZE, dtype=np.uint32)
    bigram = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.uint32)
    trigram = np.zeros((VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE), dtype=np.uint32)

    for i in range(n):
        c1, c2 = X[i, 14], X[i, 15]
        t = y[i]
        unigram[t] += 1
        bigram[c2, t] += 1
        trigram[c1, c2, t] += 1

    unigram_total = unigram.sum()
    bigram_totals = bigram.sum(axis=1)
    trigram_totals = trigram.sum(axis=2)

    # convert to uint16 (max count 24000 < 65535)
    unigram = unigram.astype(np.uint16)
    bigram = bigram.astype(np.uint16)
    trigram = trigram.astype(np.uint16)
    unigram_total = np.array(unigram_total, dtype=np.uint16)
    bigram_totals = bigram_totals.astype(np.uint16)
    trigram_totals = trigram_totals.astype(np.uint16)

    # write weights.bin
    with open(sys.argv[4], "wb") as f:
        # header
        f.write(struct.pack("<iii", MAGIC, VOCAB_SIZE, CONTEXT_LEN))
        f.write(struct.pack("<ffff", l1, l2, l3, k))
        # mapping table (256 bytes)
        f.write(byte_to_class.tobytes())
        # counts
        f.write(unigram.tobytes())
        f.write(unigram_total.tobytes())
        f.write(bigram.tobytes())
        f.write(bigram_totals.tobytes())
        f.write(trigram.tobytes())
        f.write(trigram_totals.tobytes())

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
    unsigned char *wb = slurp(argv[1], &nw);
    unsigned char *p = wb;

    // parse header
    if (nw < 12) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t magic = p[0] | p[1] << 8 | p[2] << 16 | p[3] << 24; p += 4;
    if (magic != 0x4E47524D) { fprintf(stderr, "bad magic\n"); return 3; }
    int vocab_size = p[0] | p[1] << 8 | p[2] << 16 | p[3] << 24; p += 4;
    int context_len = p[0] | p[1] << 8 | p[2] << 16 | p[3] << 24; p += 4;
    if (vocab_size != 65 || context_len != 2) { fprintf(stderr, "bad vocab/context\n"); return 3; }

    float l1 = *(float*)p; p += 4;
    float l2 = *(float*)p; p += 4;
    float l3 = *(float*)p; p += 4;
    float k  = *(float*)p; p += 4;

    // mapping table
    unsigned char *map = p; p += 256;

    // counts (uint16)
    uint16_t *unigram = (uint16_t*)p; p += 65 * 2;
    uint16_t *unigram_total_ptr = (uint16_t*)p; p += 2;
    uint16_t *bigram = (uint16_t*)p; p += 65 * 65 * 2;
    uint16_t *bigram_totals = (uint16_t*)p; p += 65 * 2;
    uint16_t *trigram = (uint16_t*)p; p += 65 * 65 * 65 * 2;
    uint16_t *trigram_totals = (uint16_t*)p; p += 65 * 65 * 2;

    // validate size
    long expected = p - wb;
    if (nw != expected) { fprintf(stderr, "weights size mismatch: %ld vs %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[65];

    for (long i = 0; i < N; i++) {
        uint32_t klen; const unsigned char *xi = next_rec(x, nx, &pos, &klen);
        if (klen != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, klen); return 3; }

        // map last two bytes
        int c1 = map[xi[14]];
        int c2 = map[xi[15]];

        // compute scores
        for (int c = 0; c < 65; c++) {
            float s = 0.0f;
            // trigram
            uint32_t tri_idx = (c1 * 65 + c2) * 65 + c;
            uint16_t tri_cnt = trigram[tri_idx];
            uint16_t tri_tot = trigram_totals[c1 * 65 + c2];
            if (tri_tot > 0) {
                s += l3 * (tri_cnt + k) / (tri_tot + 65.0f * k);
            } else {
                s += l3 / 65.0f;
            }
            // bigram
            uint16_t bi_cnt = bigram[c2 * 65 + c];
            uint16_t bi_tot = bigram_totals[c2];
            if (bi_tot > 0) {
                s += l2 * (bi_cnt + k) / (bi_tot + 65.0f * k);
            } else {
                s += l2 / 65.0f;
            }
            // unigram
            uint16_t uni_cnt = unigram[c];
            uint16_t uni_tot = *unigram_total_ptr;
            if (uni_tot > 0) {
                s += l1 * (uni_cnt + k) / (uni_tot + 65.0f * k);
            } else {
                s += l1 / 65.0f;
            }
            scores[c] = s;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < 65; c++) if (scores[c] > scores[best]) best = c;
        put_rec(stdout, &best, 1);
    }

    free(wb);
    if (x) free(x);
    return 0;
}
```