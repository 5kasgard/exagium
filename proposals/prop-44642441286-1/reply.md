```hypothesis
Family: label-adaboost (AdaBoost with decision stumps, SAMME algorithm).
Why: Optdigits pixels are ordinal 0..16; stumps on single pixels capture local contrast patterns. Boosting combines many weak learners for strong non-linear boundaries with low inference cost (one comparison per stump). Expected to beat label-linear (0.9613) with similar or lower CPU cost.
Failure: If dev accuracy ≤ 0.9613 or instructions/item > 5000, the model does not improve on label-linear.
```

```python
"""AdaBoost (SAMME) with decision stumps for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Tunes number of stumps on last 20% of train.in."""
import json
import math
import struct
import sys

import numpy as np

SEED = 1
N_STUMPS_CANDIDATES = [50, 100, 150, 200, 250]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_samme(X, y, n_stumps, C):
    """Train SAMME with decision stumps. Returns list of stumps."""
    n, d = X.shape
    w = np.ones(n, dtype=np.float64) / n
    stumps = []
    total_weight = n

    for _ in range(n_stumps):
        best_err = 1.0
        best_f = -1
        best_t = -1
        best_left_class = 0
        best_right_class = 0

        # For each feature, compute weighted class counts per pixel value (0..16)
        for f in range(d):
            # counts[value][class] = sum of weights
            counts = np.zeros((17, C), dtype=np.float64)
            # vectorized: for each class, bincount with weights
            for c in range(C):
                mask = (y == c)
                if np.any(mask):
                    vals = X[mask, f]
                    w_mask = w[mask]
                    # bincount up to 16
                    counts[:17, c] += np.bincount(vals, weights=w_mask, minlength=17)[:17]

            cum_counts = np.cumsum(counts, axis=0)  # cum_counts[t] = sum for values <= t
            total_counts = cum_counts[-1]

            for t in range(16):  # threshold t means split <= t
                left_counts = cum_counts[t]
                right_counts = total_counts - left_counts
                left_w = left_counts.sum()
                right_w = right_counts.sum()
                if left_w == 0 or right_w == 0:
                    continue
                left_class = int(left_counts.argmax())
                right_class = int(right_counts.argmax())
                correct = left_counts[left_class] + right_counts[right_class]
                err = (total_weight - correct) / total_weight
                if err < best_err:
                    best_err = err
                    best_f = f
                    best_t = t
                    best_left_class = left_class
                    best_right_class = right_class

        if best_f == -1 or best_err >= 0.5:
            # no good stump found or error too high, stop early
            break

        # compute alpha
        if best_err == 0:
            alpha = 1.0
        else:
            alpha = math.log((1 - best_err) / best_err) + math.log(C - 1)

        # update weights
        pred = np.where(X[:, best_f] <= best_t, best_left_class, best_right_class)
        w *= np.exp(alpha * (pred != y))
        w_sum = w.sum()
        if w_sum == 0:
            break
        w /= w_sum
        total_weight = w_sum

        stumps.append((best_f, best_t, best_left_class, best_right_class, alpha))

    return stumps


def evaluate(X, y, stumps, C):
    n = X.shape[0]
    scores = np.zeros((n, C), dtype=np.float64)
    for f, t, lc, rc, alpha in stumps:
        left = X[:, f] <= t
        scores[left, lc] += alpha
        scores[~left, rc] += alpha
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]

    # tuning split: last 20% of train.in in file order
    n = X.shape[0]
    n_val = int(0.2 * n)
    n_train = n - n_val
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    best_acc = -1.0
    best_stumps = None
    best_n = N_STUMPS_CANDIDATES[0]

    for n_stumps in N_STUMPS_CANDIDATES:
        stumps = train_samme(X_train, y_train, n_stumps, C)
        acc = evaluate(X_val, y_val, stumps, C)
        if acc > best_acc:
            best_acc = acc
            best_stumps = stumps
            best_n = n_stumps

    # retrain on full data with best_n
    final_stumps = train_samme(X, y, best_n, C)

    # write weights.bin
    # header: magic, n_stumps, n_classes, n_features
    magic = 0x41444142  # 'ADAB'
    n_stumps = len(final_stumps)
    out = struct.pack("<4I", magic, n_stumps, C, d)
    for f, t, lc, rc, alpha in final_stumps:
        out += struct.pack("<BBBBf", f, t, lc, rc, np.float32(alpha))
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* AdaBoost (SAMME) with decision stumps for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x41444142) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t n_stumps = hd[1];
    uint32_t C = hd[2];
    uint32_t d = hd[3];
    if (C < 1 || C > 256 || d > 255) { fprintf(stderr, "bad header\n"); return 3; }

    /* each stump: uint8 feature, threshold, left_class, right_class; float32 alpha */
    size_t expected = 16 + n_stumps * (4 + 4);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *stump_bytes = (unsigned char *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = calloc(C, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        memset(scores, 0, C * sizeof(float));
        unsigned char *p = stump_bytes;
        for (uint32_t s = 0; s < n_stumps; s++) {
            uint8_t f = p[0];
            uint8_t t = p[1];
            uint8_t lc = p[2];
            uint8_t rc = p[3];
            float alpha = *(float *)(p + 4);
            p += 8;
            if (xi[f] <= t) scores[lc] += alpha;
            else scores[rc] += alpha;
        }
        unsigned char best = 0;
        for (uint32_t c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```