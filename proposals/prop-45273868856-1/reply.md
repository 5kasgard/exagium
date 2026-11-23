```hypothesis
We propose "label-adaboost": AdaBoost with decision stumps (depth-1 trees) on the 64 ordinal pixel features (0–16).  
Stumps are weak learners that split on a single pixel threshold; AdaBoost combines them with weighted voting.  
This family often beats linear models on small ordinal-image tasks because axis-aligned splits capture local intensity patterns, while inference stays ultra-fast (one comparison per stump).  
We tune the number of stumps on the last 20% of train.in, then retrain on all data.  
Failure: dev accuracy ≤ 0.9613 (label-linear) or instructions/item > 2× label-linear.
```

```python
"""AdaBoost with decision stumps for label tasks with fixed-size ordinal byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
CANDIDATE_T = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # stump counts to try on tuning split
MAGIC = 0x41444142  # 'ADAB'


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def find_best_stump(X, y, weights, C):
    """Exhaustive search over all features (0..63) and thresholds (0..15)."""
    n, d = X.shape
    best_err = 1.0
    best_feat = 0
    best_thresh = 0
    best_left_class = 0
    best_right_class = 0
    for feat in range(d):
        vals = X[:, feat]
        for thresh in range(16):  # pixel values 0..16, threshold splits at 0..15
            left = vals <= thresh
            right = ~left
            if not left.any() or not right.any():
                continue
            # left group
            w_left = weights[left]
            y_left = y[left]
            sums_left = np.bincount(y_left, weights=w_left, minlength=C)
            left_class = int(np.argmax(sums_left))
            # right group
            w_right = weights[right]
            y_right = y[right]
            sums_right = np.bincount(y_right, weights=w_right, minlength=C)
            right_class = int(np.argmax(sums_right))
            # error
            pred = np.where(left, left_class, right_class)
            err = float(np.sum(weights * (pred != y)))
            if err < best_err:
                best_err = err
                best_feat = feat
                best_thresh = thresh
                best_left_class = left_class
                best_right_class = right_class
    return best_feat, best_thresh, best_left_class, best_right_class, best_err


def train_adaboost(X, y, T, C):
    """Train AdaBoost (SAMME) for up to T stumps. Returns list of stumps."""
    n = X.shape[0]
    weights = np.ones(n, dtype=np.float64) / n
    stumps = []
    for _ in range(T):
        feat, thresh, left_class, right_class, err = find_best_stump(X, y, weights, C)
        if err >= 1.0 - 1.0 / C or err == 0.0:
            break
        alpha = math.log((1.0 - err) / err) + math.log(C - 1)
        pred = np.where(X[:, feat] <= thresh, left_class, right_class)
        weights = weights * np.exp(alpha * (pred != y))
        weights /= np.sum(weights)
        stumps.append((feat, thresh, left_class, right_class, alpha))
    return stumps


def evaluate(X, y, stumps, C):
    n = X.shape[0]
    scores = np.zeros((n, C), dtype=np.float64)
    for feat, thresh, left_class, right_class, alpha in stumps:
        left = X[:, feat] <= thresh
        scores[left, left_class] += alpha
        scores[~left, right_class] += alpha
    pred = np.argmax(scores, axis=1)
    return float(np.mean(pred == y))


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    # tuning split: last 20% (file order)
    split = n - n // 5
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    best_T = CANDIDATE_T[0]
    best_acc = -1.0
    for T in CANDIDATE_T:
        stumps = train_adaboost(X_train, y_train, T, C)
        acc = evaluate(X_tune, y_tune, stumps, C)
        if acc > best_acc:
            best_acc = acc
            best_T = T

    # retrain on all data with best_T
    final_stumps = train_adaboost(X, y, best_T, C)

    # write weights.bin
    T_final = len(final_stumps)
    # header: magic, C, d, T (each uint32)
    out = struct.pack("<4I", MAGIC, C, d, T_final)
    for feat, thresh, left_class, right_class, alpha in final_stumps:
        out += struct.pack("<BBBBf", feat, thresh, left_class, right_class, alpha)
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* AdaBoost stump ensemble inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixels 0..16). Output record: 1 byte (class 0..9). */
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x41444142) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t C = hd[1], d = hd[2], T = hd[3];
    if (C != 10 || d != 64) { fprintf(stderr, "unexpected C or d\n"); return 3; }
    long expected = 16 + (long)T * 8;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    /* stump data starts at byte 16 */
    unsigned char *stump_data = wb + 16;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[10];
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        memset(scores, 0, sizeof(scores));
        for (uint32_t t = 0; t < T; t++) {
            unsigned char *s = stump_data + t * 8;
            uint8_t feat = s[0];
            uint8_t thresh = s[1];
            uint8_t left_class = s[2];
            uint8_t right_class = s[3];
            float alpha = *(float *)(s + 4);
            if (xi[feat] <= thresh)
                scores[left_class] += alpha;
            else
                scores[right_class] += alpha;
        }
        unsigned char best = 0;
        for (uint32_t c = 1; c < C; c++)
            if (scores[c] > scores[best]) best = c;
        out_byte = best;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```