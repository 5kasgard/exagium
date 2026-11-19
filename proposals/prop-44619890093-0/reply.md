```hypothesis
Family: label-adaboost. AdaBoost with decision stumps on the 14 byte-wise features (ordinal splits <= threshold, categorical splits == value). 
The adult dataset has mixed ordinal/categorical features; stumps capture non-linearities and interactions that linear models miss, 
while staying extremely fast (O(T*d) per item). Tuning the number of stumps on the last 20% of train.in should beat label-linear's 
0.8524 dev accuracy. If dev accuracy <= 0.8524, the hypothesis fails.
```

```python
"""AdaBoost with decision stumps for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
TUNE_FRACTION = 0.2
T_CANDIDATES = [10, 20, 50, 100, 200]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_adaboost(X, y, T, types):
    """X: (n, d) uint8, y: (n,) in {-1, +1}, types: list of 'ordinal'/'categorical'."""
    n, d = X.shape
    w = np.ones(n, dtype=np.float64) / n
    stumps = []
    classes = np.array([-1, 1], dtype=np.int8)

    for _ in range(T):
        w_sum = w.sum()
        if w_sum == 0:
            break
        w = w / w_sum

        best_err = 1.0
        best_stump = None  # (feature, split_type, threshold, left_pred, right_pred)

        for j in range(d):
            vals = X[:, j]
            # Weighted class sums per value (2 x 256)
            sum_w = np.zeros((2, 256), dtype=np.float64)
            for c_idx, c in enumerate(classes):
                mask = (y == c)
                if np.any(mask):
                    sum_w[c_idx] = np.bincount(vals[mask], weights=w[mask], minlength=256)
            total_w_c = sum_w.sum(axis=1)  # (2,)

            if types[j] == "ordinal":
                # Thresholds 0..254 (left: <= t, right: > t)
                cum_w = np.cumsum(sum_w, axis=1)  # (2, 256)
                left_w = cum_w[:, :255]           # (2, 255)
                right_w = total_w_c[:, None] - left_w
                left_pred = np.where(left_w[1] > left_w[0], 1, -1)
                right_pred = np.where(right_w[1] > right_w[0], 1, -1)
                err_left = np.where(left_pred == 1, left_w[0], left_w[1])
                err_right = np.where(right_pred == 1, right_w[0], right_w[1])
                err = err_left + err_right
                t = int(np.argmin(err))
                if err[t] < best_err:
                    best_err = err[t]
                    best_stump = (j, 0, t, int(left_pred[t]), int(right_pred[t]))
            else:  # categorical
                left_w = sum_w                     # (2, 256)
                right_w = total_w_c[:, None] - left_w
                left_pred = np.where(left_w[1] > left_w[0], 1, -1)
                right_pred = np.where(right_w[1] > right_w[0], 1, -1)
                err_left = np.where(left_pred == 1, left_w[0], left_w[1])
                err_right = np.where(right_pred == 1, right_w[0], right_w[1])
                err = err_left + err_right
                v = int(np.argmin(err))
                if err[v] < best_err:
                    best_err = err[v]
                    best_stump = (j, 1, v, int(left_pred[v]), int(right_pred[v]))

        if best_err >= 0.5 - 1e-12:
            break
        if best_err <= 1e-12:
            alpha = 1e6
        else:
            alpha = 0.5 * math.log((1.0 - best_err) / best_err)

        j, split_type, threshold, left_pred, right_pred = best_stump
        stumps.append((j, split_type, threshold, alpha, left_pred, right_pred))

        # Update weights
        vals = X[:, j]
        if split_type == 0:
            pred = np.where(vals <= threshold, left_pred, right_pred)
        else:
            pred = np.where(vals == threshold, left_pred, right_pred)
        w *= np.exp(-alpha * y * pred)

    return stumps


def predict_adaboost(X, stumps):
    n = X.shape[0]
    scores = np.zeros(n, dtype=np.float64)
    for (j, split_type, threshold, alpha, left_pred, right_pred) in stumps:
        vals = X[:, j]
        if split_type == 0:
            pred = np.where(vals <= threshold, left_pred, right_pred)
        else:
            pred = np.where(vals == threshold, left_pred, right_pred)
        scores += alpha * pred
    return np.where(scores > 0, 1, -1)


def main():
    card = json.load(open(sys.argv[1]))
    types = card["input"]["types"]
    d = card["input"]["bytes"]
    C = card["classes"]

    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    y_all = 2 * y_all - 1  # map {0,1} -> {-1,+1}

    n = X_all.shape[0]
    tune_start = int(n * (1.0 - TUNE_FRACTION))
    X_train, y_train = X_all[:tune_start], y_all[:tune_start]
    X_val, y_val = X_all[tune_start:], y_all[tune_start:]

    best_T = T_CANDIDATES[0]
    best_acc = -1.0
    for T in T_CANDIDATES:
        stumps = train_adaboost(X_train, y_train, T, types)
        if not stumps:
            acc = 0.0
        else:
            pred = predict_adaboost(X_val, stumps)
            acc = (pred == y_val).mean()
        if acc > best_acc:
            best_acc = acc
            best_T = T

    # Retrain on all data
    final_stumps = train_adaboost(X_all, y_all, best_T, types)

    # Export weights.bin
    # Magic: 0x41444142 ('ADAB' little-endian)
    # Header: magic (uint32), d (int32), T (int32), C (int32)
    # Per stump: feature (uint8), split_type (uint8), threshold (uint8),
    #            alpha (float32), left_pred (int8), right_pred (int8)
    T_final = len(final_stumps)
    header = struct.pack("<Iiii", 0x41444142, d, T_final, C)
    body = bytearray()
    for (j, split_type, threshold, alpha, left_pred, right_pred) in final_stumps:
        body += struct.pack("<BBBfbb", j, split_type, threshold, alpha, left_pred, right_pred)
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* AdaBoost decision stumps for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte class (0 or 1). */
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
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | (uint32_t)wb[3] << 24;
    if (magic != 0x41444142) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t d = *(int32_t*)(wb + 4);
    int32_t T = *(int32_t*)(wb + 8);
    int32_t C = *(int32_t*)(wb + 12);
    if (C != 2 || d != 14 || T <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + (long)T * 9;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    // Parse stumps
    uint8_t *feature = malloc(T);
    uint8_t *split_type = malloc(T);
    uint8_t *threshold = malloc(T);
    float *alpha = malloc(T * sizeof(float));
    int8_t *left_pred = malloc(T);
    int8_t *right_pred = malloc(T);
    unsigned char *p = wb + 16;
    for (int i = 0; i < T; i++) {
        feature[i] = p[0];
        split_type[i] = p[1];
        threshold[i] = p[2];
        memcpy(&alpha[i], p + 3, 4);
        left_pred[i] = (int8_t)p[7];
        right_pred[i] = (int8_t)p[8];
        p += 9;
    }
    free(wb);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        double score = 0.0;
        for (int t = 0; t < T; t++) {
            uint8_t v = xi[feature[t]];
            int8_t pred;
            if (split_type[t] == 0) {
                pred = (v <= threshold[t]) ? left_pred[t] : right_pred[t];
            } else {
                pred = (v == threshold[t]) ? left_pred[t] : right_pred[t];
            }
            score += alpha[t] * pred;
        }
        unsigned char cls = (score > 0.0) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```