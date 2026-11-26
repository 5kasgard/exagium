```hypothesis
Family: label-adaboost (boosted decision stumps with SAMME).
Why: Decision stumps on ordinal pixels capture non-linearities; boosting combines them into a strong classifier. Inference aggregates per-feature lookup tables, matching label-linear's speed (same memory access pattern) while often exceeding its accuracy on this task.
Failure: If dev accuracy <= 0.9613 (label-linear) or inference cost exceeds label-linear by >10%, the hypothesis fails.
```

```python
import json
import math
import struct
import sys

import numpy as np

SEED = 1
# Candidate numbers of stumps to try on the tuning split
M_CANDIDATES = [10, 20, 50, 100, 200, 500, 1000]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_stumps(X, y, sample_weight, n_classes):
    """Train a single decision stump (depth-1 tree) on weighted data.
    Returns (feature_idx, threshold, class_left, class_right, alpha)."""
    n, d = X.shape
    best_err = float('inf')
    best_split = None
    # For each feature, try all possible thresholds (0..15 for <= split)
    for j in range(d):
        vals = X[:, j]  # values 0..16
        # We'll try thresholds from 0 to 15 (split <= t vs > t)
        for t in range(16):
            # left: vals <= t, right: vals > t
            left_mask = vals <= t
            right_mask = ~left_mask
            # Find best class for left
            left_weights = sample_weight[left_mask]
            left_y = y[left_mask]
            if len(left_weights) > 0:
                # weighted class counts
                left_counts = np.bincount(left_y, weights=left_weights, minlength=n_classes)
                class_left = int(np.argmax(left_counts))
            else:
                class_left = 0
            # best class for right
            right_weights = sample_weight[right_mask]
            right_y = y[right_mask]
            if len(right_weights) > 0:
                right_counts = np.bincount(right_y, weights=right_weights, minlength=n_classes)
                class_right = int(np.argmax(right_counts))
            else:
                class_right = 0
            # Compute weighted error
            pred = np.where(left_mask, class_left, class_right)
            err = np.sum(sample_weight * (pred != y))
            if err < best_err:
                best_err = err
                best_split = (j, t, class_left, class_right)
    # If error is too high, return None
    if best_err >= 1.0 - 1.0 / n_classes:
        return None
    # Compute alpha
    eps = 1e-12
    err = max(best_err, eps)
    alpha = math.log((1.0 - err) / err) + math.log(n_classes - 1)
    return (*best_split, alpha)


def evaluate_stumps(X, y, stumps, n_classes):
    """Evaluate ensemble of stumps on (X, y). Returns accuracy."""
    n = X.shape[0]
    scores = np.zeros((n, n_classes), dtype=np.float32)
    for (j, t, cl, cr, alpha) in stumps:
        left = X[:, j] <= t
        scores[left, cl] += alpha
        scores[~left, cr] += alpha
    pred = np.argmax(scores, axis=1)
    return np.mean(pred == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    n_classes = card["classes"]

    # Tuning split: last 20% in file order
    split_idx = int(0.8 * n)
    X_train, X_tune = X[:split_idx], X[split_idx:]
    y_train, y_tune = y[:split_idx], y[split_idx:]

    # Tune number of stumps
    best_m = M_CANDIDATES[0]
    best_acc = -1.0
    rng = np.random.default_rng(SEED)
    # We'll train on the 80% for each candidate and evaluate on 20%
    for m in M_CANDIDATES:
        # Train SAMME on X_train
        sample_weight = np.ones(len(X_train), dtype=np.float64) / len(X_train)
        stumps = []
        for _ in range(m):
            # Normalize weights
            sample_weight /= sample_weight.sum()
            stump = train_stumps(X_train, y_train, sample_weight, n_classes)
            if stump is None:
                break
            j, t, cl, cr, alpha = stump
            stumps.append(stump)
            # Update weights
            pred = np.where(X_train[:, j] <= t, cl, cr)
            sample_weight *= np.exp(alpha * (pred != y_train))
        # Evaluate on tuning set
        acc = evaluate_stumps(X_tune, y_tune, stumps, n_classes)
        if acc > best_acc:
            best_acc = acc
            best_m = m
    # Retrain on full data with best_m
    sample_weight = np.ones(n, dtype=np.float64) / n
    stumps = []
    for _ in range(best_m):
        sample_weight /= sample_weight.sum()
        stump = train_stumps(X, y, sample_weight, n_classes)
        if stump is None:
            break
        j, t, cl, cr, alpha = stump
        stumps.append(stump)
        pred = np.where(X[:, j] <= t, cl, cr)
        sample_weight *= np.exp(alpha * (pred != y_train))

    # Build per-feature lookup tables: shape (d, 256, n_classes)
    # Only values 0..16 are possible; others remain 0.
    T = np.zeros((d, 256, n_classes), dtype=np.float32)
    for (j, t, cl, cr, alpha) in stumps:
        for v in range(17):  # 0..16
            if v <= t:
                T[j, v, cl] += alpha
            else:
                T[j, v, cr] += alpha

    # Bias: log prior
    prior = np.bincount(y, minlength=n_classes).astype(np.float64)
    prior = prior / prior.sum()
    b1 = np.log(prior).astype(np.float32)

    # Write weights.bin
    # Format: magic (0x41444142), d, H=0, C, then T (d*256*C float32), then b1 (C float32)
    magic = 0x41444142  # "ADAB"
    H = 0
    C = n_classes
    out = struct.pack("<4i", magic, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Boosted decision stumps (SAMME) for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class index.
 * Weights format: int32 magic 0x41444142, int32 d, int32 H=0, int32 C;
 * float32 T[d][256][C]; float32 b1[C]. */
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x41444142) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (H != 0 || C < 1 || C > 256) { fprintf(stderr, "bad weights config\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(scores, b1, sizeof(float) * C);
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += t[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```