```hypothesis
Family: label-ada (AdaBoost with decision stumps). Each stump splits on a single pixel threshold and predicts a class for each side; AdaBoost combines them with weighted voting. This should capture non-linear pixel interactions while staying extremely fast at inference (one comparison and two additions per stump). It will beat label-linear if dev accuracy exceeds 96.13%; if it scores ≤96.13% it does not.
```
```python
"""AdaBoost with decision stumps for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Uses SAMME algorithm.
Exported format: magic 0x41444131, int32 T, d, C; then T records of (uint8 feat, uint8 thresh, uint8 left_cls, uint8 right_cls, float32 alpha)."""
import json, math, struct, sys

import numpy as np

SEED = 1
CANDIDATE_T = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_adaboost(X, y, C, T):
    """Train AdaBoost SAMME with decision stumps. Returns list of stumps."""
    n, d = X.shape
    w = np.ones(n, dtype=np.float64) / n
    stumps = []
    for _ in range(T):
        best_err = 1.0
        best_feat = best_thresh = best_left = best_right = 0
        # For each feature, try all thresholds (0..16)
        for feat in range(d):
            vals = X[:, feat]
            # Since pixel values are 0..16, we can loop thresholds
            for thresh in range(17):  # 0..16 inclusive
                mask = vals <= thresh
                # Weighted class counts for left and right
                left_counts = np.zeros(C, dtype=np.float64)
                right_counts = np.zeros(C, dtype=np.float64)
                np.add.at(left_counts, y[mask], w[mask])
                np.add.at(right_counts, y[~mask], w[~mask])
                left_cls = int(np.argmax(left_counts))
                right_cls = int(np.argmax(right_counts))
                # Error = total weight - weight of correct predictions
                correct_weight = left_counts[left_cls] + right_counts[right_cls]
                err = 1.0 - correct_weight
                if err < best_err:
                    best_err = err
                    best_feat = feat
                    best_thresh = thresh
                    best_left = left_cls
                    best_right = right_cls
        if best_err >= 1.0 - 1.0 / C or best_err <= 0.0:
            # No improvement or perfect
            alpha = 0.0
            stumps.append((best_feat, best_thresh, best_left, best_right, alpha))
            break
        alpha = math.log((1.0 - best_err) / best_err) + math.log(C - 1)
        # Update weights
        pred = np.where(X[:, best_feat] <= best_thresh, best_left, best_right)
        w *= np.exp(alpha * (pred != y))
        w /= w.sum()
        stumps.append((best_feat, best_thresh, best_left, best_right, alpha))
    return stumps


def evaluate(X, y, stumps, C):
    n = X.shape[0]
    scores = np.zeros((n, C), dtype=np.float64)
    for feat, thresh, left, right, alpha in stumps:
        if alpha == 0.0:
            continue
        mask = X[:, feat] <= thresh
        scores[mask, left] += alpha
        scores[~mask, right] += alpha
    pred = np.argmax(scores, axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = len(y_all)
    # Tuning split: last 20%
    split = int(0.8 * n)
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]

    best_T, best_acc = 0, -1.0
    for T in CANDIDATE_T:
        stumps = train_adaboost(X_train, y_train, C, T)
        acc = evaluate(X_val, y_val, stumps, C)
        if acc > best_acc:
            best_acc, best_T = acc, T

    # Retrain on full data
    final_stumps = train_adaboost(X_all, y_all, C, best_T)

    # Write weights.bin
    magic = 0x41444131  # 'ADA1'
    out = struct.pack("<4i", magic, len(final_stumps), d, C)
    for feat, thresh, left, right, alpha in final_stumps:
        out += struct.pack("<BBBBf", feat, thresh, left, right, alpha)
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* AdaBoost decision stump ensemble for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the class with highest weighted vote. */
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x41444131) { fprintf(stderr, "bad weights\n"); return 3; }
    int T = hd[1], d = hd[2], C = hd[3];
    if (C < 1 || C > 256 || T < 0) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16 + (long)T * 8;  // 4 bytes magic+T+d+C = 16, each stump 8 bytes (BBBBf)
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    unsigned char *stumps = (unsigned char *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(scores, 0, sizeof(float) * C);
        for (int t = 0; t < T; t++) {
            unsigned char *st = stumps + t * 8;
            uint8_t feat = st[0], thresh = st[1], left = st[2], right = st[3];
            float alpha = *(float *)(st + 4);
            if (alpha == 0.0f) continue;
            if (xi[feat] <= thresh) scores[left] += alpha;
            else scores[right] += alpha;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```