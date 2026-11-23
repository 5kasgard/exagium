```hypothesis
Family: label-adaboost (AdaBoost.SAMME with decision stumps).
Why: Optdigits has 64 ordinal features (0–16) and 10 classes. Decision stumps exploit single-pixel thresholds, capturing non-linearities that linear models miss, while staying extremely fast at inference (only a few hundred additions per item). AdaBoost provides strong ensemble accuracy with minimal model size.
Failure signal: If dev accuracy ≤ 0.9613 (label-linear) or inference cost exceeds label-linear by >2×, the family does not improve the trade-off.
```

```python
"""label-adaboost: AdaBoost.SAMME with decision stumps for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (no RNG), single thread. Chooses n_stumps and learning_rate on last 20% of train.in,
then trains on all train. Exports: magic, version, d, C, S, base_scores[C], then S stumps:
uint8 feat, uint8 thresh, uint8 left_cls, uint8 right_cls, float32 alpha."""
import json, math, struct, sys

import numpy as np


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_adaboost(X, y, n_stumps, lr, classes):
    n, d = X.shape
    # prior log-probabilities as base scores
    prior = np.bincount(y, minlength=classes).astype(np.float64)
    prior = prior / prior.sum()
    base_scores = np.log(prior + 1e-12).astype(np.float32)

    # one-hot for y
    onehot = np.zeros((n, classes), dtype=np.float64)
    onehot[np.arange(n), y] = 1.0

    w = np.ones(n, dtype=np.float64) / n
    stumps = []  # (feat, thresh, left_cls, right_cls, alpha)

    for _ in range(n_stumps):
        best_err = 1.0
        best_feat = 0
        best_thresh = 0
        best_left_cls = 0
        best_right_cls = 0

        for feat in range(d):
            vals = X[:, feat]  # 0..16
            weighted_onehot = w[:, None] * onehot
            sums = np.zeros((17, classes), dtype=np.float64)
            np.add.at(sums, vals, weighted_onehot)

            # thresholds 0..15 (threshold 16 would put all in left)
            left_sums = np.cumsum(sums[:-1], axis=0)          # (16, C)
            right_sums = sums.sum(axis=0) - left_sums         # (16, C)

            left_cls = np.argmax(left_sums, axis=1)
            right_cls = np.argmax(right_sums, axis=1)

            left_correct = left_sums[np.arange(16), left_cls]
            right_correct = right_sums[np.arange(16), right_cls]
            correct = left_correct + right_correct
            err = w.sum() - correct

            idx = np.argmin(err)
            if err[idx] < best_err:
                best_err = err[idx]
                best_feat = feat
                best_thresh = idx
                best_left_cls = left_cls[idx]
                best_right_cls = right_cls[idx]

        # compute stump weight (SAMME)
        alpha = lr * (math.log((1.0 - best_err) / best_err) + math.log(classes - 1))
        stumps.append((best_feat, best_thresh, best_left_cls, best_right_cls, alpha))

        # update weights
        pred = np.where(X[:, best_feat] <= best_thresh, best_left_cls, best_right_cls)
        incorrect = (pred != y).astype(np.float64)
        w *= np.exp(alpha * incorrect)
        w /= w.sum()

    return base_scores, stumps


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]

    # tuning split: last 20% in file order
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    # hyperparameter grid
    best_acc = -1.0
    best_n_stumps = 50
    best_lr = 1.0
    for n_stumps in (20, 50, 100, 200):
        for lr in (1.0, 0.5, 0.1):
            base, stumps = train_adaboost(X_tr, y_tr, n_stumps, lr, C)
            # validate
            scores = np.tile(base, (X_va.shape[0], 1))
            for feat, thresh, lc, rc, alpha in stumps:
                mask = X_va[:, feat] <= thresh
                scores[mask, lc] += alpha
                scores[~mask, rc] += alpha
            pred = np.argmax(scores, axis=1)
            acc = (pred == y_va).mean()
            if acc > best_acc:
                best_acc = acc
                best_n_stumps = n_stumps
                best_lr = lr

    # retrain on full train
    base, stumps = train_adaboost(X, y, best_n_stumps, best_lr, C)

    # write weights.bin
    # magic "BSTP" = 0x50545342, version=1, d, C, S
    magic = 0x50545342
    version = 1
    S = len(stumps)
    header = struct.pack("<5i", magic, version, d, C, S)
    out = header + base.astype("<f4").tobytes()
    for feat, thresh, lc, rc, alpha in stumps:
        out += struct.pack("<4Bf", feat, thresh, lc, rc, alpha)
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-adaboost infer: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class index.
 * Weights format: magic(0x50545342), version, d, C, S, base_scores[C], then S stumps:
 *   uint8 feat, uint8 thresh, uint8 left_cls, uint8 right_cls, float32 alpha */
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
    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x50545342 || hd[1] != 1) { fprintf(stderr, "bad weights magic/version\n"); return 3; }
    int d = hd[2], C = hd[3], S = hd[4];
    if (d != 64 || C != 10 || S <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    size_t expected = 20 + (size_t)C * 4 + (size_t)S * (4 + 4); // 4B + float
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *base = (const float *)(wb + 20);
    const unsigned char *st = wb + 20 + C * 4;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[10];
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(scores, base, C * sizeof(float));
        for (int s = 0; s < S; s++) {
            uint8_t feat = st[0];
            uint8_t thresh = st[1];
            uint8_t lc = st[2];
            uint8_t rc = st[3];
            float alpha = *(const float *)(st + 4);
            st += 8;
            if (xi[feat] <= thresh) scores[lc] += alpha;
            else scores[rc] += alpha;
        }
        best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```