```hypothesis
Family: label-nb (Naive Bayes with per-feature histograms).
Why: Adult has 14 features (mix of ordinal quantile bins and categorical indices) with a small number of classes (2). Naive Bayes captures non-linear per-feature effects via log-likelihood tables, is extremely fast at inference (just table lookups and sums), and often matches or exceeds logistic regression on this dataset. The model has no hyperparameters except Laplace smoothing, which we tune on the held-out 20%.
Failure signal: If dev accuracy stays ≤0.8524 (label-linear) or inference cost exceeds label-linear, the family does not meet the goal.
```

```python
"""label-nb: Naive Bayes with per-feature 256-bin histograms per class.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib + numpy only.
Weights: magic 0x4E424159, int32 d, int32 C, float32 log_prior[C], float32 log_likelihood[d][256][C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0]  # Laplace smoothing candidates


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]

    # Split: last 20% for tuning, first 80% for training (file order)
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    best_alpha, best_acc = 1.0, -1.0
    rng = np.random.default_rng(SEED)

    for alpha in ALPHAS:
        # Count on training split
        prior = np.bincount(y_tr, minlength=C).astype(np.float64)
        prior = (prior + alpha) / (prior.sum() + alpha * C)
        log_prior = np.log(prior).astype(np.float32)

        # likelihood[d][256][C]
        log_lik = np.zeros((d, 256, C), dtype=np.float32)
        for j in range(d):
            for c in range(C):
                mask = (y_tr == c)
                cnt = np.bincount(X_tr[mask, j], minlength=256).astype(np.float64)
                cnt = (cnt + alpha) / (cnt.sum() + alpha * 256)
                log_lik[j, :, c] = np.log(cnt).astype(np.float32)

        # Evaluate on tuning split
        correct = 0
        for i in range(X_tune.shape[0]):
            scores = log_prior.copy()
            for j in range(d):
                scores += log_lik[j, X_tune[i, j], :]
            pred = int(np.argmax(scores))
            if pred == y_tune[i]:
                correct += 1
        acc = correct / X_tune.shape[0]
        if acc > best_acc:
            best_acc, best_alpha = acc, alpha

    # Retrain on full data with best_alpha
    prior = np.bincount(y, minlength=C).astype(np.float64)
    prior = (prior + best_alpha) / (prior.sum() + best_alpha * C)
    log_prior = np.log(prior).astype(np.float32)

    log_lik = np.zeros((d, 256, C), dtype=np.float32)
    for j in range(d):
        for c in range(C):
            mask = (y == c)
            cnt = np.bincount(X[mask, j], minlength=256).astype(np.float64)
            cnt = (cnt + best_alpha) / (cnt.sum() + best_alpha * 256)
            log_lik[j, :, c] = np.log(cnt).astype(np.float32)

    # Write weights.bin
    magic = 0x4E424159  # 'NBAY'
    out = struct.pack("<3i", magic, d, C)
    out += log_prior.astype("<f4").tobytes()
    out += log_lik.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-nb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: one byte class index (0 or 1). */
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
    if (nw < 12 || hd[0] != 0x4E424159) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 12L + 4L * C + 4L * d * 256 * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *log_prior = (const float *)(hd + 3);
    const float *log_lik = log_prior + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(scores, log_prior, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *row = log_lik + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += row[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```