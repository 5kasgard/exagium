```hypothesis
Family: label-naivebayes. Multinomial Naive Bayes with per-feature add-alpha smoothing on the 256-value discrete columns.
Why: Adult's 14 columns are all discrete (ordinal quantile-binned or categorical indices). Naive Bayes models P(x|y) per feature non-parametrically, capturing non-linear class-conditional distributions that logistic regression misses, while staying extremely fast at inference (only table lookups and additions).
Failure signal: dev accuracy ≤ 0.8524 (label-linear's score) or inference cost > 1000 Ir/item.
```
```python
"""label-naivebayes: Multinomial Naive Bayes with add-alpha smoothing on 256-value discrete features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib + numpy only.
Weights format: magic(0x5942414E), d, C, alpha_idx, log_prior[C], log_prob[d][256][C] (float32)."""
import json, math, struct, sys

import numpy as np

SEED = 1
ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0]  # tuned on last 20% of train.in


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
    tune_n = n // 5
    tr_X, tu_X = X[:-tune_n], X[-tune_n:]
    tr_y, tu_y = y[:-tune_n], y[-tune_n:]

    best_alpha, best_acc = ALPHAS[0], -1.0
    for alpha in ALPHAS:
        # prior
        prior = np.bincount(tr_y, minlength=C).astype(np.float64)
        prior = (prior + alpha) / (prior.sum() + alpha * C)
        # conditional
        cond = np.ones((d, 256, C), dtype=np.float64) * alpha
        for j in range(d):
            np.add.at(cond[j], (tr_X[:, j], tr_y), 1)
        cond /= cond.sum(axis=1, keepdims=True)
        log_prior = np.log(prior).astype(np.float32)
        log_cond = np.log(cond).astype(np.float32)
        # evaluate on tune
        scores = log_prior + np.sum(log_cond[np.arange(d)[:, None], tu_X.T, :], axis=0).T
        pred = scores.argmax(axis=1)
        acc = (pred == tu_y).mean()
        if acc > best_acc:
            best_acc, best_alpha = acc, alpha

    # retrain on full data with best_alpha
    prior = np.bincount(y, minlength=C).astype(np.float64)
    prior = (prior + best_alpha) / (prior.sum() + best_alpha * C)
    cond = np.ones((d, 256, C), dtype=np.float64) * best_alpha
    for j in range(d):
        np.add.at(cond[j], (X[:, j], y), 1)
    cond /= cond.sum(axis=1, keepdims=True)
    log_prior = np.log(prior).astype(np.float32)
    log_cond = np.log(cond).astype(np.float32)

    alpha_idx = ALPHAS.index(best_alpha)
    magic = 0x5942414E  # "NABY" little-endian
    out = struct.pack("<4i", magic, d, C, alpha_idx)
    out += log_prior.tobytes()
    out += log_cond.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-naivebayes inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class index (0 or 1). */
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
    if (nw < 16 || hd[0] != 0x5942414E) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], alpha_idx = hd[3];
    if (d != 14 || C != 2 || alpha_idx < 0 || alpha_idx >= 5) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16L + 4L * C + 4L * d * 256 * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *log_prior = (const float *)(hd + 4);
    const float *log_cond = log_prior + C;  // [d][256][C]
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float score[2] = {log_prior[0], log_prior[1]};
        for (int j = 0; j < d; j++) {
            const float *row = log_cond + ((long)j * 256 + xi[j]) * C;
            score[0] += row[0];
            score[1] += row[1];
        }
        unsigned char best = score[1] > score[0] ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```