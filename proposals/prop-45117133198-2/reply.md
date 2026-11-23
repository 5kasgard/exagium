```hypothesis
Propose a Naive Bayes family (label-nbayes) that models each of the 16 context bytes as conditionally independent given the next character. This captures per-position character statistics (e.g., 'q' strongly predicts 'u') with a tiny model: only log-prior and log-conditional probabilities. Inference requires just 16*65 additions and 65 comparisons per item, so it should be faster than label-linear. If it fails to beat label-linear (0.3375), it shows that interactions between context positions are essential for this task.
```

```python
"""Naive Bayes for next-byte prediction: each context byte independent given the class.
python3 -s -P train.py card.json train.in train.key weights.bin
Tunes additive smoothing on the last 20% of train.in, then trains on all data.
Weights: magic 0x4E424159, d, C; float32 log_prior[C], log_cond[d][256][C] (row-major)."""
import json, struct, sys

import numpy as np


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(0.8 * n)  # last 20% for tuning
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    # Count on training part
    prior_cnt = np.bincount(y_tr, minlength=C).astype(np.float64)
    cond_cnt = np.zeros((d, 256, C), dtype=np.float64)
    for j in range(d):
        # use advanced indexing to accumulate
        np.add.at(cond_cnt[j], (X_tr[:, j], y_tr), 1)

    best_alpha, best_acc = 1.0, -1.0
    for alpha in (0.1, 0.5, 1.0, 2.0, 5.0):
        log_prior = np.log(prior_cnt + alpha) - np.log(prior_cnt.sum() + alpha * C)
        log_cond = np.log(cond_cnt + alpha) - np.log(cond_cnt.sum(axis=1, keepdims=True) + alpha * C)
        # vectorized validation scoring
        scores = log_prior + np.sum(log_cond[np.arange(d), X_va.T, :], axis=0).T  # (n_va, C)
        pred = scores.argmax(axis=1)
        acc = (pred == y_va).mean()
        if acc > best_acc:
            best_acc, best_alpha = acc, alpha

    # Retrain on all data with best_alpha
    prior_cnt = np.bincount(y, minlength=C).astype(np.float64)
    cond_cnt = np.zeros((d, 256, C), dtype=np.float64)
    for j in range(d):
        np.add.at(cond_cnt[j], (X[:, j], y), 1)

    log_prior = np.log(prior_cnt + best_alpha) - np.log(prior_cnt.sum() + best_alpha * C)
    log_cond = np.log(cond_cnt + best_alpha) - np.log(cond_cnt.sum(axis=1, keepdims=True) + best_alpha * C)

    # Write weights
    magic = 0x4E424159  # 'YABN' little-endian
    out = struct.pack("<3i", magic, d, C)
    out += log_prior.astype("<f4").tobytes()
    out += log_cond.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Naive Bayes inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index (argmax of log-prior + sum log-cond). */
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
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 12L + 4L * C + 4L * d * 256 * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *log_prior = (const float *)(hd + 3);
    const float *log_cond = log_prior + C;  // [d][256][C]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = log_prior[c];
        for (int j = 0; j < d; j++) {
            const float *row = log_cond + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += row[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```