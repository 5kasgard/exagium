```hypothesis
Family: label-nb (Naive Bayes with categorical features). Optdigits pixels are ordinal 0..16, so a multinomial NB with Laplace smoothing per pixel per class captures independent feature contributions. It adds non-linear decision boundaries via per-class per-pixel log-probabilities, yet inference is just a sum of 64 lookups per class (640 adds), far cheaper than an MLP. If dev accuracy does not exceed label-linear (0.9613) or cost per item exceeds label-linear, the hypothesis fails.
```

```python
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    V = 17  # pixel counts 0..16
    n = X.shape[0]

    # tuning split: last 20% in file order
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    # try a few Laplace smoothing values
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
    best_alpha = 1.0
    best_acc = -1.0

    for alpha in alphas:
        class_counts = np.bincount(y_tr, minlength=C)
        counts = np.zeros((d, C, V), dtype=np.int64)
        for c in range(C):
            mask = (y_tr == c)
            for j in range(d):
                vals = X_tr[mask, j]
                np.add.at(counts[j, c], vals, 1)
        log_probs = np.zeros((d, C, V), dtype=np.float32)
        for j in range(d):
            for c in range(C):
                total = class_counts[c]
                denom = total + V * alpha
                log_probs[j, c] = np.log((counts[j, c] + alpha) / denom)
        log_priors = np.log(class_counts / class_counts.sum()).astype(np.float32)

        # evaluate on tuning split
        correct = 0
        for i in range(X_tune.shape[0]):
            scores = log_priors.copy()
            for j in range(d):
                scores += log_probs[j, :, X_tune[i, j]]
            if np.argmax(scores) == y_tune[i]:
                correct += 1
        acc = correct / X_tune.shape[0]
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    # train on all data with best_alpha
    class_counts = np.bincount(y, minlength=C)
    counts = np.zeros((d, C, V), dtype=np.int64)
    for c in range(C):
        mask = (y == c)
        for j in range(d):
            vals = X[mask, j]
            np.add.at(counts[j, c], vals, 1)
    log_probs = np.zeros((d, C, V), dtype=np.float32)
    for j in range(d):
        for c in range(C):
            total = class_counts[c]
            denom = total + V * best_alpha
            log_probs[j, c] = np.log((counts[j, c] + best_alpha) / denom)
    log_priors = np.log(class_counts / class_counts.sum()).astype(np.float32)

    # write weights.bin
    magic = 0x4E424C31  # "NBL1"
    out = struct.pack("<4i", magic, d, C, V)
    out += log_priors.tobytes()
    out += log_probs.tobytes()  # shape (d, C, V) in C order
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
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
    if (nw < 16 || hd[0] != 0x4E424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], V = hd[3];
    if (d != 64 || C != 10 || V != 17) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 16 + (long)C * 4 + (long)d * C * V * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *log_prior = (const float *)(hd + 4);
    const float *log_prob = (const float *)(log_prior + C);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(scores, log_prior, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            if (v >= V) { fprintf(stderr, "value out of range\n"); return 3; }
            const float *lp = log_prob + ((long)j * C) * V;
            for (int c = 0; c < C; c++) {
                scores[c] += lp[c * V + v];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```