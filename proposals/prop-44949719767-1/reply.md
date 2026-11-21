```hypothesis
Family: label-nbayes (categorical Naive Bayes with Laplace smoothing).
Why: Optdigits pixels are ordinal 0–16; NB models full per-pixel conditional distributions non-parametrically, capturing non-linearities missed by linear models. Inference uses identical table-lookup pattern as label-linear (64×10 adds/record), matching its CPU cost.
Failure signal: Dev accuracy ≤ 0.9613 (label-linear), meaning independence assumption loses more than linear's discriminative training gains.
```

```python
import json, struct, sys

import numpy as np

SEED = 1
ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])

    d = card["input"]["bytes"]
    C = card["classes"]

    X = np.frombuffer(b"".join(X_bytes), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)
    n = X.shape[0]

    val_size = n // 5
    train_size = n - val_size
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    best_alpha = 1.0
    best_acc = -1.0

    for alpha in ALPHAS:
        class_counts = np.bincount(y_train, minlength=C).astype(np.float64)
        log_prior = np.log((class_counts + alpha) / (len(y_train) + C * alpha))

        feat_counts = np.zeros((d, 256, C), dtype=np.float64)
        for j in range(d):
            for c in range(C):
                mask = (y_train == c)
                if np.any(mask):
                    vals = X_train[mask, j]
                    counts = np.bincount(vals, minlength=256)
                    feat_counts[j, :, c] = counts

        class_counts_3d = class_counts.reshape(1, 1, C)
        log_prob = np.log((feat_counts + alpha) / (class_counts_3d + 256 * alpha))

        val_scores = log_prior + np.sum(
            log_prob[np.arange(d)[:, None, None], X_val.T[:, :, None], :], axis=0
        )
        val_pred = np.argmax(val_scores, axis=1)
        acc = np.mean(val_pred == y_val)

        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    class_counts = np.bincount(y, minlength=C).astype(np.float64)
    log_prior = np.log((class_counts + best_alpha) / (n + C * best_alpha))

    feat_counts = np.zeros((d, 256, C), dtype=np.float64)
    for j in range(d):
        for c in range(C):
            mask = (y == c)
            if np.any(mask):
                vals = X[mask, j]
                counts = np.bincount(vals, minlength=256)
                feat_counts[j, :, c] = counts

    class_counts_3d = class_counts.reshape(1, 1, C)
    log_prob = np.log((feat_counts + best_alpha) / (class_counts_3d + 256 * best_alpha))

    out = struct.pack("<4i", 0x4E424159, d, C, 256)
    out += log_prior.astype(np.float32).tobytes()
    out += log_prob.astype(np.float32).tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
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
    if (nw < 16 || hd[0] != 0x4E424159) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], V = hd[3];
    if (C < 1 || C > 256 || V != 256 || d < 1 || nw != 16 + 4L * C + 4L * d * 256 * C) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *log_prior = (const float *)(hd + 4);
    const float *log_prob = (const float *)(log_prior + C);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = log_prior[c];
        for (int j = 0; j < d; j++) {
            const float *lp = log_prob + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += lp[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```