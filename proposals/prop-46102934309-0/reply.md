```hypothesis
We propose a multinomial Naive Bayes family (label-nb) with Laplace smoothing tuned on the last 20% of training data. It treats all 14 features as categorical with 256 possible values, matching the input encoding. Naive Bayes is extremely fast at inference (O(d*C) per item) and often competitive on tabular data with discrete features. If its dev accuracy falls below 0.84, it would show that the independence assumption is too strong for this dataset.
```

```python
"""Multinomial Naive Bayes for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Tunes smoothing alpha on last 20% of train.in."""
import json, math, struct, sys

import numpy as np

SEED = 1
ALPHAS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]  # smoothing candidates


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
    V = 256

    n = X.shape[0]
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)

    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    best_alpha = ALPHAS[0]
    best_acc = -1.0

    for alpha in ALPHAS:
        # count on train split
        class_count = np.zeros(C, dtype=np.int64)
        feat_count = np.zeros((d, V, C), dtype=np.int64)
        for i in range(len(y_train)):
            c = y_train[i]
            class_count[c] += 1
            for j in range(d):
                feat_count[j, X_train[i, j], c] += 1

        # log prior
        log_prior = np.log(class_count / class_count.sum())
        # log conditional with smoothing
        log_cond = np.log((feat_count + alpha) / (class_count.reshape(1, 1, C) + alpha * V))

        # evaluate on tune split
        correct = 0
        for i in range(len(y_tune)):
            scores = log_prior.copy()
            for j in range(d):
                scores += log_cond[j, X_tune[i, j], :]
            pred = np.argmax(scores)
            if pred == y_tune[i]:
                correct += 1
        acc = correct / len(y_tune)
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    # retrain on all data with best_alpha
    class_count = np.zeros(C, dtype=np.int64)
    feat_count = np.zeros((d, V, C), dtype=np.int64)
    for i in range(n):
        c = y[i]
        class_count[c] += 1
        for j in range(d):
            feat_count[j, X[i, j], c] += 1

    log_prior = np.log(class_count / class_count.sum()).astype(np.float32)
    log_cond = np.log((feat_count + best_alpha) / (class_count.reshape(1, 1, C) + best_alpha * V)).astype(np.float32)

    # write weights.bin
    # magic: 0x4E420001 (NB\0\1), d, C, V, alpha, log_prior[C], log_cond[d][V][C]
    magic = 0x4E420001
    alpha_f32 = np.float32(best_alpha)
    out = struct.pack("<5i f", magic, d, C, V, 0, alpha_f32)  # 5 ints + float (padding for alignment? we'll pack separately)
    # Actually pack as: magic, d, C, V, alpha, then log_prior, then log_cond
    # Let's do it cleanly:
    header = struct.pack("<4i f", magic, d, C, V, alpha_f32)
    out = header + log_prior.tobytes() + log_cond.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Multinomial Naive Bayes inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class. */
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
    if (nw < 20) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | (uint32_t)wb[3] << 24;
    if (magic != 0x4E420001) { fprintf(stderr, "bad weights: magic mismatch\n"); return 3; }
    int32_t d = *(int32_t*)(wb + 4);
    int32_t C = *(int32_t*)(wb + 8);
    int32_t V = *(int32_t*)(wb + 12);
    float alpha = *(float*)(wb + 16);
    if (C < 1 || C > 256 || V != 256) { fprintf(stderr, "bad weights: invalid dims\n"); return 3; }
    size_t expected = 20 + (size_t)C * 4 + (size_t)d * V * C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights: size mismatch %ld != %zu\n", nw, expected); return 3; }

    const float *log_prior = (const float*)(wb + 20);
    const float *log_cond = (const float*)(wb + 20 + C * 4);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(scores, log_prior, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *cond = log_cond + ((size_t)j * V + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += cond[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```