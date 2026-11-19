```hypothesis
Family: label-nb (Multinomial Naive Bayes with Laplace smoothing). The 14 input bytes are already discretized (255 bins + unseen), making them ideal for categorical Naive Bayes. It captures per-class feature distributions non-linearly, is extremely fast at inference (only additions and table lookups), and has one hyperparameter (alpha) tuned on the last 20% of train. If dev accuracy does not exceed label-linear's 0.8524, the conditional-independence assumption is too restrictive for this dataset.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
ALPHA_CANDIDATES = [0.1, 0.5, 1.0, 2.0, 5.0]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = b"".join(records(sys.argv[2]))
    y_bytes = b"".join(records(sys.argv[3]))
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.frombuffer(y_bytes, dtype=np.uint8)
    n, d = X.shape
    C = card["classes"]
    assert C == 2

    # split: last 20% for tuning
    split = int(n * 0.8)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    best_alpha = 1.0
    best_acc = -1.0
    for alpha in ALPHA_CANDIDATES:
        # train on first 80%
        class_count = np.zeros(C, dtype=np.int64)
        feat_count = np.zeros((C, d, 256), dtype=np.int64)
        for i in range(len(X_tr)):
            c = y_tr[i]
            class_count[c] += 1
            for j in range(d):
                feat_count[c, j, X_tr[i, j]] += 1
        # log probabilities with Laplace smoothing
        log_prior = np.log((class_count + alpha) / (class_count.sum() + alpha * C))
        log_cond = np.zeros((C, d, 256), dtype=np.float32)
        for c in range(C):
            for j in range(d):
                total = class_count[c] + alpha * 256
                log_cond[c, j, :] = np.log((feat_count[c, j, :] + alpha) / total)
        # evaluate on validation
        correct = 0
        for i in range(len(X_va)):
            scores = log_prior.copy()
            for c in range(C):
                for j in range(d):
                    scores[c] += log_cond[c, j, X_va[i, j]]
            pred = 0 if scores[0] >= scores[1] else 1
            if pred == y_va[i]:
                correct += 1
        acc = correct / len(X_va)
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    # retrain on all data with best_alpha
    class_count = np.zeros(C, dtype=np.int64)
    feat_count = np.zeros((C, d, 256), dtype=np.int64)
    for i in range(n):
        c = y[i]
        class_count[c] += 1
        for j in range(d):
            feat_count[c, j, X[i, j]] += 1
    log_prior = np.log((class_count + best_alpha) / (n + best_alpha * C))
    log_cond = np.zeros((C, d, 256), dtype=np.float32)
    for c in range(C):
        for j in range(d):
            total = class_count[c] + best_alpha * 256
            log_cond[c, j, :] = np.log((feat_count[c, j, :] + best_alpha) / total)

    # write weights: magic, C, d, alpha, log_prior[C], log_cond[C][d][256] all float32 little-endian
    magic = 0x4E424C31  # 'NBL1'
    out = struct.pack("<I B B f", magic, C, d, best_alpha)
    out += log_prior.astype("<f4").tobytes()
    out += log_cond.astype("<f4").tobytes()
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 10) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x4E424C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int C = wb[4], d = wb[5];
    float alpha = *(float*)(wb + 6);
    if (C != 2 || d != 14) { fprintf(stderr, "unexpected C or d\n"); return 3; }
    size_t expected = 10 + C * sizeof(float) + C * d * 256 * sizeof(float);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *log_prior = (const float*)(wb + 10);
    const float *log_cond = (const float*)(wb + 10 + C * sizeof(float));
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float score0 = log_prior[0], score1 = log_prior[1];
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            score0 += log_cond[0 * d * 256 + j * 256 + v];
            score1 += log_cond[1 * d * 256 + j * 256 + v];
        }
        unsigned char best = (score0 >= score1) ? 0 : 1;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```