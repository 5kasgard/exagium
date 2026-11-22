```hypothesis
Propose label-nb: a Naive Bayes classifier with per-pixel categorical distributions (values 0–16) and Laplace smoothing.
Optdigits pixels are ordinal counts; modeling each pixel independently per class captures non-linear feature–class relationships that linear models miss, while inference remains a few table lookups and additions per pixel.
We tune the smoothing parameter α on the last 20% of train.in (file order), then train on all data.
If dev accuracy does not exceed label-linear (0.9613), the model fails to exploit the ordinal structure effectively.
```

```python
"""Naive Bayes for label tasks with fixed-size byte items (ordinal features).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic: no randomness, single thread. Uses Laplace smoothing α chosen on tuning split (last 20% of train.in)."""
import json, math, struct, sys

import numpy as np


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def compute_log_probs(X, y, C, V, alpha):
    """Return log_prior[C], log_cond[d][V][C] as float32."""
    d = X.shape[1]
    n = X.shape[0]
    class_counts = np.bincount(y, minlength=C).astype(np.float64)
    log_prior = np.log((class_counts + alpha) / (n + C * alpha))

    # cond_counts[j, v, c] = alpha + count of feature j = value v in class c
    cond_counts = np.full((d, V, C), alpha, dtype=np.float64)
    for c in range(C):
        mask = (y == c)
        if not np.any(mask):
            continue
        Xc = X[mask]
        for j in range(d):
            vals = Xc[:, j]
            counts = np.bincount(vals, minlength=V)
            cond_counts[j, :, c] += counts

    # denominator per class: class_counts[c] + V * alpha
    denom = class_counts + V * alpha  # shape (C,)
    log_cond = np.log(cond_counts) - np.log(denom)[None, None, :]  # (d, V, C)
    return log_prior.astype(np.float32), log_cond.astype(np.float32)


def accuracy(X, y, log_prior, log_cond):
    """Compute accuracy on given data."""
    d, V, C = log_cond.shape
    # Precompute per-feature lookup: for each feature j, a V x C matrix
    # For each sample, sum log_prior + sum_j log_cond[j, X[i,j], :]
    scores = log_prior + np.sum(log_cond[np.arange(d), X.T, :], axis=0).T  # (n, C)
    pred = np.argmax(scores, axis=1)
    return np.mean(pred == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    V = int(X.max()) + 1  # values 0..max

    # Tuning split: last 20% in file order
    n = X.shape[0]
    tuning_size = n // 5
    train_idx = np.arange(n - tuning_size)
    val_idx = np.arange(n - tuning_size, n)

    alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    best_alpha = alphas[0]
    best_acc = -1.0
    for alpha in alphas:
        log_prior, log_cond = compute_log_probs(X[train_idx], y[train_idx], C, V, alpha)
        acc = accuracy(X[val_idx], y[val_idx], log_prior, log_cond)
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    # Train on all data with best alpha
    log_prior, log_cond = compute_log_probs(X, y, C, V, best_alpha)

    # Write weights.bin: magic, d, C, V, log_prior[C], log_cond[d*V*C]
    magic = 0x4E424159  # 'YABN' in little-endian
    header = struct.pack("<4i", magic, d, C, V)
    out = header + log_prior.tobytes() + log_cond.ravel(order='C').tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Naive Bayes for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes (0..V-1); output is one byte, the argmax class (first max wins).
 * Weights format: uint32_t magic (0x4E424159), int32_t d, int32_t C, int32_t V;
 * float32 log_prior[C]; float32 log_cond[d][V][C] (row-major: j*V*C + v*C + c). */
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
    if (nw < 16 || hd[0] != 0x4E424159) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], V = hd[3];
    if (C < 1 || C > 256 || d < 1 || V < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + (long)C * 4 + (long)d * V * C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *log_prior = (const float *)(hd + 4);
    const float *log_cond = log_prior + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(scores, log_prior, sizeof(float) * C);
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            if (v >= V) v = V - 1;  // clamp just in case
            const float *lc = log_cond + ((long)j * V + v) * C;
            for (int c = 0; c < C; c++) scores[c] += lc[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```