```hypothesis
Family: label-gnb (Gaussian Naive Bayes with variance smoothing).  
Why: Optdigits pixels are ordinal counts 0–16; class-conditional Gaussians with a tuned smoothing constant are a strong, fast baseline for this domain. The model has no iterative training, so it is deterministic and extremely cheap at inference.  
Failure signal: If dev accuracy ≤ 0.9613 (label-linear) or inference cost exceeds label-linear, the independence assumption is too strong for this data.
```

```python
"""Gaussian Naive Bayes for fixed-size byte items (label-gnb).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic, single-threaded, stdlib+numpy only.
Weights format: magic 0x474E4231, C, d, epsilon; float32 log_prior[C], mean[C*d], inv_var[C*d], log_var[C*d]."""
import json, math, struct, sys

import numpy as np

EPS_GRID = [1e-9, 1e-6, 1e-3, 0.1, 1.0, 10.0]
SEED = 1


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_gnb(X, y, C, eps):
    n, d = X.shape
    Xf = X.astype(np.float64)
    log_prior = np.zeros(C, np.float64)
    mean = np.zeros((C, d), np.float64)
    var = np.zeros((C, d), np.float64)
    for c in range(C):
        mask = (y == c)
        nc = mask.sum()
        log_prior[c] = math.log(nc / n)
        Xc = Xf[mask]
        mean[c] = Xc.mean(axis=0)
        var[c] = Xc.var(axis=0) + eps
    inv_var = 1.0 / var
    log_var = np.log(var)
    return log_prior, mean, inv_var, log_var


def predict_gnb(X, log_prior, mean, inv_var, log_var):
    n, d = X.shape
    C = log_prior.shape[0]
    Xf = X.astype(np.float64)
    scores = np.zeros((n, C), np.float64)
    const = -0.5 * math.log(2 * math.pi)
    for c in range(C):
        diff = Xf - mean[c]
        scores[:, c] = log_prior[c] + np.sum(const - 0.5 * log_var[c] - 0.5 * diff * diff * inv_var[c], axis=1)
    return np.argmax(scores, axis=1)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    best_eps, best_acc = EPS_GRID[0], -1.0
    for eps in EPS_GRID:
        lp, mn, iv, lv = train_gnb(X_tr, y_tr, C, eps)
        pred = predict_gnb(X_va, lp, mn, iv, lv)
        acc = (pred == y_va).mean()
        if acc > best_acc:
            best_acc, best_eps = acc, eps

    # Retrain on all data
    lp, mn, iv, lv = train_gnb(X, y, C, best_eps)

    # Pack weights
    magic = 0x474E4231
    out = struct.pack("<4i", magic, C, d, 0)  # last int placeholder for epsilon index? we store epsilon as float
    out += struct.pack("<f", best_eps)
    out += lp.astype(np.float32).tobytes()
    out += mn.astype(np.float32).tobytes()
    out += iv.astype(np.float32).tobytes()
    out += lv.astype(np.float32).tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Gaussian Naive Bayes inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class index (argmax log posterior). */
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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x474E4231) { fprintf(stderr, "bad magic\n"); return 3; }
    int C = hd[1], d = hd[2];
    float eps = *(float *)(hd + 3);
    const float *lp = (const float *)(hd + 4);
    const float *mn = lp + C;
    const float *iv = mn + (long)C * d;
    const float *lv = iv + (long)C * d;
    long expected = 20 + 4L * (C + 3L * C * d);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    const float LOG_2PI = logf(2.0f * 3.14159265358979323846f);
    float *scores = malloc(C * sizeof(float));
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = lp[c];
        for (int j = 0; j < d; j++) {
            float xv = xi[j];
            for (int c = 0; c < C; c++) {
                long idx = (long)c * d + j;
                float diff = xv - mn[idx];
                scores[c] += -0.5f * (LOG_2PI + lv[idx] + diff * diff * iv[idx]);
            }
        }
        best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```