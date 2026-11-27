```hypothesis
Family: label-dqda (Diagonal Quadratic Discriminant Analysis).
Why: Optdigits pixels are ordinal counts (0–16) with class-conditional variances; diagonal QDA models per-class per-feature variance, capturing heteroscedasticity that linear models miss, while being faster than MLPs (no hidden layer, just 4 flops/feature/class).
Failure: If dev accuracy ≤ 0.9613 (label-linear) or CPU cost ≥ label-mlp16.
```
```python
import json, math, struct, sys
import numpy as np

EPS_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, C, eps):
    n, d = X.shape
    log_prior = np.zeros(C, np.float32)
    mean = np.zeros((C, d), np.float32)
    half_inv_var = np.zeros((C, d), np.float32)
    half_log_inv_var = np.zeros((C, d), np.float32)
    for c in range(C):
        mask = (y == c)
        nc = mask.sum()
        if nc == 0:
            log_prior[c] = -100.0
            half_inv_var[c] = 0.5 / eps
            half_log_inv_var[c] = 0.5 * math.log(1.0 / eps)
            continue
        Xc = X[mask]
        log_prior[c] = math.log(nc / n)
        mean[c] = Xc.mean(axis=0).astype(np.float32)
        var = Xc.var(axis=0, ddof=0).astype(np.float32)
        var = np.maximum(var, eps)
        inv_var = 1.0 / var
        half_inv_var[c] = 0.5 * inv_var
        half_log_inv_var[c] = 0.5 * np.log(inv_var).astype(np.float32)
    return log_prior, mean, half_inv_var, half_log_inv_var

def eval_model(X, y, log_prior, mean, half_inv_var, half_log_inv_var):
    n, d = X.shape
    C = log_prior.shape[0]
    correct = 0
    for i in range(n):
        x = X[i]
        best_c, best_score = 0, -1e30
        for c in range(C):
            score = log_prior[c]
            diff = x - mean[c]
            score += np.sum(half_log_inv_var[c] - diff * diff * half_inv_var[c])
            if score > best_score:
                best_score = score
                best_c = c
        if best_c == y[i]:
            correct += 1
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = len(X_all)
    split = int(n * 0.8)
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]
    X_train_f = X_train.astype(np.float32)
    X_tune_f = X_tune.astype(np.float32)
    best_eps = EPS_GRID[0]
    best_acc = -1.0
    for eps in EPS_GRID:
        log_prior, mean, half_inv_var, half_log_inv_var = train_model(X_train_f, y_train, C, eps)
        acc = eval_model(X_tune_f, y_tune, log_prior, mean, half_inv_var, half_log_inv_var)
        if acc > best_acc:
            best_acc = acc
            best_eps = eps
    X_all_f = X_all.astype(np.float32)
    log_prior, mean, half_inv_var, half_log_inv_var = train_model(X_all_f, y_all, C, best_eps)
    magic = 0x44514441
    out = struct.pack("<3i", magic, d, C)
    out += log_prior.astype("<f4").tobytes()
    out += mean.astype("<f4").tobytes()
    out += half_inv_var.astype("<f4").tobytes()
    out += half_log_inv_var.astype("<f4").tobytes()
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
    if (nw < 12 || hd[0] != 0x44514441) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 12 + 4L * (C + 3L * C * d);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 3);
    const float *log_prior = p;
    const float *mean = p + C;
    const float *half_inv_var = p + C + (long)C * d;
    const float *half_log_inv_var = p + C + 2L * C * d;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char best;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float best_score = -1e30f;
        best = 0;
        for (int c = 0; c < C; c++) {
            float score = log_prior[c];
            const float *mc = mean + (long)c * d;
            const float *hiv = half_inv_var + (long)c * d;
            const float *hliv = half_log_inv_var + (long)c * d;
            for (int j = 0; j < d; j++) {
                float diff = (float)xi[j] - mc[j];
                score += hliv[j] - diff * diff * hiv[j];
            }
            if (score > best_score) {
                best_score = score;
                best = (unsigned char)c;
            }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```