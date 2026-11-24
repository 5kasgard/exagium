```hypothesis
Family: label-adaboost (AdaBoost.SAMME with decision stumps on ordinal features).
Why: 8x8 optdigits pixels (0..16) are strongly predictive individually; stumps capture single-pixel thresholds. SAMME handles 10 classes natively. Boosting combines many weak stumps into a strong classifier. Expected to beat label-linear (96.13%) by modeling non-linear feature interactions additively, at lower CPU cost (T≈200 stumps: ~200 integer compares + adds per item vs label-linear's table lookups).
Failure signal: dev accuracy ≤ 96.0% (does not beat label-linear) or instruction count > label-linear.
```
```python
import json, struct, sys
import numpy as np

MAGIC = 0x42414441  # 'ADAB' little-endian
T_MAX = 300
SEED = 1

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
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
    n = len(X)
    np.random.seed(SEED)

    # Tuning split: last 20% in file order
    n_tune = n // 5
    train_idx = np.arange(n - n_tune)
    tune_idx = np.arange(n - n_tune, n)

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]
    n_tr = len(X_tr)

    # Pass 1: train on train_idx, validate on tune_idx to pick T
    w = np.ones(n_tr, dtype=np.float64) / n_tr
    tune_scores = np.zeros((len(tune_idx), C), dtype=np.float64)
    best_tune_acc = -1.0
    best_T = 1
    stumps_pass1 = []

    for t in range(T_MAX):
        best_err = 1.0
        best_stump = None

        for j in range(d):
            counts = np.zeros((C, 17), dtype=np.float64)
            for c in range(C):
                mask = (y_tr == c)
                if np.any(mask):
                    vals = X_tr[mask, j]
                    w_c = w[mask]
                    counts[c] = np.bincount(vals, weights=w_c, minlength=17)

            left_sums = np.cumsum(counts[:, :16], axis=1)
            total_sums = counts.sum(axis=1)
            right_sums = total_sums[:, None] - left_sums

            c_left = left_sums.argmax(axis=0)
            c_right = right_sums.argmax(axis=0)

            correct = left_sums[c_left, np.arange(16)] + right_sums[c_right, np.arange(16)]
            err = 1.0 - correct / w.sum()

            min_idx = err.argmin()
            if err[min_idx] < best_err:
                best_err = err[min_idx]
                best_stump = (j, min_idx, c_left[min_idx], c_right[min_idx])

        if best_err >= 0.5 or best_stump is None:
            break

        feat, thresh, c_left, c_right = best_stump
        alpha = 0.5 * np.log((1 - best_err) / best_err) + np.log(C - 1)
        stumps_pass1.append((feat, thresh, c_left, c_right, alpha))

        pred = np.where(X_tr[:, feat] <= thresh, c_left, c_right)
        w *= np.exp(alpha * (pred != y_tr))
        w /= w.sum()

        tune_pred = np.where(X_tune[:, feat] <= thresh, c_left, c_right)
        tune_scores[np.arange(len(tune_idx)), tune_pred] += alpha
        tune_pred_labels = tune_scores.argmax(axis=1)
        tune_acc = (tune_pred_labels == y_tune).mean()

        if tune_acc > best_tune_acc:
            best_tune_acc = tune_acc
            best_T = len(stumps_pass1)

    # Pass 2: train on ALL data for best_T iterations
    w = np.ones(n, dtype=np.float64) / n
    final_stumps = []

    for t in range(best_T):
        best_err = 1.0
        best_stump = None

        for j in range(d):
            counts = np.zeros((C, 17), dtype=np.float64)
            for c in range(C):
                mask = (y == c)
                if np.any(mask):
                    vals = X[mask, j]
                    w_c = w[mask]
                    counts[c] = np.bincount(vals, weights=w_c, minlength=17)

            left_sums = np.cumsum(counts[:, :16], axis=1)
            total_sums = counts.sum(axis=1)
            right_sums = total_sums[:, None] - left_sums

            c_left = left_sums.argmax(axis=0)
            c_right = right_sums.argmax(axis=0)

            correct = left_sums[c_left, np.arange(16)] + right_sums[c_right, np.arange(16)]
            err = 1.0 - correct / w.sum()

            min_idx = err.argmin()
            if err[min_idx] < best_err:
                best_err = err[min_idx]
                best_stump = (j, min_idx, c_left[min_idx], c_right[min_idx])

        if best_err >= 0.5 or best_stump is None:
            best_T = t
            break

        feat, thresh, c_left, c_right = best_stump
        alpha = 0.5 * np.log((1 - best_err) / best_err) + np.log(C - 1)
        final_stumps.append((feat, thresh, c_left, c_right, alpha))

        pred = np.where(X[:, feat] <= thresh, c_left, c_right)
        w *= np.exp(alpha * (pred != y))
        w /= w.sum()

    # Write weights.bin
    T = len(final_stumps)
    out = struct.pack("<4i", MAGIC, T, d, C)
    for feat, thresh, c_left, c_right, alpha in final_stumps:
        out += struct.pack("<BBBBf", feat, thresh, c_left, c_right, alpha)
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x42414441) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int T = hd[1], d = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    if (nw != 16 + (long)T * 8) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *stump_data = wb + 16;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = calloc(C, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(scores, 0, C * sizeof(float));
        for (int t = 0; t < T; t++) {
            unsigned char *s = stump_data + t * 8;
            uint8_t feat = s[0];
            uint8_t thresh = s[1];
            uint8_t c_left = s[2];
            uint8_t c_right = s[3];
            float alpha; memcpy(&alpha, s + 4, 4);
            int pred = (xi[feat] <= thresh) ? c_left : c_right;
            scores[pred] += alpha;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```