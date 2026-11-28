```hypothesis
Family: label-stumpboost (SAMME-boosted decision stumps). 
Why: Optdigits features are 64 ordinal pixels (0..16); stumps on single pixels capture localized stroke patterns, and SAMME combines them into a strong multi-class learner. Inference is just T threshold checks and additions (T~200), far fewer ops than label-linear's 640 MACs, so it should be faster and more accurate. 
Failure: If dev accuracy ≤ 0.9613 (label-linear) or Ir/item ≥ 5000, the family does not deliver on its promise.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
T_GRID = [50, 100, 200, 300]
NU_GRID = [0.1, 0.5, 1.0]


def read_records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_samme_stumps(X, y, C, T, nu):
    n, d = X.shape
    w = np.ones(n, dtype=np.float32) / n
    stumps = []
    for _ in range(T):
        best_err = 1.0
        best_feat = best_thresh = best_cl = best_cr = 0
        for feat in range(d):
            Wvc = np.zeros((17, C), dtype=np.float32)
            for c in range(C):
                mask = (y == c)
                if mask.any():
                    np.add.at(Wvc[:, c], X[mask, feat], w[mask])
            Wvc_cum = Wvc.cumsum(axis=0)
            total_wc = Wvc.sum(axis=0)
            total_w = w.sum()
            left_w = Wvc_cum[:16]
            right_w = total_wc - left_w
            left_sum = left_w.sum(axis=1)
            right_sum = total_w - left_sum
            valid = (left_sum > 0) & (right_sum > 0)
            if not valid.any():
                continue
            left_major = left_w.argmax(axis=1)
            right_major = right_w.argmax(axis=1)
            left_major_w = left_w[np.arange(16), left_major]
            right_major_w = right_w[np.arange(16), right_major]
            err = (left_sum - left_major_w) + (right_sum - right_major_w)
            err[~valid] = 1.0
            idx = err.argmin()
            if err[idx] < best_err:
                best_err = err[idx]
                best_feat = feat
                best_thresh = idx
                best_cl = left_major[idx]
                best_cr = right_major[idx]
        if best_err >= 0.5 - 1e-9:
            break
        alpha = math.log((1 - best_err) / best_err) + math.log(C - 1)
        alpha *= nu
        pred = np.where(X[:, best_feat] <= best_thresh, best_cl, best_cr)
        w *= np.exp(alpha * (pred != y).astype(np.float32))
        w /= w.sum()
        stumps.append((int(best_feat), int(best_thresh), int(best_cl), int(best_cr), float(alpha)))
    return stumps


def predict_stumps(X, stumps, C):
    n = X.shape[0]
    scores = np.zeros((n, C), dtype=np.float32)
    for feat, thresh, cl, cr, alpha in stumps:
        pred = np.where(X[:, feat] <= thresh, cl, cr)
        scores[np.arange(n), pred] += alpha
    return scores.argmax(axis=1)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(read_records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in read_records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]

    val_size = n // 5
    tr_idx = np.arange(n - val_size)
    va_idx = np.arange(n - val_size, n)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    best_acc = -1.0
    best_T, best_nu = T_GRID[0], NU_GRID[0]
    for T in T_GRID:
        for nu in NU_GRID:
            stumps = train_samme_stumps(X_tr, y_tr, C, T, nu)
            preds = predict_stumps(X_va, stumps, C)
            acc = (preds == y_va).mean()
            if acc > best_acc:
                best_acc = acc
                best_T, best_nu = T, nu

    stumps = train_samme_stumps(X, y, C, best_T, best_nu)

    out = struct.pack("<4i", 0x5354554D, len(stumps), C, d)
    for feat, thresh, cl, cr, alpha in stumps:
        out += struct.pack("<BBBBf", feat, thresh, cl, cr, alpha)
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
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | (uint32_t)wb[3] << 24;
    if (magic != 0x5354554D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int T = wb[4] | wb[5] << 8 | wb[6] << 16 | (uint32_t)wb[7] << 24;
    int C = wb[8] | wb[9] << 8 | wb[10] << 16 | (uint32_t)wb[11] << 24;
    int d = wb[12] | wb[13] << 8 | wb[14] << 16 | (uint32_t)wb[15] << 24;
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + (long)T * 8;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const unsigned char *stumps = wb + 16;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = calloc(C, sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(scores, 0, C * sizeof(float));
        for (int t = 0; t < T; t++) {
            int feat = stumps[t * 8 + 0];
            int thresh = stumps[t * 8 + 1];
            int cl = stumps[t * 8 + 2];
            int cr = stumps[t * 8 + 3];
            float alpha = *(const float*)(stumps + t * 8 + 4);
            int pred = (xi[feat] <= thresh) ? cl : cr;
            scores[pred] += alpha;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```