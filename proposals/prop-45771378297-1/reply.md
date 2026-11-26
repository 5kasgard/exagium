```hypothesis
Family: label-adaboost-stumps (one-vs-all AdaBoost with decision stumps).
Why: Optdigits pixels have clear ordinal structure (0-16); axis-aligned stumps capture stroke/curve patterns non-linearly. One-vs-all with ~40 stumps/class gives ~400 weak learners—similar inference cost to label-linear (hundreds of ops) but with adaptive decision boundaries. Deterministic threshold search (midpoints of 0..16) and fixed seed ensure reproducibility.
Failure: dev accuracy ≤ 0.9613 (label-linear) or Ir/item ≥ 5058 at that accuracy.
```
```python
"""label-adaboost-stumps: one-vs-all AdaBoost with decision stumps for ordinal byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy. Exports stumps per class."""
import json, struct, sys
import numpy as np

SEED = 1
STUMPS_PER_CLASS = 40  # T
MAGIC = 0x41425331  # 'ABS1'

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X_bytes.shape
    C = card["classes"]

    # Ordinal z-scoring per feature (like label-linear) for stable thresholds
    X = X_bytes.astype(np.float32)
    for j in range(d):
        mu = X[:, j].mean()
        sd = X[:, j].std()
        if sd > 0:
            X[:, j] = (X[:, j] - mu) / sd
        else:
            X[:, j] = 0.0

    rng = np.random.default_rng(SEED)
    # Pre-compute candidate thresholds per feature (midpoints of sorted unique values)
    # Since original values are 0..16, thresholds in original space are 0.5..15.5
    # But we work in z-scored space; compute thresholds from z-scored unique values.
    thresholds = []
    for j in range(d):
        uniq = np.unique(X[:, j])
        if len(uniq) > 1:
            th = (uniq[:-1] + uniq[1:]) / 2.0
        else:
            th = np.array([uniq[0]])
        thresholds.append(th.astype(np.float32))

    all_stumps = []  # list per class: (feat, thresh, polarity, alpha)
    for k in range(C):
        # Binary labels: +1 for class k, -1 otherwise
        y_bin = np.where(y == k, 1.0, -1.0).astype(np.float32)
        w = np.ones(n, dtype=np.float32) / n
        stumps_k = []
        for t in range(STUMPS_PER_CLASS):
            best_err = 1.0
            best_j = best_th = best_pol = 0
            # Exhaustive search over features and thresholds (d=64, ~16 thresholds each)
            for j in range(d):
                xj = X[:, j]
                for th in thresholds[j]:
                    for pol in (1.0, -1.0):
                        pred = np.where(pol * (xj > th) > 0, 1.0, -1.0)
                        err = np.sum(w * (pred != y_bin))
                        if err < best_err:
                            best_err = err
                            best_j, best_th, best_pol = j, th, pol
            # Avoid zero/one error
            eps = 1e-10
            err = max(eps, min(best_err, 1.0 - eps))
            alpha = 0.5 * np.log((1.0 - err) / err)
            # Update weights
            pred_best = np.where(best_pol * (X[:, best_j] > best_th) > 0, 1.0, -1.0)
            w *= np.exp(-alpha * y_bin * pred_best)
            w /= np.sum(w)
            stumps_k.append((best_j, best_th, best_pol, alpha))
        all_stumps.append(stumps_k)

    # Pack weights: magic, C, T, d, then per class per stump: uint8 feat, float32 thresh, int8 pol, float32 alpha
    # But threshold is float32; to keep it simple and deterministic, store as float32.
    # However, infer.c expects fixed-size records. Use float32 for threshold.
    out = struct.pack("<4I", MAGIC, C, STUMPS_PER_CLASS, d)
    for k in range(C):
        for (feat, thresh, pol, alpha) in all_stumps[k]:
            out += struct.pack("<Bfbf", feat, thresh, int(pol), alpha)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-adaboost-stumps inference: ./infer weights.bin in.rec N
 * One-vs-all AdaBoost with decision stumps. Each stump: feature, threshold, polarity, alpha.
 * Output: argmax over classes of sum(alpha * (polarity * (x[feat] > threshold) ? 1 : -1)). */
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
    if (hd[0] != 0x41425331) { fprintf(stderr, "bad weights magic\n"); return 3; }
    uint32_t C = hd[1], T = hd[2], d = hd[3];
    if (C < 1 || C > 256 || d < 1 || d > 256 || T == 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    size_t expected = 16 + C * T * (1 + 4 + 1 + 4); /* feat(u8), thresh(f32), pol(i8), alpha(f32) */
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld expect %zu\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = calloc(C, sizeof(float));
    const unsigned char *p = wb + 16;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        memset(scores, 0, C * sizeof(float));
        const unsigned char *stump_ptr = p;
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t t = 0; t < T; t++) {
                uint8_t feat = stump_ptr[0];
                float thresh = *(const float *)(stump_ptr + 1);
                int8_t pol = *(const int8_t *)(stump_ptr + 5);
                float alpha = *(const float *)(stump_ptr + 6);
                stump_ptr += 10;
                float xv = (float)xi[feat];
                float vote = (pol * (xv > thresh) > 0) ? 1.0f : -1.0f;
                scores[c] += alpha * vote;
            }
        }
        unsigned char best = 0;
        for (uint32_t c = 1; c < C; c++) if (scores[c] > scores[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```