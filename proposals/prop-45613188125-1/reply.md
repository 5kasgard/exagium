```hypothesis
Family: label-stump (AdaBoost.SAMME with decision stumps).
Why: Each stump thresholds one pixel (0..16) and votes for a class; 100 stumps capture pixel-wise non-linearities the linear model misses while inference stays ~100 compares + adds (faster than linear's 640 multiply-adds). Deterministic via fixed seed and ordered tie-breaking.
Failure signal: dev accuracy ≤ 0.9613 (label-linear) or Ir/item ≥ 5058.
```
```python
"""label-stump: AdaBoost.SAMME with decision stumps for ordinal byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy. Tunes num_stumps on last 20% of train."""
import json, math, struct, sys

import numpy as np

SEED = 1
CANDIDATE_STUMPS = [50, 100, 150, 200]  # tuned on last 20%
MAGIC = 0x504D5453  # 'STMP' little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_stumps(X, y, C, max_stumps, rng):
    """Returns list of (feat, thresh, cls_t, cls_f, alpha)."""
    n = X.shape[0]
    D = np.full(n, 1.0 / n, dtype=np.float64)
    stumps = []
    for _ in range(max_stumps):
        best_err = 1.0
        best = None
        # Search all features and thresholds
        for f in range(X.shape[1]):
            v = X[:, f]
            # thresholds 0..15 (since >16 is always false)
            for t in range(16):
                mask = v > t
                # Weighted class counts for each branch
                w_t = np.zeros(C, dtype=np.float64)
                w_f = np.zeros(C, dtype=np.float64)
                for c in range(C):
                    w_t[c] = D[mask & (y == c)].sum()
                    w_f[c] = D[~mask & (y == c)].sum()
                c_t = int(w_t.argmax())
                c_f = int(w_f.argmax())
                correct = w_t[c_t] + w_f[c_f]
                err = 1.0 - correct
                if err < best_err:
                    best_err = err
                    best = (f, t, c_t, c_f)
        if best is None or best_err >= 1.0 - 1.0 / C:
            break
        f, t, c_t, c_f = best
        alpha = math.log((1.0 - best_err) / best_err) + math.log(C - 1)
        pred = np.where(X[:, f] > t, c_t, c_f)
        D *= np.exp(alpha * (pred != y))
        D /= D.sum()
        stumps.append((f, t, c_t, c_f, alpha))
    return stumps


def accuracy(X, y, stumps, C):
    n = X.shape[0]
    scores = np.zeros((n, C), dtype=np.float64)
    for f, t, c_t, c_f, alpha in stumps:
        pred_t = (X[:, f] > t)
        scores[pred_t, c_t] += alpha
        scores[~pred_t, c_f] += alpha
    return float((scores.argmax(axis=1) == y).mean())


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    # Split: last 20% for tuning
    n = X.shape[0]
    n_tune = n // 5
    idx_train = np.arange(n - n_tune)
    idx_tune = np.arange(n - n_tune, n)
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_tu, y_tu = X[idx_tune], y[idx_tune]

    # Tune num_stumps
    best_stumps = None
    best_acc = -1.0
    best_k = CANDIDATE_STUMPS[0]
    for k in CANDIDATE_STUMPS:
        st = train_stumps(X_tr, y_tr, C, k, rng)
        acc = accuracy(X_tu, y_tu, st, C)
        if acc > best_acc:
            best_acc = acc
            best_stumps = st
            best_k = k

    # Retrain on all data with best_k
    final_stumps = train_stumps(X, y, C, best_k, rng)

    # Pack weights: magic, d, C, num_stumps, then per stump: uint8 feat, thresh, cls_t, cls_f, float32 alpha
    num_stumps = len(final_stumps)
    header = struct.pack("<4i", MAGIC, d, C, num_stumps)
    body = bytearray()
    for f, t, c_t, c_f, alpha in final_stumps:
        body += struct.pack("<4Bf", f, t, c_t, c_f, alpha)
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
/* label-stump inference: ./infer weights.bin in.rec N
 * Each stump: if x[feat] > thresh then cls_t else cls_f, weighted by alpha. */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x504D5453) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = (int)(wb[4] | wb[5] << 8 | wb[6] << 16 | wb[7] << 24);
    int C = (int)(wb[8] | wb[9] << 8 | wb[10] << 16 | wb[11] << 24);
    int S = (int)(wb[12] | wb[13] << 8 | wb[14] << 16 | wb[15] << 24);
    if (C < 1 || C > 256 || d != 64 || nw != 16 + (long)S * 8) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const unsigned char *st = wb + 16;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = calloc(C, sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(scores, 0, C * sizeof(float));
        for (int s = 0; s < S; s++) {
            uint8_t feat = st[0];
            uint8_t thresh = st[1];
            uint8_t cls_t = st[2];
            uint8_t cls_f = st[3];
            float alpha = *(const float *)(st + 4);
            st += 8;
            if (xi[feat] > thresh) scores[cls_t] += alpha;
            else scores[cls_f] += alpha;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```