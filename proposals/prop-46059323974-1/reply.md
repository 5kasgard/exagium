```hypothesis
Family: label-boosted-stumps (SAMME with decision stumps).
Why: Optdigits pixels are ordinal 0..16 on an 8x8 grid; single-pixel thresholds capture stroke presence/absence nonlinearly. SAMME builds an additive ensemble of depth-1 trees, each splitting on one pixel <= t, giving a strong non-linear boundary with minimal compute. At T≈50 stumps inference is ~50 compare-add ops/item, far below label-linear's 640 mul-adds.
Failure signal: Dev accuracy ≤ 0.961 (linear) or Ir/item ≥ 5000.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
T_CANDIDATES = [10, 20, 30, 50, 75, 100, 150, 200]
VAL_FRACTION = 0.2


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_samme(X, y, T, sample_weight=None):
    n, d = X.shape
    C = 10
    if sample_weight is None:
        sample_weight = np.ones(n, dtype=np.float64) / n
    stumps = []
    for _ in range(T):
        best_err = 1.0
        best_j = best_t = best_cL = best_cR = 0
        for j in range(d):
            for t in range(16):
                left = X[:, j] <= t
                right = ~left
                wL = sample_weight[left].sum()
                wR = sample_weight[right].sum()
                if wL == 0 or wR == 0:
                    continue
                cL = np.argmax([sample_weight[left & (y == c)].sum() for c in range(C)])
                cR = np.argmax([sample_weight[right & (y == c)].sum() for c in range(C)])
                pred = np.where(left, cL, cR)
                err = sample_weight[pred != y].sum()
                if err < best_err:
                    best_err, best_j, best_t, best_cL, best_cR = err, j, t, cL, cR
        if best_err >= 0.5 or best_err == 0:
            break
        alpha = math.log((1 - best_err) / best_err) + math.log(C - 1)
        left = X[:, best_j] <= best_t
        pred = np.where(left, best_cL, best_cR)
        sample_weight *= np.exp(alpha * (pred != y).astype(np.float64))
        sample_weight /= sample_weight.sum()
        stumps.append((best_j, best_t, best_cL, best_cR, alpha))
    return stumps


def predict_stumps(X, stumps):
    n = X.shape[0]
    C = 10
    scores = np.zeros((n, C), dtype=np.float64)
    for j, t, cL, cR, alpha in stumps:
        left = X[:, j] <= t
        scores[left, cL] += alpha
        scores[~left, cR] += alpha
    return scores.argmax(axis=1)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    val_size = int(n * VAL_FRACTION)
    tr = n - val_size
    X_tr, X_val = X[:tr], X[tr:]
    y_tr, y_val = y[:tr], y[tr:]

    best_T, best_acc, best_stumps = T_CANDIDATES[0], -1.0, None
    for T in T_CANDIDATES:
        stumps = train_samme(X_tr, y_tr, T)
        if not stumps:
            continue
        acc = (predict_stumps(X_val, stumps) == y_val).mean()
        if acc > best_acc:
            best_acc, best_T, best_stumps = acc, T, stumps

    final_stumps = train_samme(X, y, best_T)

    out = struct.pack("<4i", 0x424F4F53, len(final_stumps), X.shape[1], 10)
    for j, t, cL, cR, alpha in final_stumps:
        out += struct.pack("<4Bf", j, t, cL, cR, alpha)
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x424F4F53) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int T = hd[1], d = hd[2], C = hd[3];
    if (C != 10 || d != 64 || T < 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    if (nw != 16 + (long)T * 8) { fprintf(stderr, "bad weights size\n"); return 3; }
    unsigned char *stump_data = wb + 16;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[10];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = 0.0f;
        for (int s = 0; s < T; s++) {
            unsigned char *st = stump_data + s * 8;
            uint8_t j = st[0], t = st[1], cL = st[2], cR = st[3];
            float alpha = *(float*)(st + 4);
            if (xi[j] <= t) scores[cL] += alpha;
            else scores[cR] += alpha;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```