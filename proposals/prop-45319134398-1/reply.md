```hypothesis
Family: label-adaboost (SAMME with decision stumps). Optdigits pixels are ordinal 0..16; stumps threshold single pixels, capturing non-linear boundaries cheaply. 100 stumps should exceed linear (0.9613) and approach mlp16 (0.9770) at ~2000 Ir/item (vs 5058 for linear). Failure: dev accuracy ≤ 0.962 or Ir/item ≥ 6000.
```
```python
import json, struct, sys
import numpy as np

N_STUMPS = 100
SEED = 1

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
    n, d = X.shape
    C = card["classes"]
    assert d == 64 and C == 10

    rng = np.random.default_rng(SEED)
    weights = np.ones(n, dtype=np.float64) / n
    stumps = []

    for _ in range(N_STUMPS):
        total_w = weights.sum()
        best_err = 1.0
        best_feat = 0
        best_thresh = 0
        best_cl = 0
        best_cr = 0

        for j in range(d):
            Xj = X[:, j]
            W_class_val = np.zeros((C, 17), dtype=np.float64)
            for c in range(C):
                mask = (y == c)
                if mask.any():
                    W_class_val[c] = np.bincount(Xj[mask], weights=weights[mask], minlength=17)

            W_left = np.cumsum(W_class_val[:, :16], axis=1)
            total_per_class = W_class_val.sum(axis=1)
            W_right = total_per_class[:, None] - W_left

            best_left_class = W_left.argmax(axis=0)
            best_right_class = W_right.argmax(axis=0)
            best_left_weight = W_left.max(axis=0)
            best_right_weight = W_right.max(axis=0)

            correct_weight = best_left_weight + best_right_weight
            err = 1.0 - correct_weight / total_w

            t_idx = err.argmin()
            feat_err = err[t_idx]
            if feat_err < best_err:
                best_err = feat_err
                best_feat = j
                best_thresh = t_idx
                best_cl = best_left_class[t_idx]
                best_cr = best_right_class[t_idx]

        if best_err >= 0.9 or best_err <= 1e-12:
            break

        alpha = np.log((1 - best_err) / best_err) + np.log(C - 1)
        stumps.append((best_feat, best_thresh, best_cl, best_cr, alpha))

        pred = np.where(X[:, best_feat] <= best_thresh, best_cl, best_cr)
        weights *= np.exp(alpha * (pred != y))
        weights /= weights.sum()

    out = struct.pack("<Iiii", 0x41444142, len(stumps), C, d)
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
    uint32_t magic = *(uint32_t*)wb;
    if (magic != 0x41444142) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t *hd = (int32_t*)(wb + 4);
    int n_stumps = hd[0], C = hd[1], d = hd[2];
    if (C != 10 || d != 64 || n_stumps <= 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + (long)n_stumps * 8;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    struct Stump { uint8_t feat, thresh, cl, cr; float alpha; };
    struct Stump *stumps = (struct Stump*)(wb + 16);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[10];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = 0.0f;
        for (int s = 0; s < n_stumps; s++) {
            struct Stump *st = &stumps[s];
            uint8_t val = xi[st->feat];
            if (val <= st->thresh) scores[st->cl] += st->alpha;
            else scores[st->cr] += st->alpha;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```