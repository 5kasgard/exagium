```hypothesis
Family: label-stumps (gradient-boosted decision stumps on raw byte features).
Why: The Adult data mixes ordinal and categorical columns; a linear model ignores threshold effects and feature interactions. Boosted stumps capture non-linear splits (e.g., "age > 50", "education == Bachelors") with minimal parameters. With 32 stumps, inference is ~32 branches + adds per item, cheaper than label-linear's 28 multiply-adds per class, while modeling non-linearities.
Failure signal: dev accuracy ≤ 0.8524 (label-linear) or Ir/item ≥ 719 (label-linear cost).
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1
N_STUMPS = 32
LR = 0.1
L2_REG = 1.0


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
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.float32)
    n, d = X.shape
    types = card["input"]["types"]

    p = y.mean()
    base_logit = math.log(p / (1 - p + 1e-12))
    logits = np.full(n, base_logit, dtype=np.float32)

    stumps = []

    for _ in range(N_STUMPS):
        probs = 1.0 / (1.0 + np.exp(-logits))
        grad = y - probs
        hess = probs * (1 - probs)

        best_gain = -1.0
        best_feat = 0
        best_thresh = 0
        best_is_cat = 0
        best_left = 0.0
        best_right = 0.0

        for j in range(d):
            col = X[:, j]
            is_cat = (types[j] == "categorical")

            g_hist = np.bincount(col, weights=grad, minlength=256).astype(np.float32)
            h_hist = np.bincount(col, weights=hess, minlength=256).astype(np.float32)

            if is_cat:
                G_all = grad.sum()
                H_all = hess.sum()
                for v in range(256):
                    G_v = g_hist[v]
                    H_v = h_hist[v]
                    if H_v < 1e-6:
                        continue
                    G_r = G_all - G_v
                    H_r = H_all - H_v
                    if H_r < 1e-6:
                        continue
                    gain = (G_v * G_v) / (H_v + L2_REG) + (G_r * G_r) / (H_r + L2_REG) - (G_all * G_all) / (H_all + L2_REG)
                    if gain > best_gain:
                        best_gain = gain
                        best_feat = j
                        best_thresh = v
                        best_is_cat = 1
                        best_left = -G_v / (H_v + L2_REG) * LR
                        best_right = -G_r / (H_r + L2_REG) * LR
            else:
                G_cum = 0.0
                H_cum = 0.0
                G_all = grad.sum()
                H_all = hess.sum()
                for v in range(255):
                    G_cum += g_hist[v]
                    H_cum += h_hist[v]
                    if H_cum < 1e-6 or (H_all - H_cum) < 1e-6:
                        continue
                    G_left = G_cum
                    H_left = H_cum
                    G_right = G_all - G_cum
                    H_right = H_all - H_cum
                    gain = (G_left * G_left) / (H_left + L2_REG) + (G_right * G_right) / (H_right + L2_REG) - (G_all * G_all) / (H_all + L2_REG)
                    if gain > best_gain:
                        best_gain = gain
                        best_feat = j
                        best_thresh = v
                        best_is_cat = 0
                        best_left = -G_left / (H_left + L2_REG) * LR
                        best_right = -G_right / (H_right + L2_REG) * LR

        if best_gain <= 0:
            break

        stumps.append((best_feat, best_thresh, best_is_cat, best_left, best_right))
        col = X[:, best_feat]
        if best_is_cat:
            mask = (col == best_thresh)
        else:
            mask = (col <= best_thresh)
        logits[mask] += best_left
        logits[~mask] += best_right

    out = struct.pack("<Iiif", 0x5354554D, d, len(stumps), base_logit)
    for feat, thresh, is_cat, left, right in stumps:
        out += struct.pack("<BBBBff", feat, thresh, is_cat, 0, left, right)
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
    uint32_t magic = wb[0] | wb[1] << 8 | wb[2] << 16 | wb[3] << 24;
    if (magic != 0x5354554D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t d = *(int32_t*)(wb + 4);
    int32_t n_stumps = *(int32_t*)(wb + 8);
    float base_logit = *(float*)(wb + 12);
    long expected = 16 + (long)n_stumps * 16;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float logit = base_logit;
        for (int s = 0; s < n_stumps; s++) {
            unsigned char *st = wb + 16 + s * 16;
            int feat = st[0];
            int thresh = st[1];
            int is_cat = st[2];
            float left = *(float*)(st + 4);
            float right = *(float*)(st + 8);
            int val = xi[feat];
            int cond = is_cat ? (val == thresh) : (val <= thresh);
            logit += cond ? left : right;
        }
        float prob = 1.0f / (1.0f + expf(-logit));
        unsigned char cls = (prob > 0.5f) ? 1 : 0;
        put_rec(stdout, &cls, 1);
    }
    return 0;
}
```