```hypothesis
Family: label-adaboost (One-vs-Rest AdaBoost with decision stumps on raw pixel values).
Why: Decision stumps are extremely fast at inference (integer compare + conditional add); 20 stumps/class × 10 classes ≈ 200 ops/item, far below label-linear's 640 float adds. On ordinal 0–16 pixels, stumps capture non-linear boundaries that linear models miss, often reaching >96% on optdigits.
Failure: If dev accuracy < 0.955, the stump ensemble is too weak for this task.
```

```python
# train.py
import json, math, struct, sys
import numpy as np

T = 20  # stumps per class
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

    all_stumps = []  # list per class of (pixel, threshold, alpha, polarity)
    rng = np.random.default_rng(SEED)

    for c in range(C):
        binary_y = (y == c).astype(np.float64) * 2 - 1  # +1 / -1
        pos_mask = (binary_y == 1)
        neg_mask = (binary_y == -1)
        sw = np.ones(n, dtype=np.float64) / n
        stumps = []
        for _ in range(T):
            best_err = 1.0
            best_j = 0
            best_k = 0
            best_pol = 1
            # Scan all pixels
            for j in range(d):
                xj = X[:, j]
                # Weighted counts per pixel value 0..16
                w_pos = np.zeros(17, dtype=np.float64)
                w_neg = np.zeros(17, dtype=np.float64)
                np.add.at(w_pos, xj[pos_mask], sw[pos_mask])
                np.add.at(w_neg, xj[neg_mask], sw[neg_mask])
                cum_pos = np.cumsum(w_pos)
                cum_neg = np.cumsum(w_neg)
                total_pos = cum_pos[-1]
                total_neg = cum_neg[-1]
                # Thresholds at k+0.5 for k=0..15
                for k in range(16):
                    # Polarity +1: predict +1 if x<=k else -1
                    err = cum_neg[k] + (total_pos - cum_pos[k])
                    if err < best_err:
                        best_err, best_j, best_k, best_pol = err, j, k, 1
                    # Polarity -1: predict -1 if x<=k else +1
                    err_flip = cum_pos[k] + (total_neg - cum_neg[k])
                    if err_flip < best_err:
                        best_err, best_j, best_k, best_pol = err_flip, j, k, -1
            alpha = 0.5 * math.log((1.0 - best_err) / best_err) if best_err > 0 else 10.0
            stumps.append((best_j, best_k, alpha, best_pol))
            # Update weights
            pred = np.where(X[:, best_j] <= best_k, 1, -1) * best_pol
            sw *= np.exp(-alpha * binary_y * pred)
            sw /= sw.sum()
        all_stumps.append(stumps)

    # Write weights.bin
    # Magic 'ADAB' = 0x41444142, T, C
    out = struct.pack("<3i", 0x41444142, T, C)
    for c in range(C):
        for (pix, thresh, alpha, pol) in all_stumps[c]:
            out += struct.pack("<BBfb", pix, thresh, alpha, pol)
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
// infer.c
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
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1]<<8 | wb[2]<<16 | wb[3]<<24;
    if (magic != 0x41444142) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int T = wb[4] | wb[5]<<8 | wb[6]<<16 | wb[7]<<24;
    int C = wb[8] | wb[9]<<8 | wb[10]<<16 | wb[11]<<24;
    if (C != 10 || T <= 0) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 12 + (long)C * T * 7;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    // Parse stumps: per class, per stump: pixel (u8), thresh (u8), alpha (f32), polarity (i8)
    const unsigned char *p = wb + 12;
    // Store in flat arrays for fast access
    uint8_t *stump_pix = malloc(C * T);
    uint8_t *stump_thresh = malloc(C * T);
    float *stump_alpha = malloc(C * T * sizeof(float));
    int8_t *stump_pol = malloc(C * T);
    for (int c = 0; c < C; c++) {
        for (int t = 0; t < T; t++) {
            int idx = c * T + t;
            stump_pix[idx] = p[0];
            stump_thresh[idx] = p[1];
            memcpy(&stump_alpha[idx], p + 2, 4);
            stump_pol[idx] = (int8_t)p[6];
            p += 7;
        }
    }
    free(wb);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        float best_score = -1e30f;
        unsigned char best_class = 0;
        for (int c = 0; c < C; c++) {
            float score = 0.0f;
            for (int t = 0; t < T; t++) {
                int idx = c * T + t;
                uint8_t pix = stump_pix[idx];
                uint8_t thresh = stump_thresh[idx];
                float alpha = stump_alpha[idx];
                int8_t pol = stump_pol[idx];
                int pred = (xi[pix] <= thresh) ? 1 : -1;
                score += alpha * pol * pred;
            }
            if (score > best_score) {
                best_score = score;
                best_class = (unsigned char)c;
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    free(stump_pix); free(stump_thresh); free(stump_alpha); free(stump_pol);
    if (x) free(x);
    return 0;
}
```