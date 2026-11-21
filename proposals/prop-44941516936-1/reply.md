```hypothesis
Family: label-poly2 (degree-2 polynomial features + linear multinomial logistic regression).
Why: The 8×8 optdigits pixels are ordinal (0..16). A linear model on raw pixels ignores pixel interactions.
Adding all pairwise products (including squares) captures simple non-linearities (strokes, curves) at low cost:
2144 features vs 64, still tiny. Training with the same Adam/L2 recipe as label-linear should beat 0.9613.
Failure mode: If dev accuracy ≤ 0.9613 or inference cost exceeds label-gbdt (~4500 ns/item), the family is not worthwhile.
```

```python
"""label-poly2: degree-2 polynomial features on ordinal inputs, then linear multinomial logistic regression.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weight format: magic 0x504F4C32, d, C, mean[d], std[d], W[feat*C], b[C] (all float32 little-endian).
feat = d + d*(d+1)/2 = 64 + 2080 = 2144 for optdigits."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x504F4C32  # "POL2"

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def poly_features(Z):
    """Z: (n, d) float32 normalized ordinal features. Return (n, feat) with linear + quadratic terms."""
    n, d = Z.shape
    feat = d + d * (d + 1) // 2
    Phi = np.empty((n, feat), dtype=np.float32)
    Phi[:, :d] = Z
    col = d
    for i in range(d):
        # j from i to d-1
        Phi[:, col:col + d - i] = Z[:, i:i+1] * Z[:, i:]
        col += d - i
    return Phi

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    # all types are ordinal per card
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)
    Phi = poly_features(Z)
    n, k = Phi.shape
    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            phi, yy = Phi[idx], y[idx]
            # softmax gradient
            S = phi @ W + b
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grad_W = phi.T @ g + L2 * W
            grad_b = g.sum(axis=0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    # pack weights
    out = struct.pack("<3i", MAGIC, d, C)
    out += mu.astype("<f4").tobytes() + sd.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-poly2 inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (uint8 pixels). Output record: 1 byte class (0..9).
 * Weight file: magic 0x504F4C32, int32 d, int32 C, float32 mean[d], float32 std[d],
 *              float32 W[feat*C], float32 b[C] where feat = d + d*(d+1)/2.
 * All little-endian. Single-threaded, no external deps. */
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
    int32_t *hdr = (int32_t *)wb;
    if (nw < 12 || hdr[0] != 0x504F4C32) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hdr[1], C = hdr[2];
    if (d != 64 || C != 10) { fprintf(stderr, "unexpected d=%d C=%d\n", d, C); return 3; }
    int feat = d + d * (d + 1) / 2;           // 2144
    long expected = 12L + 4L * (2L * d + (long)feat * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hdr + 3);
    const float *std  = mean + d;
    const float *W    = std + d;
    const float *b    = W + (long)feat * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = (float *)malloc(d * sizeof(float));
    float *phi = (float *)malloc(feat * sizeof(float));
    float *score = (float *)malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        /* normalize */
        for (int j = 0; j < d; j++) z[j] = (xi[j] - mean[j]) / std[j];
        /* polynomial features: linear then quadratic (i<=j) */
        int idx = 0;
        for (int j = 0; j < d; j++) phi[idx++] = z[j];
        for (int ii = 0; ii < d; ii++) {
            for (int jj = ii; jj < d; jj++) phi[idx++] = z[ii] * z[jj];
        }
        /* scores = b + phi^T W */
        for (int c = 0; c < C; c++) score[c] = b[c];
        for (int f = 0; f < feat; f++) {
            float ph = phi[f];
            const float *wf = W + (long)f * C;
            for (int c = 0; c < C; c++) score[c] += ph * wf[c];
        }
        /* argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(z); free(phi); free(score);
    if (x) free(x);
    free(wb);
    return 0;
}
```