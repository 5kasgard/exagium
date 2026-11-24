```hypothesis
Family: label-elm (Extreme Learning Machine). Fixed random projection (64→32 ReLU) followed by ridge-regression output layer. Why: random features give non-linear separation; closed-form training is deterministic, fast, and often matches backprop on small ordinal data like optdigits. Failure: dev accuracy ≤ 0.9613 (label-linear) or Ir/item ≥ 5058.
```
```python
"""label-elm: Extreme Learning Machine with fixed random first layer and ridge-regression readout.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 42), single thread, stdlib+numpy only.
Encodes each byte as z-score (ordinal). Random W1 (64x32), b1=0. Hidden = ReLU(X @ W1).
Ridge regression (λ=1e-3) for W2 (32x10), b2. Exports: magic, d, H, C, means, stds, W1, b1, W2, b2."""
import json, struct, sys

import numpy as np

SEED, H, LAMBDA = 42, 32, 1e-3
MAGIC = 0x454C4D31  # "ELM1"


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

    # z-score per feature (all ordinal)
    means = X_bytes.mean(axis=0).astype(np.float32)
    stds = X_bytes.std(axis=0).astype(np.float32)
    stds[stds == 0] = 1.0
    Z = ((X_bytes.astype(np.float32) - means) / stds)

    rng = np.random.default_rng(SEED)
    W1 = rng.normal(0, 1/np.sqrt(d), (d, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)

    H_act = np.maximum(Z @ W1 + b1, 0, dtype=np.float32)  # ReLU

    # Ridge regression: (H^T H + λI) W2 = H^T Y
    Y = np.eye(C, dtype=np.float32)[y]
    A = H_act.T @ H_act + LAMBDA * np.eye(H, dtype=np.float32)
    B = H_act.T @ Y
    W2 = np.linalg.solve(A, B).astype(np.float32)
    b2 = (Y.mean(axis=0) - H_act.mean(axis=0) @ W2).astype(np.float32)

    # Pack weights
    out = struct.pack("<4i", MAGIC, d, H, C)
    out += means.tobytes() + stds.tobytes()
    out += W1.tobytes() + b1.tobytes()
    out += W2.tobytes() + b2.tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-elm inference: ./infer weights.bin in.rec N
 * Input: d bytes (z-scored per feature). Hidden: ReLU(x*W1 + b1). Output: argmax(h*W2 + b2). */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x454C4D31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (d != 64 || H != 32 || C != 10) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16L + 8L*d + 4L*H + 4L*d*H + 4L*H*C + 4L*C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *means = (const float *)(hd + 4);
    const float *stds  = means + d;
    const float *W1    = stds + d;
    const float *b1    = W1 + (long)d * H;
    const float *W2    = b1 + H;
    const float *b2    = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *h = malloc(sizeof(float) * H);
    float *s = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) z[j] = (xi[j] - means[j]) / stds[j];
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int j = 0; j < d; j++) sum += z[j] * W1[(long)j * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
            s[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```