```hypothesis
Family: label-elm (Extreme Learning Machine). Fixed random hidden layer (64 ReLU units) with ridge-regressed output weights. Should exploit nonlinear pixel interactions at inference cost ~4x label-linear. Fails if random features don't capture structure or ridge lambda is misspecified.
```

```python
import json, struct, sys
import numpy as np

SEED = 1
HIDDEN = 64
LAMBDA = 1e-3

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = b"".join(records(sys.argv[2]))
    d = card["input"]["bytes"]
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]

    # Scale pixel counts 0..16 to [0,1]
    X_float = X.astype(np.float32) / 16.0

    rng = np.random.default_rng(SEED)
    # He initialization for ReLU
    W1 = rng.normal(0, np.sqrt(2.0 / d), size=(d, HIDDEN)).astype(np.float32)
    b1 = np.zeros(HIDDEN, dtype=np.float32)

    # Hidden activations
    H = np.maximum(X_float @ W1 + b1, 0)  # (n, HIDDEN)

    # Add bias column
    H_aug = np.hstack([H, np.ones((n, 1), dtype=np.float32)])  # (n, HIDDEN+1)

    # One-hot labels
    Y = np.zeros((n, C), dtype=np.float32)
    Y[np.arange(n), y] = 1.0

    # Ridge regression: (H_aug^T H_aug + lambda I) W = H_aug^T Y
    A = H_aug.T @ H_aug
    A[np.diag_indices_from(A)] += LAMBDA
    B = H_aug.T @ Y
    W_aug = np.linalg.solve(A, B).astype(np.float32)
    W2 = W_aug[:HIDDEN, :]
    b2 = W_aug[HIDDEN, :]

    # Pack weights: magic, d, H, C; then W1, b1, W2, b2 (all float32 LE)
    magic = 0x454C4D31  # "ELM1"
    header = struct.pack("<4i", magic, d, HIDDEN, C)
    w1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    w2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()
    out = header + w1_bytes + b1_bytes + w2_bytes + b2_bytes
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x454C4D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 16L + 4L * ((long)d * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W1 = (const float *)(hd + 4);
    const float *b1 = W1 + (long)d * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int j = 0; j < d; j++) {
                sum += (xi[j] / 16.0f) * W1[(long)j * H + h];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) {
                sum += hidden[h] * W2[(long)h * C + c];
            }
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```