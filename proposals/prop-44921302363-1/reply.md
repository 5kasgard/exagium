```hypothesis
Family: label-onehotlinear (per-pixel lookup table logistic regression).
Why: Optdigits pixels are ordinal 0..16 but class relationships may be non-monotonic; one-hot encoding each pixel value allows arbitrary per-pixel contributions while keeping inference as 64 additions per class (same cost as label-linear). Training is convex multinomial logistic regression with L2, deterministic and fast.
Failure: If dev accuracy <= 0.9613 (label-linear) or inference cost exceeds label-linear by >2x.
```
```python
"""label-onehotlinear: multinomial logistic regression on one-hot encoded pixel values (0..16).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy.
Weights format: magic 0x4F4E4548, d=64, V=17, C=10; float32 bias[C], weight[d][V][C] (row-major)."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.1, 1e-4, 64, 600000, 100
D, V, C = 64, 17, 10  # fixed for optdigits

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    # card["input"]["bytes"] == 64, types all ordinal
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, D)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]

    # One-hot encode: each pixel 0..16 -> 17 features
    Z = np.zeros((n, D * V), dtype=np.float32)
    for j in range(D):
        vals = X_bytes[:, j]
        Z[np.arange(n), j * V + vals] = 1.0

    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (D * V, C)) * math.sqrt(6.0 / (D * V))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # softmax gradient
            scores = z @ W + b
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = probs / len(yy)  # (batch, C)

            grad_W = z.T @ grad + L2 * W
            grad_b = grad.sum(axis=0)

            t += 1
            for i, (p, gr) in enumerate(zip(params, [grad_W, grad_b])):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    magic = 0x4F4E4548  # "ONEH" in little-endian
    out = struct.pack("<4i", magic, D, V, C)
    out += b.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-onehotlinear inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixel values 0..16). Output: one byte class (0..9). */
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4F4E4548) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], V = hd[2], C = hd[3];
    if (d != 64 || V != 17 || C != 10) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected = 16L + 4L * C + 4L * d * V * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *weight = bias + C;  // [d][V][C]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            if (v >= V) { fprintf(stderr, "pixel value %d >= %d\n", v, V); return 3; }
            const float *w = weight + ((long)j * V + v) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```