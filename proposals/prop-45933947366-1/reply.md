```hypothesis
Family: label-gam (generalized additive model with per-pixel lookup tables).
Why: Optdigits pixels have non-monotonic class influence (e.g., high ink in a region may indicate one digit, medium another). A GAM learns an arbitrary 256→ℝ mapping per pixel per class, capturing non-linearities while keeping inference at one table lookup per pixel (same cost as label-linear). It has 64×256×10 ≈ 164k parameters, still tiny.
Failure signal: If dev accuracy ≤ 0.961 (label-linear), the non-linearity doesn't help or overfits despite L2.
```
```python
"""Generalized Additive Model for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Learns per-pixel per-class contribution tables T[d,256,C] and bias b[C].
Exports in label-linear format (magic 0x54424C31, d, H=0, C, then T, then b)."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # Parameters: T[d, 256, C], b[C]
    T = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    mT, vT = np.zeros_like(T), np.zeros_like(T)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # logits = b + sum_j T[j, xb[:,j], :]
            logits = b.copy()
            for j in range(d):
                logits += T[j, xb[:, j], :]

            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs  # dL/dlogits

            # gradients
            gT = np.zeros_like(T)
            gb = probs.sum(axis=0).astype(np.float32)
            for j in range(d):
                # accumulate probs into gT[j, xb[:,j], :]
                np.add.at(gT[j], xb[:, j], probs)

            # L2 regularization
            gT += L2 * T
            gb += L2 * b

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            for param, grad, m, v in [(T, gT, mT, vT), (b, gb, mb, vb)]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Write weights.bin in label-linear format (H=0, W=C)
    magic, H = 0x54424C31, 0
    out = struct.pack("<4i", magic, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* GAM inference (identical to label-linear H=0 path). */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || H != 0 || nw != 16 + 4L * ((long)d * 256 * W + W)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b = T + (long)d * 256 * W;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(sc, b, sizeof(float) * C);
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) sc[c] += t[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```