```hypothesis
label-factorized: a low-rank factorized linear model (rank 32) that decomposes the 16×256×65 weight tensor into U (16×256×32) and V (32×65). This reduces parameters from 266K to 67K, providing strong regularization for the 24K training examples. It should beat label-linear (0.3375) by preventing overfitting while keeping inference fast (~15k Ir/item). Failure: dev accuracy ≤ 0.3375 or inference > 20k Ir/item.
```

```python
"""Factorized linear model for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Factorizes the per-position per-byte per-class weight tensor W[d,256,C] into
U[d,256,R] and V[R,C] (R=32). Trains with Adam (lr=1e-2, L2=1e-4) on the
full one-hot encoding. Deterministic (seed=1), single thread.
Exports: int32 magic 0x46414354, d, R, C; float32 U[d*256*R], V[R*C], bias[C]."""
import json, math, struct, sys

import numpy as np

R = 32
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


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
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Initialize parameters
    # U: (d, 256, R), V: (R, C), bias: (C,)
    scale_U = math.sqrt(6.0 / (256 + R))
    scale_V = math.sqrt(6.0 / (R + C))
    U = (rng.uniform(-1, 1, (d, 256, R)) * scale_U).astype(np.float32)
    V = (rng.uniform(-1, 1, (R, C)) * scale_V).astype(np.float32)
    bias = np.zeros(C, np.float32)

    # Adam state
    mU, vU = np.zeros_like(U), np.zeros_like(U)
    mV, vV = np.zeros_like(V), np.zeros_like(V)
    mb, vb = np.zeros_like(bias), np.zeros_like(bias)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward: h = sum_j U[j, xb[:,j], :]  -> (B, R)
            h = np.zeros((B, R), np.float32)
            for j in range(d):
                h += U[j, xb[:, j], :]

            logits = h @ V + bias   # (B, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dlogits = (probs / B).astype(np.float32)   # (B, C)

            # Gradients
            dV = h.T @ dlogits + L2 * V                 # (R, C)
            dbias = dlogits.sum(axis=0)                 # (C,)
            dh = dlogits @ V.T                          # (B, R)

            dU = np.zeros_like(U)
            for j in range(d):
                np.add.at(dU[j], xb[:, j], dh)

            dU += L2 * U

            # Adam update
            t += 1
            beta1_t = BETA1 ** t
            beta2_t = BETA2 ** t
            lr_t = LR * math.sqrt(1 - beta2_t) / (1 - beta1_t)

            for param, grad, m, v in [
                (U, dU, mU, vU),
                (V, dV, mV, vV),
                (bias, dbias, mb, vb)
            ]:
                m *= BETA1; m += (1 - BETA1) * grad
                v *= BETA2; v += (1 - BETA2) * grad * grad
                param -= lr_t * m / (np.sqrt(v) + EPS)

    # Pack weights: magic, d, R, C; then U, V, bias as float32 LE
    out = struct.pack("<4i", 0x46414354, d, R, C)
    out += U.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorized linear model inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte (class index).
 * Weights format: int32 magic=0x46414354, d, R, C; float32 U[d*256*R], V[R*C], bias[C]. */
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
    if (nw < 16 || hd[0] != 0x46414354) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], R = hd[2], C = hd[3];
    if (d != 16 || C != 65 || R <= 0 || R > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * R + (long)R * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *U = (const float *)(hd + 4);
    const float *V = U + (long)d * 256 * R;
    const float *bias = V + (long)R * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = (float *)malloc(sizeof(float) * R);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int r = 0; r < R; r++) acc[r] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *u = U + ((long)j * 256 + xi[j]) * R;
            for (int r = 0; r < R; r++) acc[r] += u[r];
        }

        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int r = 0; r < R; r++) {
            float ar = acc[r];
            const float *vr = V + (long)r * C;
            for (int c = 0; c < C; c++) scores[c] += ar * vr[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(acc); free(scores);
    return 0;
}
```