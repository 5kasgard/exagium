```hypothesis
Family: label-fm (Factorization Machine). Adds pairwise column interactions via low-rank embeddings (R=8) to a linear model, capturing feature crosses crucial for tabular data like Adult. The model remains lightweight: O(d*R) per item (d=14, R=8) with ~140KB weights. Expected to beat label-linear (0.8524) and approach label-mlp16 (0.8591) at lower CPU cost. Failure if dev accuracy <= 0.8524.
```

```python
"""Factorization Machine for tabular label tasks: ./train.py card.json train.in train.key weights.bin
Adds pairwise column interactions via low-rank embeddings to a linear per-value model.
Exported as float32 lookup tables: magic 0x464D4C31, d, R, C; T_lin[d][256][C], V[d][256][R], b[C], u[C]."""
import json, math, struct, sys

import numpy as np

R = 8
SEED, LR, L2, BATCH, EPOCHS = 1, 0.01, 1e-4, 64, 50


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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Parameters
    T_lin = np.zeros((d, 256, C), dtype=np.float32)
    V = rng.normal(0, 0.01, size=(d, 256, R)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    u = np.zeros(C, dtype=np.float32)

    params = [T_lin, V, b, u]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(idx)

            # Forward
            linear_acc = np.tile(b, (B, 1))
            for j in range(d):
                linear_acc += T_lin[j, xb[:, j], :]

            S = np.zeros((B, R), dtype=np.float32)
            sq_sum = np.zeros(B, dtype=np.float32)
            for j in range(d):
                Vj = V[j, xb[:, j], :]
                S += Vj
                sq_sum += np.sum(Vj * Vj, axis=1)

            interaction = 0.5 * (np.sum(S * S, axis=1) - sq_sum)
            logits = linear_acc + interaction[:, None] * u[None, :]

            # Softmax cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            # Backward
            grad_u = np.dot(grad_logits.T, interaction)
            grad_b = grad_logits.sum(axis=0)

            grad_T_lin = np.zeros_like(T_lin)
            for j in range(d):
                v = xb[:, j]
                for c in range(C):
                    np.add.at(grad_T_lin[j, :, c], v, grad_logits[:, c])

            grad_inter = np.dot(grad_logits, u)
            grad_V = np.zeros_like(V)
            for j in range(d):
                v = xb[:, j]
                Vj = V[j, v, :]
                factor = grad_inter[:, None] * (S - Vj)
                for r in range(R):
                    np.add.at(grad_V[j, :, r], v, factor[:, r])

            # L2 regularization
            grad_T_lin += L2 * T_lin
            grad_V += L2 * V
            grad_u += L2 * u

            grads = [grad_T_lin, grad_V, grad_b, grad_u]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    out = struct.pack("<4i", 0x464D4C31, d, R, C)
    out += T_lin.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += u.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class. */
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
    uint32_t k = (uint32_t)buf[*pos] | (uint32_t)buf[*pos+1]<<8 | (uint32_t)buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, (k>>8)&255, (k>>16)&255, (k>>24)&255};
    fwrite(h, 1, 4, f);
    fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], R = hd[2], C = hd[3];
    if (C < 1 || C > 256 || R < 1 || d < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + (long)d * 256 * R + C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T_lin = (const float *)(hd + 4);
    const float *V = T_lin + (long)d * 256 * C;
    const float *b = V + (long)d * 256 * R;
    const float *u = b + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *linear_acc = malloc(sizeof(float) * C);
    float *S = malloc(sizeof(float) * R);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) linear_acc[c] = b[c];
        for (int r = 0; r < R; r++) S[r] = 0.0f;
        float sq_sum = 0.0f;
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *tlin = T_lin + ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) linear_acc[c] += tlin[c];
            const float *vemb = V + ((long)j * 256 + v) * R;
            for (int r = 0; r < R; r++) {
                float val = vemb[r];
                S[r] += val;
                sq_sum += val * val;
            }
        }
        float dotSS = 0.0f;
        for (int r = 0; r < R; r++) dotSS += S[r] * S[r];
        float interaction = 0.5f * (dotSS - sq_sum);
        for (int c = 0; c < C; c++) scores[c] = linear_acc[c] + u[c] * interaction;
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```