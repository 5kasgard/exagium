```hypothesis
label-rff: Random Fourier Features (RBF kernel approximation) with deterministic projection, then logistic regression.
The 8x8 digit pixels have spatial structure that linear models miss; RFF captures nonlinearities with a fixed random first layer (cosine) and trained linear readout. 
Fixed seed ensures reproducibility. With D=32 hidden units, inference cost ~2x label-linear but accuracy should approach label-mlp16 (0.977).
Failure: dev accuracy <= 0.962 (no meaningful gain over linear).
```
```python
"""label-rff: Random Fourier Features for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Architecture: fixed random projection W1 (D x k) + bias b1, cosine nonlinearity, trained linear readout W2 (D x C) + b2.
Exported as per-byte lookup tables: magic 0x52464631, d, D, C; float32 T[d][256][D], b1[D], W2[D][C], b2[C]."""
import json, math, struct, sys

import numpy as np

D = 32          # number of random Fourier features
GAMMA = 0.05    # RBF kernel bandwidth parameter
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
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # Encode each byte feature independently (same as label-linear)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape

    # Fixed random projection for RFF: W1 ~ N(0, gamma*I), b1 ~ Uniform(0, 2pi)
    W1 = (rng.normal(0, math.sqrt(GAMMA), (D, k))).astype(np.float32)
    b1 = (rng.uniform(0, 2*math.pi, D)).astype(np.float32)

    # Compute hidden features: sqrt(2/D) * cos(Z @ W1.T + b1)
    H = (Z @ W1.T + b1).astype(np.float32)
    np.cos(H, out=H)
    H *= math.sqrt(2.0 / D)

    # Train linear readout (logistic regression with Adam) on H
    W2 = np.zeros((D, C), np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            h, yy = H[idx], y[idx]
            g = softmax_grad(h @ W2 + b2, yy)
            grads = [h.T @ g + L2 * W2, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Build per-byte lookup tables for first layer: T[j][v][u] = W1[u, j] * enc[j][v]
    # W1 is (D, k), enc[j] is (256, dim_j)
    T_list = []
    off = 0
    for j, E in enumerate(enc):
        dim_j = E.shape[1]
        # contribution of feature j to each RFF unit: E (256, dim_j) @ W1[:, off:off+dim_j].T (dim_j, D) -> (256, D)
        W1_j = W1[:, off:off+dim_j].T  # (dim_j, D)
        Tj = (E @ W1_j).astype(np.float32)  # (256, D)
        T_list.append(Tj)
        off += dim_j

    # Pack weights: magic, d, D, C; T[d][256][D]; b1[D]; W2[D][C]; b2[C]
    out = struct.pack("<4i", 0x52464631, X.shape[1], D, C)
    out += np.stack(T_list).astype("<f4").tobytes()  # (d, 256, D)
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-rff: Random Fourier Features classifier.
 * ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (class index).
 * Weights format: magic 0x52464631, d, D, C; float32 T[d][256][D], b1[D], W2[D][C], b2[C]. */
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
    if (nw < 16 || hd[0] != 0x52464631) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 4);          // [d][256][D]
    const float *b1 = T + (long)d * 256 * D;           // [D]
    const float *W2 = b1 + D;                          // [D][C]
    const float *b2 = W2 + (long)D * C;                // [C]

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // hidden[u] = b1[u] + sum_j T[j][xi[j]][u]
        for (int u = 0; u < D; u++) hidden[u] = b1[u];
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((long)j * 256 + xi[j]) * D;
            for (int u = 0; u < D; u++) hidden[u] += tj[u];
        }
        // cosine nonlinearity
        for (int u = 0; u < D; u++) hidden[u] = cosf(hidden[u]);

        // scores[c] = b2[c] + sum_u hidden[u] * W2[u][c]
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int u = 0; u < D; u++) {
            float hu = hidden[u];
            const float *w2u = W2 + (long)u * C;
            for (int c = 0; c < C; c++) scores[c] += hu * w2u[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```