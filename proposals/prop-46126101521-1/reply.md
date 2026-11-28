```hypothesis
Family: label-fm (Factorization Machine with rank-4 pairwise interactions).
Why: Optdigits 8x8 pixels have strong local correlations (strokes); FM captures pairwise interactions at O(k*d) cost (k=4) instead of O(d^2). With ~13k params and ~3k ops/item, it should exceed linear's 0.9613 accuracy at similar or lower CPU cost.
Failure: If dev accuracy ≤ 0.96, the low-rank approximation is insufficient or optimization failed to beat the linear baseline.
```

```python
"""Factorization Machine (rank-k) for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Exports: magic, d, k, C; float32 mu[d], inv_sd[d], bias[C], W[C][d], V[C][k][d]."""
import json, math, struct, sys

import numpy as np

K = 4
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_grad(scores, y):
    n = scores.shape[0]
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return probs / n

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X_bytes.shape[1]
    n = X_bytes.shape[0]

    X = X_bytes.astype(np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    inv_sd = 1.0 / sd
    Z = (X - mu) * inv_sd
    Z2 = Z * Z

    rng = np.random.default_rng(SEED)
    scale = 0.01
    W = (rng.normal(0, scale, (C, d))).astype(np.float32)
    V = (rng.normal(0, scale, (C, K, d))).astype(np.float32)
    bias = np.zeros(C, np.float32)

    m_W = np.zeros_like(W); v_W = np.zeros_like(W)
    m_V = np.zeros_like(V); v_V = np.zeros_like(V)
    m_b = np.zeros_like(bias); v_b = np.zeros_like(bias)
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx].astype(np.float32)
            z2 = Z2[idx].astype(np.float32)
            yy = y[idx]
            b = len(yy)

            scores = np.zeros((b, C), np.float32)
            S = np.zeros((b, C, K), np.float32)

            for c in range(C):
                scores[:, c] = bias[c] + z @ W[c]
                for f in range(K):
                    s_cf = z @ V[c, f]
                    S[:, c, f] = s_cf
                    scores[:, c] += 0.5 * s_cf * s_cf
                    scores[:, c] -= 0.5 * z2 @ (V[c, f] * V[c, f])

            grad = softmax_grad(scores, yy)
            t += 1

            g_b = grad.sum(axis=0)
            m_b = 0.9 * m_b + 0.1 * g_b
            v_b = 0.999 * v_b + 0.001 * g_b * g_b
            m_b_hat = m_b / (1 - 0.9 ** t)
            v_b_hat = v_b / (1 - 0.999 ** t)
            bias -= LR * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)

            g_W = (z.T @ grad).T + L2 * W
            m_W = 0.9 * m_W + 0.1 * g_W
            v_W = 0.999 * v_W + 0.001 * g_W * g_W
            m_W_hat = m_W / (1 - 0.9 ** t)
            v_W_hat = v_W / (1 - 0.999 ** t)
            W -= LR * m_W_hat / (np.sqrt(v_W_hat) + 1e-8)

            for c in range(C):
                g_c = grad[:, c]
                z2T_g = z2.T @ g_c
                for f in range(K):
                    S_cf = S[:, c, f]
                    term1 = z.T @ (g_c * S_cf)
                    term2 = V[c, f] * z2T_g
                    g_V = term1 - term2 + L2 * V[c, f]
                    m_V[c, f] = 0.9 * m_V[c, f] + 0.1 * g_V
                    v_V[c, f] = 0.999 * v_V[c, f] + 0.001 * g_V * g_V
                    m_V_hat = m_V[c, f] / (1 - 0.9 ** t)
                    v_V_hat = v_V[c, f] / (1 - 0.999 ** t)
                    V[c, f] -= LR * m_V_hat / (np.sqrt(v_V_hat) + 1e-8)

    magic = 0x464D0001
    out = struct.pack("<Iiii", magic, d, K, C)
    out += mu.astype("<f4").tobytes()
    out += inv_sd.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine (rank-k) for label tasks: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class index. */
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = (int)hd[1], K = (int)hd[2], C = (int)hd[3];
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 4L * (2L * d + C + (long)C * d + (long)C * K * d);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *mu = p; p += d;
    const float *inv_sd = p; p += d;
    const float *bias = p; p += C;
    const float *W = p; p += (long)C * d;
    const float *V = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = (float *)malloc(d * sizeof(float));
    float *scores = (float *)malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k