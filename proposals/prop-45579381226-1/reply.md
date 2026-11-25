```hypothesis
Family: label-quadratic. Quadratic logistic regression on z-scored pixels (64 linear + 2080 pairwise features).
Captures stroke intersections and curves missed by linear model. Expect ~0.97 dev accuracy (beats label-linear 0.961)
at ~15k Ir/item (still low cost). Fails if quadratic terms overfit despite L2, showing no gain over linear.
```
```python
"""Quadratic logistic regression for fixed-size byte items (ordinal only).
python3 -s -P train.py card.json train.in train.key weights.bin
Features: 64 z-scored pixels + 2080 pairwise products (i<=j). Multinomial logistic regression with Adam.
Exports: magic 0x51554144, d, C, mu[d], sd[d], W[2144][C], b[C] (all float32 little-endian)."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-3, 64, 600000, 100
MAGIC = 0x51554144  # "QUAD"
FEAT_D = 64
QUAD_D = FEAT_D * (FEAT_D + 1) // 2  # 2080
TOTAL_D = FEAT_D + QUAD_D  # 2144


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    assert card["input"]["bytes"] == FEAT_D
    assert all(t == "ordinal" for t in card["input"]["types"])
    C = card["classes"]

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, FEAT_D)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # z-score per pixel
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Z = ((X.astype(np.float32) - mu) / sd)  # (n, 64)

    # build quadratic features (i <= j)
    quad = np.empty((n, QUAD_D), dtype=np.float32)
    idx = 0
    for i in range(FEAT_D):
        for j in range(i, FEAT_D):
            quad[:, idx] = Z[:, i] * Z[:, j]
            idx += 1
    Z_all = np.concatenate([Z, quad], axis=1).astype(np.float32)  # (n, 2144)

    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (TOTAL_D, C)) * math.sqrt(6.0 / TOTAL_D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_all[idx], y[idx]
            logits = z @ W + b
            g = softmax_grad(logits, yy)
            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(axis=0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # export
    out = struct.pack("<3i", MAGIC, FEAT_D, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Quadratic logistic regression: ./infer weights.bin in.rec N
 * Input: 64-byte records. Output: 1-byte class (argmax). */
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
    if (nw < 12) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x51554144) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (d != 64 || C < 1 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    const int QUAD_D = d * (d + 1) / 2;
    const int TOTAL_D = d + QUAD_D;
    long expected = 12L + 4L * d + 4L * d + 4L * TOTAL_D * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *mu = (const float *)(wb + 12);
    const float *sd = mu + d;
    const float *W = sd + d;
    const float *b = W + (long)TOTAL_D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * TOTAL_D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* linear features: normalized pixels */
        for (int j = 0; j < d; j++) {
            z[j] = (xi[j] - mu[j]) / sd[j];
        }
        /* quadratic features: pairwise products (i <= j) */
        int idx = d;
        for (int i1 = 0; i1 < d; i1++) {
            float zi = z[i1];
            for (int i2 = i1; i2 < d; i2++) {
                z[idx++] = zi * z[i2];
            }
        }

        /* scores = z @ W + b */
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < TOTAL_D; f++) {
            float zf = z[f];
            if (zf == 0.0f) continue;
            const float *wf = W + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += zf * wf[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```