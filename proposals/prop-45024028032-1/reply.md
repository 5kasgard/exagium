```hypothesis
Family: label-poly2 (degree-2 polynomial logistic regression).
Why: Optdigits pixels exhibit local correlations (strokes); pairwise products capture curvature and intersections that linear models miss.
This is a classic strong baseline for digit recognition, expected to surpass label-linear (96.1%) and approach label-mlp16 (97.7%) with moderate CPU cost (~23k FLOPs/item vs 640 for linear).
Failure mode: If dev accuracy ≤ 96.1%, interactions are insufficient or overfitting dominates despite L2.
```
```python
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-3, 64, 600000, 100


def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    for t in card["input"]["types"]:
        if t != "ordinal":
            raise ValueError("label-poly2 only supports ordinal features")
    rng = np.random.default_rng(SEED)

    # Z-scoring per feature
    mean = X.mean(axis=0).astype(np.float64)
    std = X.std(axis=0).astype(np.float64)
    std[std == 0] = 1.0
    z_table = ((np.arange(256, dtype=np.float64) - mean[:, None]) / std[:, None]).astype(np.float32)  # (d, 256)

    Z = ((X - mean) / std).astype(np.float32)  # (n, d)
    n = Z.shape[0]
    K = d * (d + 1) // 2

    # Build polynomial features Phi (n, K)
    Phi = np.zeros((n, K), dtype=np.float32)
    col = 0
    for i in range(d):
        for j in range(i, d):
            Phi[:, col] = Z[:, i] * Z[:, j]
            col += 1

    # Initialize weights
    W = (rng.uniform(-1, 1, (K, C)) * math.sqrt(6.0 / K)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            phi = Phi[idx]  # (B, K)
            yy = y[idx]
            # Forward
            logits = phi @ W + b  # (B, C)
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            # Gradient
            probs[np.arange(len(yy)), yy] -= 1.0
            dlogits = (probs / len(yy)).astype(np.float32)
            grad_W = phi.T @ dlogits + L2 * W
            grad_b = dlogits.sum(axis=0)
            grads = [grad_W, grad_b]
            # Adam
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Magic: 0x504F4C32 ('POL2')
    # Header: magic, d, C, K (int32 little-endian)
    # Then z_table (d*256 float32), W (K*C float32), b (C float32)
    out = struct.pack("<4i", 0x504F4C32, d, C, K)
    out += z_table.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
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
    uint32_t k = (uint32_t)buf[*pos] | (uint32_t)buf[*pos+1] << 8 | (uint32_t)buf[*pos+2] << 16 | (uint32_t)buf[*pos+3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
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
    if (nw < 16 || hd[0] != 0x504F4C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2], K = hd[3];
    if (C < 1 || C > 256 || d < 1 || K != d * (d + 1) / 2) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 + (long)K * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *z_table = (const float *)(hd + 4);
    const float *W = z_table + (long)d * 256;
    const float *b = W + (long)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *f = (float *)malloc(sizeof(float) * d);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            f[j] = z_table[(long)j * 256 + xi[j]];
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        long w_idx = 0;
        for (int i1 = 0; i1 < d; i1++) {
            float fi = f[i1];
            for (int i2 = i1; i2 < d; i2++) {
                float prod = fi * f[i2];
                for (int c = 0; c < C; c++) {
                    scores[c] += prod * W[w_idx * C + c];
                }
                w_idx++;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```