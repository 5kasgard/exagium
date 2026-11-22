```hypothesis
Family: label-multiscale-linear. A linear classifier on raw pixels plus multi-scale average pooling features (2x2, 4x4, 8x8). Captures local and global spatial structure at minimal compute cost (~850 FLOPs/item vs label-linear's 640). Expects to exceed label-linear's 96.1% dev accuracy; if dev accuracy ≤96.3%, the spatial pooling adds insufficient signal.
```
```python
"""label-multiscale-linear: Linear model on raw 8x8 pixels + multi-scale pooling features.
Features: 64 raw + 16 (2x2 avg) + 4 (4x4 avg) + 1 (8x8 avg) = 85 total.
Z-scored per feature, trained with Adam (same hyperparams as label-linear).
Weights format: magic, d_raw, d_pool, C; float32 mu_raw, sd_raw, mu_pool, sd_pool, W_raw, W_pool, bias."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x4D534C31  # "MSL1"
D_RAW, D_POOL, C = 64, 21, 10  # 16+4+1=21 pooled


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def compute_pool(x):  # x: (n, 64) uint8 -> (n, 21) float32
    n = x.shape[0]
    pool = np.zeros((n, 21), dtype=np.float32)
    img = x.reshape(n, 8, 8).astype(np.float32)
    # 2x2 avg pool -> 4x4 = 16
    pool[:, :16] = img.reshape(n, 4, 2, 4, 2).mean(axis=(2, 4)).reshape(n, 16)
    # 4x4 avg pool -> 2x2 = 4
    pool[:, 16:20] = img.reshape(n, 2, 4, 2, 4).mean(axis=(2, 4)).reshape(n, 4)
    # 8x8 avg pool -> 1
    pool[:, 20] = img.mean(axis=(1, 2))
    return pool


def main():
    card = json.load(open(sys.argv[1]))
    X_raw = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_raw.shape[0]
    assert card["classes"] == C and card["input"]["bytes"] == D_RAW

    X_pool = compute_pool(X_raw)
    X_all = np.concatenate([X_raw.astype(np.float32), X_pool], axis=1)  # (n, 85)

    # Z-score per feature
    mu = X_all.mean(axis=0).astype(np.float32)
    sd = X_all.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Z = ((X_all - mu) / sd).astype(np.float32)

    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (85, C)) * math.sqrt(6.0 / 85)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits, dtype=np.float32)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W + b
            g = softmax_grad(logits, yy)
            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(axis=0)
            t += 1
            for i, (p, gr) in enumerate(zip(params, [grad_W, grad_b])):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Pack weights: magic, d_raw, d_pool, C; mu_raw, sd_raw, mu_pool, sd_pool; W_raw, W_pool; bias
    W_raw, W_pool = W[:D_RAW], W[D_RAW:]
    mu_raw, mu_pool = mu[:D_RAW], mu[D_RAW:]
    sd_raw, sd_pool = sd[:D_RAW], sd[D_RAW:]

    out = struct.pack("<4i", MAGIC, D_RAW, D_POOL, C)
    out += mu_raw.astype("<f4").tobytes() + sd_raw.astype("<f4").tobytes()
    out += mu_pool.astype("<f4").tobytes() + sd_pool.astype("<f4").tobytes()
    out += W_raw.astype("<f4").tobytes() + W_pool.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-multiscale-linear inference: ./infer weights.bin in.rec N
 * Input: 64-byte records (8x8 pixels 0..16). Output: 1-byte class 0..9.
 * Weights: magic(MSL1), d_raw=64, d_pool=21, C=10; float32 arrays:
 *   mu_raw[64], sd_raw[64], mu_pool[21], sd_pool[21],
 *   W_raw[64*10], W_pool[21*10], bias[10] */
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
    if (nw < 16 || hd[0] != 0x4D534C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d_raw = hd[1], d_pool = hd[2], C = hd[3];
    if (d_raw != 64 || d_pool != 21 || C != 10) { fprintf(stderr, "bad dims\n"); return 3; }
    size_t expected = 16 + 4L * (64*2 + 21*2 + 64*10 + 21*10 + 10);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *mu_raw = p; p += 64;
    const float *sd_raw = p; p += 64;
    const float *mu_pool = p; p += 21;
    const float *sd_pool = p; p += 21;
    const float *W_raw = p; p += 64*10;
    const float *W_pool = p; p += 21*10;
    const float *bias = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    float pool[21];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }

        // Initialize with bias
        for (int c = 0; c < C; c++) acc[c] = bias[c];

        // Raw features: 64 pixels
        for (int j = 0; j < 64; j++) {
            float z = (xi[j] - mu_raw[j]) / sd_raw[j];
            const float *w = W_raw + j * 10;
            for (int c = 0; c < 10; c++) acc[c] += z * w[c];
        }

        // Pooled features: 2x2 (16), 4x4 (4), 8x8 (1)
        // 2x2 avg pool -> 4x4 = 16
        int idx = 0;
        for (int r = 0; r < 8; r += 2) {
            for (int c = 0; c < 8; c += 2) {
                float sum = xi[r*8 + c] + xi[r*8 + c+1] + xi[(r+1)*8 + c] + xi[(r+1)*8 + c+1];
                pool[idx++] = sum * 0.25f;
            }
        }
        // 4x4 avg pool -> 2x2 = 4
        for (int r = 0; r < 8; r += 4) {
            for (int c = 0; c < 8; c += 4) {
                float sum = 0;
                for (int dr = 0; dr < 4; dr++)
                    for (int dc = 0; dc < 4; dc++)
                        sum += xi[(r+dr)*8 + c+dc];
                pool[idx++] = sum * 0.0625f; // 1/16
            }
        }
        // 8x8 avg pool -> 1
        float sum = 0;
        for (int j = 0; j < 64; j++) sum += xi[j];
        pool[idx++] = sum * (1.0f/64.0f);

        // Add pooled contributions
        for (int j = 0; j < 21; j++) {
            float z = (pool[j] - mu_pool[j]) / sd_pool[j];
            const float *w = W_pool + j * 10;
            for (int c = 0; c < 10; c++) acc[c] += z * w[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < 10; c++) if (acc[c] > acc[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```