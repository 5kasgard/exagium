```hypothesis
Family: label-local (linear + adjacent-pixel products).
Why: Digits have local stroke structure; pairwise products of neighboring z-scored pixels capture curvature and orientation missed by pure linear model, with only 112 extra features (56 horizontal + 56 vertical).
Failure: If dev accuracy ≤ label-linear (0.9613) or instructions/item > 2× label-linear (~10k Ir/item), the local interactions don't justify their cost.
```
```python
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
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]  # 64

    rng = np.random.default_rng(SEED)

    # Z-score each pixel (ordinal)
    X = X_bytes.astype(np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Z = (X - mean) / std  # (n, 64)

    # Pairwise features: horizontal and vertical adjacent products
    n = Z.shape[0]
    Z_pair = np.zeros((n, 112), dtype=np.float64)
    idx = 0
    # Horizontal (r, c) * (r, c+1)
    for r in range(8):
        for c in range(7):
            i1 = r * 8 + c
            i2 = r * 8 + c + 1
            Z_pair[:, idx] = Z[:, i1] * Z[:, i2]
            idx += 1
    # Vertical (r, c) * (r+1, c)
    for r in range(7):
        for c in range(8):
            i1 = r * 8 + c
            i2 = (r + 1) * 8 + c
            Z_pair[:, idx] = Z[:, i1] * Z[:, i2]
            idx += 1
    assert idx == 112

    # Combined features: 64 original + 112 pairwise = 176
    Z_all = np.concatenate([Z, Z_pair], axis=1).astype(np.float32)
    k = Z_all.shape[1]

    # Initialize weights (Xavier)
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Adam training
    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx_batch = order[s:s + BATCH]
            z, yy = Z_all[idx_batch], y[idx_batch]

            S = z @ W + b
            S = S - S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            G = (P / len(yy)).astype(np.float32)

            grad_W = (z.T @ G).astype(np.float32) + L2 * W
            grad_b = G.sum(axis=0).astype(np.float32)
            grads = [grad_W, grad_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Split weights for saving
    W_orig = W[:64]
    W_pair = W[64:]

    # Magic: 0x4C434F4C ("LOCL" little-endian)
    magic = 0x4C434F4C
    out = struct.pack("<4I", magic, d, C, 112)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W_orig.astype("<f4").tobytes()
    out += W_pair.astype("<f4").tobytes()
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
    uint32_t *hd = (uint32_t *)wb;
    if (nw < 16 || hd[0] != 0x4C434F4C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], n_pairs = hd[3];
    if (d != 64 || C != 10 || n_pairs != 112) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected = 16 + 1898L * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    float *ptr = (float *)(wb + 16);
    float *mean = ptr; ptr += 64;
    float *std = ptr; ptr += 64;
    float *W_orig = ptr; ptr += 64 * 10;
    float *W_pair = ptr; ptr += 112 * 10;
    float *bias = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float z[64];
    float scores[10];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < 64; j++) {
            z[j] = (xi[j] - mean[j]) / std[j];
        }

        for (int c = 0; c < C; c++) scores[c] = bias[c];

        for (int j = 0; j < 64; j++) {
            float zj = z[j];
            const float *w = W_orig + j * 10;
            for (int c = 0; c < 10; c++) scores[c] += zj * w[c];
        }

        int pair_idx = 0;
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 7; c++) {
                int i1 = r * 8 + c;
                int i2 = r * 8 + c + 1;
                float prod = z[i1] * z[i2];
                const float *w = W_pair + pair_idx * 10;
                for (int cl = 0; cl < 10; cl++) scores[cl] += prod * w[cl];
                pair_idx++;
            }
        }
        for (int r = 0; r < 7; r++) {
            for (int c = 0; c < 8; c++) {
                int i1 = r * 8 + c;
                int i2 = (r + 1) * 8 + c;
                float prod = z[i1] * z[i2];
                const float *w = W_pair + pair_idx * 10;
                for (int cl = 0; cl < 10; cl++) scores[cl] += prod * w[cl];
                pair_idx++;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```