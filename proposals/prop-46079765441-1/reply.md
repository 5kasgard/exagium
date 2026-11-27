```hypothesis
label-maxlin: linear classifier on 2×2 max-pooled features (16 features from 8×8 input). Max pooling preserves stroke presence while suppressing noise; 4× fewer parameters (160 vs 640 weights) and ~3× faster inference (16 vs 64 table lookups). Fails if max pooling discards critical stroke-thickness information that the linear model needs.
```
```python
import json, math, struct, sys

import numpy as np

H = 0
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def maxpool2x2(X):
    # X: (n, 64) uint8, row-major 8x8 -> (n, 16) uint8 max over 2x2 blocks
    n = X.shape[0]
    X8 = X.reshape(n, 8, 8)
    # blocks at rows 0,2,4,6 and cols 0,2,4,6
    blocks = []
    for r in (0, 2, 4, 6):
        for c in (0, 2, 4, 6):
            block = X8[:, r:r+2, c:c+2]
            blocks.append(block.max(axis=(1, 2)))
    return np.stack(blocks, axis=1).astype(np.uint8)


def main():
    card = json.load(open(sys.argv[1]))
    X_raw = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]

    # Max-pool 8x8 -> 4x4 = 16 features
    X = maxpool2x2(X_raw)
    d = X.shape[1]  # 16

    rng = np.random.default_rng(SEED)
    enc = []
    for j in range(d):
        v = X[:, j].astype(np.float64)
        mu, sd = v.mean(), v.std() or 1.0
        enc.append(((np.arange(256) - mu) / sd)[:, None])

    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape
    W1 = np.zeros((k, C), np.float32)
    b1 = np.zeros(C, np.float32)
    params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            S = z @ W1 + b1
            S = S - S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    off, T = 0, []
    for E in enc:
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    out = struct.pack("<4i", 0x54424C31, d, H, C) + np.stack(T).astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
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

static inline unsigned char max4(unsigned char a, unsigned char b, unsigned char c, unsigned char d) {
    unsigned char m = a > b ? a : b;
    m = m > c ? m : c;
    return m > d ? m : d;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (d != 16 || H != 0 || C != 10) { fprintf(stderr, "weights must be d=16 H=0 C=10\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * W + W);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        int f = 0;
        for (int r = 0; r < 8; r += 2) {
            for (int c = 0; c < 8; c += 2) {
                unsigned char m = max4(xi[r*8 + c], xi[r*8 + c + 1], xi[(r+1)*8 + c], xi[(r+1)*8 + c + 1]);
                const float *t = T + ((long)f * 256 + m) * W;
                for (int w = 0; w < W; w++) acc[w] += t[w];
                f++;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```