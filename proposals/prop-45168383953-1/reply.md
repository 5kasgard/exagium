```hypothesis
label-block: Linear model on 64 raw pixels + 16 non-overlapping 2×2 block sums (80 features).  
Captures local stroke density that pure pixel-wise linear misses; expects ~0.97 dev accuracy at ~1.2× linear CPU cost.  
Fails if block sums add only noise (accuracy ≤ label-linear) or if 2×2 topology mismatches the data.
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
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    if d != 64:
        raise ValueError("label-block expects 64-byte input (8x8)")

    # 16 non-overlapping 2x2 block sums
    X_img = X.reshape(-1, 8, 8).astype(np.float32)
    blocks = (X_img[:, ::2, ::2] + X_img[:, ::2, 1::2] +
              X_img[:, 1::2, ::2] + X_img[:, 1::2, 1::2])
    X_feat = np.concatenate([X.astype(np.float32), blocks.reshape(-1, 16)], axis=1)  # (n, 80)

    rng = np.random.default_rng(SEED)
    mu = X_feat.mean(axis=0)
    sd = X_feat.std(axis=0)
    sd[sd == 0] = 1.0
    Z = ((X_feat - mu) / sd).astype(np.float32)

    n, k = Z.shape  # k=80
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)

    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            S = z @ W + b
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            gradW = z.T @ g + L2 * W
            gradb = g.sum(0)
            t += 1
            for param, grad, m, v in [(W, gradW, mW, vW), (b, gradb, mb, vb)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                param -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Weights: magic 'BLOC', d, H=0, C, mu[80], sd[80], W[80][C], b[C]
    magic = 0x424C4F43
    out = struct.pack("<4i", magic, d, 0, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x424C4F43) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (d != 64 || H != 0 || C != 10) { fprintf(stderr, "bad weights config\n"); return 3; }
    if (nw != 3896) { fprintf(stderr, "bad weights size %ld != 3896\n", nw); return 3; }

    float *mu = (float *)(hd + 4);
    float *sd = mu + 80;
    float *W = sd + 80;
    float *b = W + 80 * 10;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float feat[80];
    float scores[10];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }

        for (int j = 0; j < 64; j++)
            feat[j] = (xi[j] - mu[j]) / sd[j];

        for (int bi = 0; bi < 4; bi++) {
            for (int bj = 0; bj < 4; bj++) {
                int p00 = (2*bi)*8 + 2*bj;
                int p01 = p00 + 1;
                int p10 = (2*bi+1)*8 + 2*bj;
                int p11 = p10 + 1;
                float sum = xi[p00] + xi[p01] + xi[p10] + xi[p11];
                int idx = 64 + bi*4 + bj;
                feat[idx] = (sum - mu[idx]) / sd[idx];
            }
        }

        for (int c = 0; c < 10; c++) scores[c] = b[c];
        for (int j = 0; j < 80; j++) {
            float fj = feat[j];
            float *wj = W + j * 10;
            for (int c = 0; c < 10; c++) scores[c] += fj * wj[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < 10; c++) if (scores[c] > scores[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```