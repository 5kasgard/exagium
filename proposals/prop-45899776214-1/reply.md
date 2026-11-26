```hypothesis
Family: label-linear-pool (linear model on raw pixels plus fixed 2x2/4x4/8x8 spatial pooling features).
Why: Optdigits 8x8 images benefit from local spatial aggregation; adding 21 pooled sums (16+4+1) to the 64 pixels gives a linear model more expressive power with only ~30% more multiply-adds than label-linear.
Failure signal: Dev accuracy ≤ 0.9613 (label-linear) or inference cost > 1.5× label-linear.
```
```python
"""label-linear-pool: linear model on 64 pixels + 21 spatial pooling features (2x2, 4x4, 8x8 sums).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format (little-endian):
  uint32 magic=0x4C53504C ('LSPL')
  uint32 d=64, C=10, extra=21
  float32 mean[85], std[85]
  float32 W[85*10] (row-major: 85 rows, 10 cols)
  float32 b[10]
"""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x4C53504C  # 'LSPL'
EXTRA = 21  # 16 (2x2) + 4 (4x4) + 1 (8x8)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def pool_features(X):  # X: (n, 64) uint8
    n = X.shape[0]
    feats = np.zeros((n, EXTRA), dtype=np.float64)
    # 2x2 blocks: 16 blocks, each 2x2
    idx = 0
    for br in range(0, 8, 2):
        for bc in range(0, 8, 2):
            feats[:, idx] = (X[:, br*8+bc] + X[:, br*8+bc+1] +
                             X[:, (br+1)*8+bc] + X[:, (br+1)*8+bc+1])
            idx += 1
    # 4x4 blocks: 4 blocks
    for br in range(0, 8, 4):
        for bc in range(0, 8, 4):
            s = 0.0
            for dr in range(4):
                for dc in range(4):
                    s += X[:, (br+dr)*8 + (bc+dc)]
            feats[:, idx] = s
            idx += 1
    # 8x8 block: 1 block
    feats[:, idx] = X.sum(axis=1)
    return feats

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    assert X.shape[1] == 64 and C == 10

    # Build 85 features: 64 pixels + 21 pooled
    pool = pool_features(X)
    feats = np.hstack([X.astype(np.float64), pool])  # (n, 85)

    # Normalize per feature
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std[std == 0] = 1.0
    Z = ((feats - mean) / std).astype(np.float32)

    n, k = Z.shape  # k = 85
    rng = np.random.default_rng(SEED)
    W = np.zeros((k, C), np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W + b
            g = softmax_grad(logits, yy)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Write weights
    out = struct.pack("<4I", MAGIC, 64, C, EXTRA)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-linear-pool infer.c
 * ./infer weights.bin in.rec N
 * Input record: 64 bytes (8x8 pixels). Output record: 1 byte (class 0-9). */
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
    if (nw < 20) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x4C53504C) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], C = hd[2], extra = hd[3];
    if (d != 64 || C != 10 || extra != 21) { fprintf(stderr, "unexpected dims\n"); return 3; }
    int K = d + extra;  // 85
    long expected = 16L + 4L * (2L * K + (long)K * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    float *mean = (float *)(wb + 16);
    float *std  = mean + K;
    float *W    = std + K;
    float *b    = W + (long)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * K);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // First 64 features: raw pixels
        for (int j = 0; j < d; j++) feat[j] = (float)xi[j];

        // 2x2 block sums (16)
        int idx = d;
        for (int br = 0; br < 8; br += 2) {
            for (int bc = 0; bc < 8; bc += 2) {
                float sum = xi[br*8+bc] + xi[br*8+bc+1] +
                            xi[(br+1)*8+bc] + xi[(br+1)*8+bc+1];
                feat[idx++] = sum;
            }
        }
        // 4x4 block sums (4)
        for (int br = 0; br < 8; br += 4) {
            for (int bc = 0; bc < 8; bc += 4) {
                float sum = 0.0f;
                for (int dr = 0; dr < 4; dr++)
                    for (int dc = 0; dc < 4; dc++)
                        sum += xi[(br+dr)*8 + (bc+dc)];
                feat[idx++] = sum;
            }
        }
        // 8x8 sum (1)
        float sum = 0.0f;
        for (int j = 0; j < d; j++) sum += xi[j];
        feat[idx++] = sum;

        // Normalize and compute scores
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < K; j++) {
            float z = (feat[j] - mean[j]) / std[j];
            if (z != 0.0f) {
                const float *w = W + (long)j * C;
                for (int c = 0; c < C; c++) scores[c] += z * w[c];
            }
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```