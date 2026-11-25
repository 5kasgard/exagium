```hypothesis
Family: label-linear-aug (linear classifier with spatial aggregate features).
Why: Optdigits 8x8 pixels have strong 2D structure (strokes, loops) that a linear model on raw pixels misses. Adding 8 row sums, 8 column sums, and 16 non-overlapping 2x2 block sums (96 features total) captures this structure with minimal extra inference cost (~2x linear). The model remains a simple linear logistic regression, so training is fast, deterministic, and the inference is a few hundred FLOPs. Beats label-linear (96.1%) on accuracy; fails if spatial aggregates don't help (e.g., shuffled pixels) or if overfitting on the 96 features drops dev accuracy below 96.1%.
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

def augment(X):  # X: (n, 64) uint8 -> (n, 96) float32
    Xf = X.astype(np.float32)
    n = Xf.shape[0]
    pixels = Xf
    rows = Xf.reshape(n, 8, 8).sum(axis=2)
    cols = Xf.reshape(n, 8, 8).sum(axis=1)
    blocks = Xf.reshape(n, 4, 2, 4, 2).sum(axis=(2, 4)).reshape(n, 16)
    return np.concatenate([pixels, rows, cols, blocks], axis=1)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]

    Z = augment(X)  # (n, 96) float32
    n, d = Z.shape

    # z-score per feature (stats in float64, then cast)
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0)
    sd[sd == 0] = 1.0
    mu32 = mu.astype(np.float32)
    sd32 = sd.astype(np.float32)
    Z = (Z - mu32) / sd32

    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (d, C)) * math.sqrt(6.0 / d)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            S = z @ W + b
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # bake normalization into weights: W_eff = W / sd, b_eff = b - mu @ W_eff
    W_eff = (W / sd32[:, None]).astype(np.float32)
    b_eff = (b - mu32 @ W_eff).astype(np.float32)

    # magic "GUAL" (0x4C415547 little-endian), d=96, C=10
    out = struct.pack("<3i", 0x4C415547, d, C)
    out += W_eff.tobytes() + b_eff.tobytes()
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

static void augment(const unsigned char *x, float *feat) {
    for (int i = 0; i < 64; i++) feat[i] = (float)x[i];
    for (int r = 0; r < 8; r++) {
        float sum = 0;
        for (int c = 0; c < 8; c++) sum += (float)x[r * 8 + c];
        feat[64 + r] = sum;
    }
    for (int c = 0; c < 8; c++) {
        float sum = 0;
        for (int r = 0; r < 8; r++) sum += (float)x[r * 8 + c];
        feat[72 + c] = sum;
    }
    int idx = 80;
    for (int br = 0; br < 4; br++) {
        for (int bc = 0; bc < 4; bc++) {
            float sum = 0;
            for (int dr = 0; dr < 2; dr++) {
                for (int dc = 0; dc < 2; dc++) {
                    int r = br * 2 + dr;
                    int c = bc * 2 + dc;
                    sum += (float)x[r * 8 + c];
                }
            }
            feat[idx++] = sum;
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 12 || hd[0] != 0x4C415547) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (d != 96 || C != 10 || nw != 12 + 4L * ((long)d * C + C)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *W = (const float *)(hd + 3);
    const float *b = W + (long)d * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float feat[96], scores[10];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        augment(xi, feat);
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d; j++) {
            float f = feat[j];
            for (int c = 0; c < C; c++) scores[c] += f * W[(long)j * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```