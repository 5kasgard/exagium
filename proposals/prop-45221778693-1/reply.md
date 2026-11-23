```hypothesis
Family: label-poly2 (degree-2 polynomial features + logistic regression).
Why: Optdigits pixels have local spatial correlations; pairwise products capture stroke intersections and curvature cheaply.
Adding 2080 quadratic features to 64 linear ones (2144 total) fits in L1 cache, keeps inference ~4× linear cost, and should
push accuracy toward 0.97 without trees or deep nets. If dev accuracy ≤ 0.962, the extra features don't help.
```
```python
"""label-poly2: polynomial features (degree 2) + logistic regression.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Weight format (little-endian):
  uint32 magic=0x504F4C32 ('POL2')
  int32  d (input bytes, 64)
  int32  C (classes, 10)
  int32  F (features = d + d*(d+1)//2 = 2144)
  float32 mean[d], std[d]
  float32 W[F][C] (row-major)
  float32 b[C]
"""
import json, math, struct, sys
import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def poly_features(Z):          # Z: (n, d) float32, returns (n, F) float32
    n, d = Z.shape
    F = d + d*(d+1)//2
    out = np.empty((n, F), dtype=np.float32)
    out[:, :d] = Z
    col = d
    for i in range(d):
        out[:, col:col+d-i] = Z[:, i:i+1] * Z[:, i:]
        col += d - i
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]

    rng = np.random.default_rng(SEED)
    # z-score per pixel
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)

    # polynomial expansion
    Zp = poly_features(Z)
    n, F = Zp.shape

    # logistic regression weights
    W = (rng.uniform(-1, 1, (F, C)) * math.sqrt(6.0 / F)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            z, yy = Zp[idx], y[idx]
            g = softmax_grad(z @ W + b, yy)
            grads = [z.T @ g + L2 * W, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)

    # pack weights
    magic = 0x504F4C32
    header = struct.pack("<4i", magic, d, C, F)
    body = (mu.tobytes() + sd.tobytes() +
            W.astype("<f4").tobytes() +
            b.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```
```c
/* label-poly2 inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixels). Output record: 1 byte (class 0-9). */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint8_t *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wb[0] | wb[1]<<8 | wb[2]<<16 | wb[3]<<24;
    if (magic != 0x504F4C32) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = *(int32_t*)(wb+4);
    int C = *(int32_t*)(wb+8);
    int F = *(int32_t*)(wb+12);
    if (d != 64 || C != 10 || F != 2144) { fprintf(stderr, "unexpected dims\n"); return 3; }
    size_t expect = 16 + (size_t)d*2*4 + (size_t)F*C*4 + (size_t)C*4;
    if (nw != (long)expect) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *mean = (const float*)(wb + 16);
    const float *sd   = mean + d;
    const float *W    = sd + d;
    const float *b    = W + (size_t)F * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(d * sizeof(float));
    float *feat = malloc(F * sizeof(float));
    float *score = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) z[j] = (xi[j] - mean[j]) / sd[j];
        /* linear features */
        memcpy(feat, z, d * sizeof(float));
        /* quadratic features */
        int col = d;
        for (int a = 0; a < d; a++) {
            for (int b = a; b < d; b++) {
                feat[col++] = z[a] * z[b];
            }
        }
        /* scores */
        for (int c = 0; c < C; c++) score[c] = b[c];
        for (int f = 0; f < F; f++) {
            float fv = feat[f];
            if (fv == 0.0f) continue;
            const float *wf = W + (size_t)f * C;
            for (int c = 0; c < C; c++) score[c] += fv * wf[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```