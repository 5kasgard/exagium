```hypothesis
Family: label-rff (Random Fourier Features). Approximates an RBF kernel by projecting z-scored inputs into M=128 random Fourier features (cosine activations with fixed Gaussian weights), then learns a linear classifier on top. Deterministic via fixed seed. For optdigits' 64 ordinal pixels, this captures non-linear stroke variations better than pure linear while keeping inference at ~2× label-linear cost (64×128 + 128×10 MACs). Fails if dev accuracy ≤ 0.9613 (label-linear) or instructions/item > 2× label-linear.
```
```python
"""label-rff: Random Fourier Features + linear classifier for ordinal byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, numpy only.
Exports: magic 0x52464631, d, M, C; float32 T[d][256] (z-score tables), W[d][M], b[M], V[M][C], c[C]."""
import json, math, struct, sys
import numpy as np

M = 128          # number of random Fourier features
SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # per-feature z-score tables for ordinal bytes (0..255)
    T = np.zeros((d, 256), dtype=np.float32)
    for j in range(d):
        v = X[:, j].astype(np.float64)
        mu, sd = v.mean(), v.std() or 1.0
        T[j] = ((np.arange(256) - mu) / sd).astype(np.float32)

    # encode training set
    Z = T[np.arange(d), X.T].T.astype(np.float32)   # (n, d)
    n = Z.shape[0]

    # random Fourier weights: W ~ N(0, gamma^2), b ~ Uniform(0, 2pi)
    # gamma = 1/sigma for RBF kernel k(x,y)=exp(-||x-y||^2/(2 sigma^2))
    # heuristic: sigma^2 = median pairwise distance / 2 -> gamma ~ sqrt(2/median)
    # For simplicity, use gamma=1.0 (can be tuned on tuning split)
    gamma = 1.0
    W = (rng.normal(0, gamma, (d, M))).astype(np.float32)
    b = (rng.uniform(0, 2*math.pi, M)).astype(np.float32)

    # compute random features: phi = sqrt(2/M) * cos(Z @ W + b)
    Phi = np.cos(Z @ W + b) * math.sqrt(2.0 / M)   # (n, M)

    # linear classifier on Phi (Adam, same as label-linear)
    V = np.zeros((M, C), np.float32)
    c = np.zeros(C, np.float32)
    params = [V, c]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            phi, yy = Phi[idx], y[idx]
            S = phi @ V + c
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grads = [phi.T @ g + L2 * V, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)

    # pack weights: magic, d, M, C; T (d*256), W (d*M), b (M), V (M*C), c (C)
    magic = 0x52464631  # 'RFF1'
    out = struct.pack("<4i", magic, d, M, C)
    out += T.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += c.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-rff inference: ./infer weights.bin in.rec N
 * Input: d bytes per record (ordinal 0..255). Output: 1 byte class (0..C-1).
 * Weights format: magic 0x52464631, int32 d, M, C; float32 T[d][256], W[d][M], b[M], V[M][C], c[C]. */
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
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t*)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x52464631) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], M = hd[2], C = hd[3];
    if (d < 1 || M < 1 || C < 1 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L*((long)d*256 + (long)d*M + M + (long)M*C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *T = (const float*)(hd + 4);
    const float *W = T + (long)d * 256;
    const float *b = W + (long)d * M;
    const float *V = b + M;
    const float *c = V + (long)M * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *phi = (float*)malloc(sizeof(float) * M);
    float *sc  = (float*)malloc(sizeof(float) * C);
    if (!phi || !sc) { fprintf(stderr, "oom\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld: %u bytes, expected %d\n", i, k, d); return 3; }

        /* phi_m = sqrt(2/M) * cos( sum_j T[j][xi[j]] * W[j][m] + b[m] ) */
        const float scale = sqrtf(2.0f / M);
        for (int m = 0; m < M; m++) {
            float acc = b[m];
            for (int j = 0; j < d; j++)
                acc += T[(long)j * 256 + xi[j]] * W[(long)j * M + m];
            phi[m] = scale * cosf(acc);
        }

        /* scores = phi @ V + c */
        for (int cls = 0; cls < C; cls++) {
            float acc = c[cls];
            for (int m = 0; m < M; m++)
                acc += phi[m] * V[(long)m * C + cls];
            sc[cls] = acc;
        }

        unsigned char best = 0;
        for (int cls = 1; cls < C; cls++)
            if (sc[cls] > sc[best]) best = (unsigned char)cls;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```