```hypothesis
Family: label-rff (Random Fourier Features + linear output). The 8x8 digit pixels have smooth non-linear boundaries; RFF with RBF kernel approximates a Gaussian-process classifier at a fraction of kernel-SVM cost. Fixed random hidden layer (64 frequencies, tuned bandwidth) keeps inference fast (~2*D*d + 2*D*C ops) while capturing pixel interactions. If dev accuracy ≤ 0.9613 (label-linear), the random features are insufficient for this discrete, low-resolution task.
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1
D = 64
GAMMA_GRID = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = records(sys.argv[2])
    y_bytes = records(sys.argv[3])
    X = (np.frombuffer(b"".join(X_bytes), dtype=np.uint8)
         .reshape(-1, card["input"]["bytes"]).astype(np.float32) / 16.0)
    y = np.array([r[0] for r in y_bytes], dtype=np.int64)
    C = card["classes"]
    n, d = X.shape
    assert d == 64

    split = int(n * 0.8)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    best_gamma = GAMMA_GRID[0]
    best_acc = -1.0
    best_Wr = best_br = best_Wo = best_bo = None

    for gamma in GAMMA_GRID:
        rng = np.random.default_rng(SEED)
        Wr = rng.normal(0, math.sqrt(gamma), (D, d)).astype(np.float32)
        br = rng.uniform(0, 2 * math.pi, D).astype(np.float32)

        def featurize(Xb):
            Z = Xb @ Wr.T + br
            Z = np.concatenate([np.cos(Z), np.sin(Z)], axis=1) * math.sqrt(2.0 / D)
            return Z.astype(np.float32)

        Ztr = featurize(X_tr)
        Zva = featurize(X_va)
        k = Ztr.shape[1]

        rng = np.random.default_rng(SEED)
        Wo = np.zeros((k, C), np.float32)
        bo = np.zeros(C, np.float32)
        params = [Wo, bo]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))
        for _ in range(epochs):
            order = rng.permutation(len(X_tr))
            for s in range(0, len(X_tr), BATCH):
                idx = order[s:s + BATCH]
                z, yy = Ztr[idx], y_tr[idx]
                S = z @ Wo + bo
                S -= S.max(axis=1, keepdims=True)
                P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                g = (P / len(yy)).astype(np.float32)
                grads = [z.T @ g + L2 * Wo, g.sum(0)]
                t += 1
                for i, (p, gr) in enumerate(zip(params, grads)):
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

        pred = (Zva @ Wo + bo).argmax(axis=1)
        acc = (pred == y_va).mean()
        if acc > best_acc:
            best_acc = acc
            best_gamma = gamma
            best_Wr, best_br = Wr.copy(), br.copy()
            best_Wo, best_bo = Wo.copy(), bo.copy()

    # Retrain on full data with best_gamma
    rng = np.random.default_rng(SEED)
    Wr = rng.normal(0, math.sqrt(best_gamma), (D, d)).astype(np.float32)
    br = rng.uniform(0, 2 * math.pi, D).astype(np.float32)

    def featurize(Xb):
        Z = Xb @ Wr.T + br
        Z = np.concatenate([np.cos(Z), np.sin(Z)], axis=1) * math.sqrt(2.0 / D)
        return Z.astype(np.float32)

    Zfull = featurize(X)
    k = Zfull.shape[1]

    rng = np.random.default_rng(SEED)
    Wo = np.zeros((k, C), np.float32)
    bo = np.zeros(C, np.float32)
    params = [Wo, bo]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Zfull[idx], y[idx]
            S = z @ Wo + bo
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            grads = [z.T @ g + L2 * Wo, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Save: magic, d, D, C, gamma, Wr[D*d], br[D], Wo[2*D*C], bo[C]
    out = struct.pack("<4if", 0x52464631, d, D, C, best_gamma)
    out += best_Wr.astype("<f4").tobytes()
    out += best_br.astype("<f4").tobytes()
    out += Wo.astype("<f4").tobytes()
    out += bo.astype("<f4").tobytes()
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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x52464631) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    float gamma = *(float *)(hd + 4);
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 20 + 4L * ((long)D * d + D + 2L * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *Wr = (const float *)(wb + 20);
    const float *br = Wr + (long)D * d;
    const float *Wo = br + D;
    const float *bo = Wo + 2L * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * D);
    float *feat = malloc(sizeof(float) * 2 * D);
    float *scores = malloc(sizeof(float) * C);
    const float scale = sqrtf(2.0f / D);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < D; j++) {
            float sum = br[j];
            const float *wrow = Wr + (long)j * d;
            for (int p = 0; p < d; p++) sum += wrow[p] * (xi[p] / 16.0f);
            z[j] = sum;
        }
        for (int j = 0; j < D; j++) {
            feat[j] = cosf(z[j]) * scale;
            feat[D + j] = sinf(z[j]) * scale;
        }
        for (int c = 0; c < C; c++) {
            float sum = bo[c];
            for (int f = 0; f < 2 * D; f++) sum += feat[f] * Wo[(long)f * C + c];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```