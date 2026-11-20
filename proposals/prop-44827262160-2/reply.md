```hypothesis
Family: label-fm (Factorization Machine with class-specific interaction weights).
Why: Character-level language modeling relies heavily on n-gram statistics (pairwise interactions between nearby bytes). A factorization machine captures all pairwise interactions via low-rank embeddings (size K=8), while class-specific interaction weights let each next-character class exploit different n-gram patterns. This is more expressive than the linear baseline (label-linear) but far smaller and faster than GBDT or MLP.
Failure signal: If the optimal dependencies are primarily single-position (linear) or require higher-order (>2) interactions, the FM may not improve over label-linear; overfitting on the small 24k training set could also reduce accuracy.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
K = 8
LR = 0.01
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 50

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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    P = d * (d - 1) // 2

    rng = np.random.default_rng(SEED)
    E = rng.normal(0, 0.01, size=(d, 256, K)).astype(np.float32)
    W = np.zeros((d, 256, C), dtype=np.float32)
    U = np.zeros((C, P), dtype=np.float32)
    b = np.zeros(C, dtype=np.float32)

    params = [E, W, U, b]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    pairs = [(i, j) for i in range(d) for j in range(i+1, d)]

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb = X[idx]
            yb = y[idx]
            B = len(yb)

            emb = E[np.arange(d), Xb.T].transpose(1, 0, 2)  # (B, d, K)

            lin = np.zeros((B, C), dtype=np.float32)
            for i in range(d):
                lin += W[i, Xb[:, i], :]
            lin += b

            dots = np.zeros((B, P), dtype=np.float32)
            for p, (i, j) in enumerate(pairs):
                dots[:, p] = np.sum(emb[:, i] * emb[:, j], axis=1)

            inter = dots @ U.T
            scores = lin + inter

            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            grad_scores = probs
            grad_scores[np.arange(B), yb] -= 1.0
            grad_scores /= B

            grad_U = dots.T @ grad_scores
            grad_b = grad_scores.sum(axis=0)
            grad_W = np.zeros_like(W)
            for i in range(d):
                np.add.at(grad_W[i], Xb[:, i], grad_scores)

            grad_dots = grad_scores @ U
            grad_emb = np.zeros_like(emb)
            for p, (i, j) in enumerate(pairs):
                g = grad_dots[:, p, np.newaxis]
                grad_emb[:, i] += g * emb[:, j]
                grad_emb[:, j] += g * emb[:, i]

            grad_E = np.zeros_like(E)
            for i in range(d):
                np.add.at(grad_E[i], Xb[:, i], grad_emb[:, i])

            grad_E += L2 * E
            grad_W += L2 * W
            grad_U += L2 * U

            grads = [grad_E, grad_W, grad_U, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] *= 0.9; m[i] += 0.1 * gr
                v[i] *= 0.999; v[i] += 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    magic = 0x464D4C31
    header = struct.pack("<4i", magic, d, K, C)
    out = header
    out += E.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += U.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    int P = d * (d - 1) / 2;
    long expected = 16 + 4L * ((long)d * 256 * K + (long)d * 256 * C + (long)C * P + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)d * 256 * K;
    const float *U = W + (long)d * 256 * C;
    const float *b = U + (long)C * P;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    int *pairs_i = malloc(sizeof(int) * P);
    int *pairs_j = malloc(sizeof(int) * P);
    int pidx = 0;
    for (int i = 0; i < d; i++) {
        for (int j = i + 1; j < d; j++) {
            pairs_i[pidx] = i;
            pairs_j[pidx] = j;
            pidx++;
        }
    }

    float *emb = malloc(sizeof(float) * d * K);
    float *dots = malloc(sizeof(float) * P);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            const float *src = E + ((long)j * 256 + xi[j]) * K;
            memcpy(emb + j * K, src, sizeof(float) * K);
        }

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d; j++) {
            const float *w = W + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }

        for (int p = 0; p < P; p++) {
            int i1 = pairs_i[p], i2 = pairs_j[p];
            float sum = 0.0f;
            for (int kk = 0; kk < K; kk++) {
                sum += emb[i1 * K + kk] * emb[i2 * K + kk];
            }
            dots[p] = sum;
        }

        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            const float *u = U + (long)c * P;
            for (int p = 0; p < P; p++) {
                sum += u[p] * dots[p];
            }
            scores[c] += sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (scores[c] > scores[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }

    return 0;
}
```