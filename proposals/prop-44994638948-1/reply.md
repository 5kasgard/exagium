```hypothesis
Family: label-rff (Random Fourier Features). Approximates an RBF kernel with explicit random features, capturing non-linear pixel interactions that linear models miss. The 8x8 digit pixels have strong local correlations; RFF with a tuned bandwidth should outperform label-linear (0.9613) while staying faster than label-mlp16. If dev accuracy ≤ 0.965 or inference cost > 2× label-linear, the approach fails.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
CANDIDATES = [
    (50, 0.1), (50, 0.5), (50, 1.0), (50, 2.0),
    (100, 0.1), (100, 0.5), (100, 1.0), (100, 2.0),
    (200, 0.1), (200, 0.5), (200, 1.0), (200, 2.0),
]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_logreg(Z, y, C, rng, epochs):
    n, M = Z.shape
    W = (rng.uniform(-1, 1, (M, C)) * math.sqrt(6.0 / M)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
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
    return W, b

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]
    split = int(0.8 * n)
    idx = np.arange(n)
    tr_idx, tu_idx = idx[:split], idx[split:]

    # Standardize on full train (for final model), but for tuning use train split stats
    mu_full, sd_full = X.mean(0), X.std(0)
    sd_full[sd_full == 0] = 1.0

    best_acc, best_M, best_gamma, best_Wr, best_br, best_W2, best_b2 = -1, 0, 0, None, None, None, None

    for M, gamma in CANDIDATES:
        rng_feat = np.random.default_rng(SEED + M * 1000 + int(gamma * 100))
        W_rff = (rng_feat.normal(0, math.sqrt(gamma), (M, d))).astype(np.float32)
        b_rff = (rng_feat.uniform(0, 2*math.pi, M)).astype(np.float32)

        # Standardize using training split stats for tuning
        mu_tr, sd_tr = X[tr_idx].mean(0), X[tr_idx].std(0)
        sd_tr[sd_tr == 0] = 1.0
        X_tr_std = (X[tr_idx] - mu_tr) / sd_tr
        X_tu_std = (X[tu_idx] - mu_tr) / sd_tr

        Z_tr = np.cos(X_tr_std @ W_rff.T + b_rff).astype(np.float32)
        Z_tu = np.cos(X_tu_std @ W_rff.T + b_rff).astype(np.float32)

        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(tr_idx)))
        rng_train = np.random.default_rng(SEED + M * 10000 + int(gamma * 1000))
        W2, b2 = train_logreg(Z_tr, y[tr_idx], C, rng_train, epochs)

        # Evaluate on tuning
        S = Z_tu @ W2 + b2
        pred = S.argmax(1)
        acc = (pred == y[tu_idx]).mean()
        if acc > best_acc:
            best_acc = acc
            best_M, best_gamma = M, gamma
            best_Wr, best_br = W_rff, b_rff
            best_W2, best_b2 = W2, b2

    # Retrain on full data with best config
    X_std = (X - mu_full) / sd_full
    Z_full = np.cos(X_std @ best_Wr.T + best_br).astype(np.float32)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng_train = np.random.default_rng(SEED + best_M * 10000 + int(best_gamma * 1000))
    W2, b2 = train_logreg(Z_full, y, C, rng_train, epochs)

    # Pack weights: magic, d, M, C; mean[d], std[d]; W_rff[M*d], b_rff[M]; W2[M*C], b2[C]
    magic = 0x52464631  # 'RFF1'
    out = struct.pack("<4i", magic, d, best_M, C)
    out += mu_full.astype("<f4").tobytes()
    out += sd_full.astype("<f4").tobytes()
    out += best_Wr.astype("<f4").tobytes()
    out += best_br.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x52464631) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], M = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (2L * d + (long)M * d + M + (long)M * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *mean = p; p += d;
    const float *std  = p; p += d;
    const float *Wr   = p; p += (long)M * d;
    const float *br   = p; p += M;
    const float *W2   = p; p += (long)M * C;
    const float *b2   = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * M);
    float *score = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int m = 0; m < M; m++) {
            float acc = br[m];
            const float *wrm = Wr + (long)m * d;
            for (int j = 0; j < d; j++) {
                float xstd = (xi[j] - mean[j]) / std[j];
                acc += xstd * wrm[j];
            }
            feat[m] = cosf(acc);
        }

        for (int c = 0; c < C; c++) score[c] = b2[c];
        for (int m = 0; m < M; m++) {
            float fm = feat[m];
            const float *w2m = W2 + (long)m * C;
            for (int c = 0; c < C; c++) score[c] += fm * w2m[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```