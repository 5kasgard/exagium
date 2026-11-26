```hypothesis
The label-quad family adds pairwise interactions among the most variable pixels, capturing local stroke patterns that a linear model misses while keeping inference fast (~1000 multiply-adds). It tunes the number of selected pixels (5,10,15,20) on the 20% tuning split, then trains on all data. If dev accuracy does not exceed 0.965, the added complexity is not justified.
```
```python
import json, math, struct, sys
import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
CANDIDATE_M = [5, 10, 15, 20]

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
    n, d = X.shape
    C = card["classes"]

    # last 20% for tuning
    split = int(n * 0.8)
    X_tr, X_tu = X[:split], X[split:]
    y_tr, y_tu = y[:split], y[split:]

    mean_tr = X_tr.mean(axis=0).astype(np.float32)
    std_tr = X_tr.std(axis=0).astype(np.float32)
    std_tr[std_tr == 0] = 1.0

    def zscore(X_data, mean, std):
        return (X_data.astype(np.float32) - mean) / std

    def augment(X_data, mean, std, selected):
        Z = zscore(X_data, mean, std)
        M = len(selected)
        K = M * (M - 1) // 2
        if K == 0:
            return Z
        Z_sel = Z[:, selected]
        inter = np.zeros((X_data.shape[0], K), dtype=np.float32)
        idx = 0
        for i in range(M):
            for j in range(i + 1, M):
                inter[:, idx] = Z_sel[:, i] * Z_sel[:, j]
                idx += 1
        return np.hstack([Z, inter])

    best_acc = -1.0
    best_M = CANDIDATE_M[0]
    best_selected = None
    best_W = None
    best_b = None

    for M in CANDIDATE_M:
        var = X_tr.var(axis=0)
        selected = np.argsort(var)[-M:]
        selected.sort()

        Z_tr = augment(X_tr, mean_tr, std_tr, selected)
        Z_tu = augment(X_tu, mean_tr, std_tr, selected)
        k = Z_tr.shape[1]

        rng = np.random.default_rng(SEED)
        W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
        b = np.zeros(C, np.float32)
        params = [W, b]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))
        for _ in range(epochs):
            order = rng.permutation(len(X_tr))
            for s in range(0, len(X_tr), BATCH):
                idx = order[s:s + BATCH]
                z, yy = Z_tr[idx], y_tr[idx]
                S = z @ W + b
                S -= S.max(axis=1, keepdims=True)
                P = np.exp(S)
                P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                grad = (P / len(yy)).astype(np.float32)
                grad_W = z.T @ grad + L2 * W
                grad_b = grad.sum(axis=0)
                grads = [grad_W, grad_b]
                t += 1
                for i, (p, gr) in enumerate(zip(params, grads)):
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

        S_tu = Z_tu @ W + b
        acc = (S_tu.argmax(axis=1) == y_tu).mean()
        if acc > best_acc:
            best_acc = acc
            best_M = M
            best_selected = selected.copy()
            best_W = W.copy()
            best_b = b.copy()

    # retrain on full data with best_M
    M = best_M
    selected = best_selected
    K = M * (M - 1) // 2
    mean_full = X.mean(axis=0).astype(np.float32)
    std_full = X.std(axis=0).astype(np.float32)
    std_full[std_full == 0] = 1.0

    Z_full = augment(X, mean_full, std_full, selected)
    k = Z_full.shape[1]

    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_full[idx], y[idx]
            S = z @ W + b
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            grad = (P / len(yy)).astype(np.float32)
            grad_W = z.T @ grad + L2 * W
            grad_b = grad.sum(axis=0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    W_orig = W[:d]
    W_inter = W[d:]

    magic = 0x51554144  # 'QUAD'
    out = struct.pack("<4I", magic, d, M, C)
    out += mean_full.astype("<f4").tobytes()
    out += std_full.astype("<f4").tobytes()
    out += selected.astype(np.uint8).tobytes()
    out += W_orig.astype("<f4").tobytes()
    out += W_inter.astype("<f4").tobytes()
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
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x51554144) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], M = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    int K = M * (M - 1) / 2;
    long expected = 16L + 8L * d + M + 4L * C * (d + K + 1);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(wb + 16);
    const float *std  = mean + d;
    const uint8_t *sel = (const uint8_t *)(wb + 16 + 8 * d);
    const float *W_orig = (const float *)(wb + 16 + 8 * d + M);
    const float *W_inter = W_orig + (long)d * C;
    const float *bias = W_inter + (long)K * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *z_sel = malloc(sizeof(float) * M);
    float *score = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            z[j] = (xi[j] - mean[j]) / std[j];
        }
        for (int c = 0; c < C; c++) score[c] = bias[c];
        for (int j = 0; j < d; j++) {
            float zj = z[j];
            const float *w = W_orig + (long)j * C;
            for (int c = 0; c < C; c++) score[c] += zj * w[c];
        }
        for (int idx = 0; idx < M; idx++) z_sel[idx] = z[sel[idx]];
        int pair = 0;
        for (int i1 = 0; i1 < M; i1++) {
            for (int i2 = i1 + 1; i2 < M; i2++) {
                float prod = z_sel[i1] * z_sel[i2];
                const float *w = W_inter + (long)pair * C;
                for (int c = 0; c < C; c++) score[c] += prod * w[c];
                pair++;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```