```hypothesis
Family: label-lda (Linear Discriminant Analysis with shrinkage).
Why: LDA models class-conditional Gaussians with a shared covariance matrix, capturing pixel correlations that the linear model ignores. On 8×8 optdigits, spatial correlations are strong; LDA's Mahalanobis metric should improve over Euclidean (linear) with nearly identical inference cost (one extra standardize pass). Hyperparameter (shrinkage α) is chosen on the last 20% of train.in.
Failure signal: If dev accuracy ≤ 0.9613 (label-linear), the Gaussian assumption or shrinkage schedule is inadequate for this data.
```
```python
import json, struct, sys
import numpy as np

ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

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
    d, C, n = X.shape[1], card["classes"], X.shape[0]

    # Tuning split: last 20% in file order
    split = int(n * 0.8)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    def fit_standardize(X):
        mean = X.mean(axis=0).astype(np.float32)
        std = X.std(axis=0).astype(np.float32)
        std[std == 0] = 1.0
        return mean, (1.0 / std).astype(np.float32)

    def apply_standardize(X, mean, inv_std):
        return (X.astype(np.float32) - mean) * inv_std

    mean_tr, inv_std_tr = fit_standardize(X_tr)
    Z_tr = apply_standardize(X_tr, mean_tr, inv_std_tr)
    Z_val = apply_standardize(X_val, mean_tr, inv_std_tr)

    # Class stats on training split
    priors_tr = np.zeros(C, np.float32)
    M_tr = np.zeros((d, C), np.float32)
    for c in range(C):
        m = (y_tr == c)
        nc = m.sum()
        priors_tr[c] = nc / len(y_tr)
        if nc:
            M_tr[:, c] = Z_tr[m].mean(axis=0)

    # Pooled covariance
    Zc_tr = Z_tr - M_tr[:, y_tr].T
    Sigma_tr = (Zc_tr.T @ Zc_tr) / (len(y_tr) - C)

    best_alpha, best_acc, best_W, best_b = ALPHAS[0], -1.0, None, None
    for alpha in ALPHAS:
        Sigma_reg = Sigma_tr + alpha * np.eye(d, dtype=np.float32)
        try:
            W = np.linalg.solve(Sigma_reg, M_tr)
        except np.linalg.LinAlgError:
            continue
        b = np.zeros(C, np.float32)
        for c in range(C):
            b[c] = -0.5 * np.dot(M_tr[:, c], W[:, c]) + np.log(priors_tr[c] + 1e-12)
        acc = (Z_val @ W + b).argmax(axis=1).mean() == y_val
        acc = ( (Z_val @ W + b).argmax(axis=1) == y_val ).mean()
        if acc > best_acc:
            best_acc, best_alpha, best_W, best_b = acc, alpha, W.copy(), b.copy()

    # Retrain on full data with best_alpha
    mean_full, inv_std_full = fit_standardize(X)
    Z_full = apply_standardize(X, mean_full, inv_std_full)

    priors_full = np.zeros(C, np.float32)
    M_full = np.zeros((d, C), np.float32)
    for c in range(C):
        m = (y == c)
        nc = m.sum()
        priors_full[c] = nc / n
        if nc:
            M_full[:, c] = Z_full[m].mean(axis=0)

    Zc_full = Z_full - M_full[:, y].T
    Sigma_full = (Zc_full.T @ Zc_full) / (n - C)

    Sigma_reg = Sigma_full + best_alpha * np.eye(d, dtype=np.float32)
    W_final = np.linalg.solve(Sigma_reg, M_full)
    b_final = np.zeros(C, np.float32)
    for c in range(C):
        b_final[c] = -0.5 * np.dot(M_full[:, c], W_final[:, c]) + np.log(priors_full[c] + 1e-12)

    # Write weights: magic, d, C, mean[d], inv_std[d], W[d*C] (row-major), b[C]
    out = struct.pack("<3i", 0x4C444131, d, C)
    out += mean_full.astype("<f4").tobytes()
    out += inv_std_full.astype("<f4").tobytes()
    out += W_final.astype("<f4").tobytes()
    out += b_final.astype("<f4").tobytes()
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 12 || hd[0] != 0x4C444131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (d != 64 || C != 10) { fprintf(stderr, "unexpected d/C\n"); return 3; }
    long expected = 12 + 4L * (2 * d + d * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    float *mean = (float *)(hd + 3);
    float *inv_std = mean + d;
    float *W = inv_std + d;
    float *b = W + d * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * d);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) z[j] = (xi[j] - mean[j]) * inv_std[j];
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int j = 0; j < d; j++) sum += z[j] * W[j * C + c];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```