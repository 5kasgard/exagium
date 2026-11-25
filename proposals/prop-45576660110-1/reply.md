```hypothesis
Family: label-lda (Linear Discriminant Analysis with shrinkage).
Why: LDA finds maximally separating linear projections, often outperforming logistic regression on small Gaussian-like datasets (e.g., optdigits). Shrinkage stabilizes covariance estimates. Projects 64D→9D then uses Mahalanobis distance to class centroids.
Failure: If dev accuracy ≤ label-linear's 0.9613, the hypothesis is wrong.
```
```python
"""label-lda: Linear Discriminant Analysis with shrinkage for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (fixed seed), single thread, stdlib+numpy only.
Exports: magic, d, k, C; float32 mean[d], inv_std[d], W[d][k], mu_c[C][k], Sigma_inv[k][k], log_prior[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
SHRINK_SW_GRID = [0.01, 0.1, 0.5]
SHRINK_SIG_GRID = [0.001, 0.01, 0.1]
TUNE_FRAC = 0.2


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def standardize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def lda_fit(X, y, C, shrink_sw, shrink_sig):
    n, d = X.shape
    k = C - 1
    # Class stats
    mu_c = np.zeros((C, d), np.float64)
    n_c = np.zeros(C, np.int64)
    for c in range(C):
        mask = (y == c)
        n_c[c] = mask.sum()
        mu_c[c] = X[mask].mean(axis=0)
    mu = X.mean(axis=0)
    # Within-class scatter
    Sw = np.zeros((d, d), np.float64)
    for c in range(C):
        mask = (y == c)
        if n_c[c] > 1:
            Xc = X[mask] - mu_c[c]
            Sw += Xc.T @ Xc
    Sw /= (n - C)
    # Shrink Sw
    tr = np.trace(Sw) / d
    Sw = (1 - shrink_sw) * Sw + shrink_sw * tr * np.eye(d)
    # Between-class scatter
    Sb = np.zeros((d, d), np.float64)
    for c in range(C):
        diff = (mu_c[c] - mu).reshape(-1, 1)
        Sb += n_c[c] * (diff @ diff.T)
    Sb /= n
    # Generalized eigendecomp via Cholesky
    L = np.linalg.cholesky(Sw)
    L_inv = np.linalg.inv(L)
    M = L_inv @ Sb @ L_inv.T
    eigvals, eigvecs_M = np.linalg.eigh(M)
    idx = np.argsort(eigvals)[::-1][:k]
    W = (L_inv.T @ eigvecs_M[:, idx]).astype(np.float64)
    # Project
    Z = (X - mu) @ W
    # Class means in Z-space
    mu_c_z = np.zeros((C, k), np.float64)
    for c in range(C):
        mu_c_z[c] = Z[y == c].mean(axis=0)
    # Shared covariance in Z-space
    Sigma = np.zeros((k, k), np.float64)
    for c in range(C):
        mask = (y == c)
        Zc = Z[mask] - mu_c_z[c]
        Sigma += Zc.T @ Zc
    Sigma /= (n - C)
    tr = np.trace(Sigma) / k
    Sigma = (1 - shrink_sig) * Sigma + shrink_sig * tr * np.eye(k)
    Sigma_inv = np.linalg.inv(Sigma)
    log_prior = np.log(n_c / n)
    return W, mu, mu_c_z, Sigma_inv, log_prior


def lda_score(X, mean, inv_std, W, mu_c_z, Sigma_inv, log_prior):
    Z = ((X - mean) * inv_std) @ W
    diff = Z[:, None, :] - mu_c_z[None, :, :]          # (n, C, k)
    mahal = np.einsum('nck,kl,ncl->nc', diff, Sigma_inv, diff)
    scores = -0.5 * mahal + log_prior[None, :]
    return scores.argmax(axis=1)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]
    # Split for tuning (last 20%)
    n_tune = int(n * TUNE_FRAC)
    n_train = n - n_tune
    idx = np.arange(n)
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]
    # Standardize on train portion
    X_tr_s, mean, std = standardize(X_tr.astype(np.float64))
    inv_std = 1.0 / std
    X_te_s = (X_te.astype(np.float64) - mean) * inv_std
    # Grid search on tuning split
    best_acc, best_params = -1, None
    for shrink_sw in SHRINK_SW_GRID:
        for shrink_sig in SHRINK_SIG_GRID:
            try:
                W, _, mu_c_z, Sigma_inv, log_prior = lda_fit(X_tr_s, y_tr, C, shrink_sw, shrink_sig)
                pred = lda_score(X_te_s, np.zeros(d), inv_std, W, mu_c_z, Sigma_inv, log_prior)
                acc = (pred == y_te).mean()
                if acc > best_acc:
                    best_acc, best_params = acc, (shrink_sw, shrink_sig, W, mu_c_z, Sigma_inv, log_prior)
            except np.linalg.LinAlgError:
                continue
    shrink_sw, shrink_sig, W, mu_c_z, Sigma_inv, log_prior = best_params
    # Retrain on all data
    X_all_s, mean, std = standardize(X.astype(np.float64))
    inv_std = 1.0 / std
    W, _, mu_c_z, Sigma_inv, log_prior = lda_fit(X_all_s, y, C, shrink_sw, shrink_sig)
    k = W.shape[1]
    # Pack weights: magic, d, k, C; mean[d], inv_std[d], W[d][k], mu_c[C][k], Sigma_inv[k][k], log_prior[C]
    magic = 0x4C444131  # 'LDA1'
    hdr = struct.pack("<4i", magic, d, k, C)
    body = (
        mean.astype("<f4").tobytes() +
        inv_std.astype("<f4").tobytes() +
        W.astype("<f4").tobytes() +
        mu_c_z.astype("<f4").tobytes() +
        Sigma_inv.astype("<f4").tobytes() +
        log_prior.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(hdr + body)


if __name__ == "__main__":
    main()
```
```c
/* label-lda inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: 1 byte class per record.
 * Weights format (float32 little-endian):
 *   int32 magic(0x4C444131), d, k, C
 *   float32 mean[d], inv_std[d], W[d][k], mu_c[C][k], Sigma_inv[k][k], log_prior[C] */
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
    float *hd = (float *)slurp(argv[1], &nw);
    if (nw < 16 || *(int32_t *)hd != 0x4C444131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = (int)hd[1], k = (int)hd[2], C = (int)hd[3];
    long need = 16L + 4L * (2*d + d*k + C*k + k*k + C);
    if (nw != need) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, need); return 3; }
    float *p = hd + 4;
    float *mean = p; p += d;
    float *inv_std = p; p += d;
    float *W = p; p += d * k;
    float *mu_c = p; p += C * k;
    float *Sigma_inv = p; p += k * k;
    float *log_prior = p;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * k);
    float *diff = malloc(sizeof(float) * k);
    float *tmp = malloc(sizeof(float) * k);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        /* Project to discriminant space: z = (x - mean) * inv_std * W */
        for (int u = 0; u < k; u++) {
            float acc = 0.0f;
            for (int j = 0; j < d; j++) {
                float xj = (xi[j] - mean[j]) * inv_std[j];
                acc += xj * W[(long)j * k + u];
            }
            z[u] = acc;
        }
        /* Score each class: -0.5 * (z - mu_c)^T Sigma_inv (z - mu_c) + log_prior */
        for (int c = 0; c < C; c++) {
            for (int u = 0; u < k; u++) diff[u] = z[u] - mu_c[(long)c * k + u];
            for (int u = 0; u < k; u++) {
                float acc = 0.0f;
                for (int v = 0; v < k; v++) acc += diff[v] * Sigma_inv[(long)v * k + u];
                tmp[u] = acc;
            }
            float mahal = 0.0f;
            for (int u = 0; u < k; u++) mahal += diff[u] * tmp[u];
            scores[c] = -0.5f * mahal + log_prior[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```