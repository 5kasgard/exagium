```hypothesis
Family: label-qda (Quadratic Discriminant Analysis with shrinkage).
Why: Optdigits has 64 ordinal pixel features (0–16) and 10 classes; QDA models per-class covariance, capturing feature interactions that linear models miss, while shrinkage stabilizes estimates on 2867 samples. Inference is O(C·d²) ≈ 40k multiply-adds/item, far cheaper than k-NN or GBDT.
Failure signal: Dev accuracy ≤ 0.961 (label-linear) or CPU cost > 2× label-linear.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
SHRINKAGE_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TUNING_FRAC = 0.2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def standardize(X, mean, std):
    return (X - mean) / std

def fit_qda(X, y, C, shrinkage):
    d = X.shape[1]
    priors = np.bincount(y, minlength=C).astype(np.float64)
    priors /= priors.sum()
    means = np.zeros((C, d), np.float64)
    covs = np.zeros((C, d, d), np.float64)
    for c in range(C):
        Xc = X[y == c]
        means[c] = Xc.mean(axis=0)
        if len(Xc) > 1:
            covs[c] = np.cov(Xc, rowvar=False, ddof=1)
        else:
            covs[c] = np.eye(d)
    # Shrinkage: Σ_shrunk = (1-λ)Σ + λ * trace(Σ)/d * I
    for c in range(C):
        tr = np.trace(covs[c])
        covs[c] = (1 - shrinkage) * covs[c] + shrinkage * (tr / d) * np.eye(d)
    # Invert and logdet
    inv_covs = np.zeros((C, d, d), np.float64)
    logdets = np.zeros(C, np.float64)
    for c in range(C):
        try:
            inv_covs[c] = np.linalg.inv(covs[c])
            sign, logdet = np.linalg.slogdet(covs[c])
            logdets[c] = logdet if sign > 0 else 1e30
        except np.linalg.LinAlgError:
            inv_covs[c] = np.eye(d) * 1e6
            logdets[c] = 1e30
    return priors, means, inv_covs, logdets

def predict_qda(X, priors, means, inv_covs, logdets):
    C = priors.shape[0]
    scores = np.zeros((X.shape[0], C), np.float64)
    for c in range(C):
        diff = X - means[c]
        mahal = np.einsum('ni,ij,nj->n', diff, inv_covs[c], diff)
        scores[:, c] = mahal + logdets[c] - 2 * np.log(priors[c] + 1e-30)
    return scores.argmin(axis=1)

def main():
    card = json.load(open(sys.argv[1]))
    X_raw = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X_raw.shape[1]

    # Feature standardization (z-score) on full training set
    mean = X_raw.mean(axis=0).astype(np.float64)
    std = X_raw.std(axis=0).astype(np.float64)
    std[std == 0] = 1.0
    X = standardize(X_raw.astype(np.float64), mean, std)

    n = X.shape[0]
    tune_size = int(n * TUNING_FRAC)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    tune_idx = idx[:tune_size]
    train_idx = idx[tune_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    # Grid search shrinkage on tuning split
    best_shrinkage, best_acc = 0.0, -1.0
    for s in SHRINKAGE_GRID:
        priors, means, inv_covs, logdets = fit_qda(X_train, y_train, C, s)
        pred = predict_qda(X_tune, priors, means, inv_covs, logdets)
        acc = (pred == y_tune).mean()
        if acc > best_acc:
            best_acc, best_shrinkage = acc, s

    # Retrain on all data with best shrinkage
    priors, means, inv_covs, logdets = fit_qda(X, y, C, best_shrinkage)

    # Pack weights.bin: magic, d, C, shrinkage, feature_mean[d], feature_std[d],
    # then per class: prior, mean[d], inv_cov[d*d], logdet
    magic = 0x51444131  # "QDA1"
    out = struct.pack("<4i", magic, d, C, 0)  # last int reserved
    out += struct.pack("<f", best_shrinkage)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    for c in range(C):
        out += struct.pack("<f", priors[c])
        out += means[c].astype("<f4").tobytes()
        out += inv_covs[c].astype("<f4").tobytes()
        out += struct.pack("<f", logdets[c])
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
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
    if (hd[0] != 0x51444131) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], C = hd[2];
    float *fp = (float *)(hd + 4);
    float shrinkage = *fp++;
    float *feat_mean = fp; fp += d;
    float *feat_std = fp; fp += d;

    // Per-class parameters
    float *priors = malloc(C * sizeof(float));
    float *means = malloc(C * d * sizeof(float));
    float *inv_covs = malloc(C * d * d * sizeof(float));
    float *logdets = malloc(C * sizeof(float));

    for (int c = 0; c < C; c++) {
        priors[c] = *fp++;
        memcpy(means + c * d, fp, d * sizeof(float)); fp += d;
        memcpy(inv_covs + c * d * d, fp, d * d * sizeof(float)); fp += d * d;
        logdets[c] = *fp++;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_std = malloc(d * sizeof(float));
    float *diff = malloc(d * sizeof(float));
    float *scores = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Standardize
        for (int j = 0; j < d; j++) {
            x_std[j] = (xi[j] - feat_mean[j]) / feat_std[j];
        }
        // Compute scores
        for (int c = 0; c < C; c++) {
            const float *mu = means + c * d;
            const float *ic = inv_covs + c * d * d;
            for (int j = 0; j < d; j++) diff[j] = x_std[j] - mu[j];
            float mahal = 0.0f;
            for (int r = 0; r < d; r++) {
                float sum = 0.0f;
                for (int col = 0; col < d; col++) sum += diff[col] * ic[col * d + r];
                mahal += diff[r] * sum;
            }
            scores[c] = mahal + logdets[c] - 2.0f * logf(priors[c] + 1e-30f);
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] < scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```