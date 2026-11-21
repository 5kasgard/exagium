```hypothesis
Family: label-lda (Linear Discriminant Analysis with ridge regularization).
Why: LDA is a generative linear classifier assuming Gaussian class-conditional densities with shared covariance. The 64 pixel-count features of optdigits are roughly Gaussian per digit and share covariance structure, making LDA a natural fit. It has no iterative hyperparameters, trains in closed form (deterministic), and its inference cost is identical to label-linear (one dot product per class).
Failure mode: If class covariances differ significantly or are non-Gaussian, LDA's shared-covariance assumption will hurt accuracy below label-linear's discriminative logistic regression.
```
```python
"""label-lda: Linear Discriminant Analysis with ridge regularization for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin  (deterministic, single thread)
Preprocessing: ordinal features z-scored using training mean/std.
Model: LDA on z-scored features with ridge parameter alpha chosen on last 20% of train.in (file order),
then retrained on all train.in. Exported in the table-family format (magic 0x54424C31, d, H=0, C)."""
import json, math, struct, sys

import numpy as np


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
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]

    # Split: last 20% for tuning (file order)
    tune_size = n // 5
    train_size = n - tune_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_tune, y_tune = X[train_size:], y[train_size:]

    # Compute encoding parameters on training split
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X_train[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X_train[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)

    def encode(X_raw, enc_list):
        Z_parts = [E[X_raw[:, j]] for j, E in enumerate(enc_list)]
        return np.concatenate(Z_parts, axis=1).astype(np.float32)

    Z_train = encode(X_train, enc)
    Z_tune = encode(X_tune, enc)

    # Class stats on training split
    priors = np.bincount(y_train, minlength=C).astype(np.float64) / train_size
    class_means = np.zeros((C, d), dtype=np.float64)
    for c in range(C):
        class_means[c] = Z_train[y_train == c].mean(axis=0)

    # Pooled covariance
    Sigma = np.zeros((d, d), dtype=np.float64)
    for c in range(C):
        diff = Z_train[y_train == c] - class_means[c]
        Sigma += diff.T @ diff
    Sigma /= (train_size - C)

    # Ridge parameters to try (log scale)
    alphas = [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0]
    best_alpha = alphas[0]
    best_acc = -1.0

    for alpha in alphas:
        Sigma_reg = Sigma + alpha * np.eye(d)
        try:
            Sigma_inv = np.linalg.inv(Sigma_reg)
        except np.linalg.LinAlgError:
            continue
        W1 = (Sigma_inv @ class_means.T).T.astype(np.float32)  # C x d -> d x C after transpose? Actually class_means is C x d, Sigma_inv is d x d. Sigma_inv @ class_means.T is d x C. We need W1 as d x C for Z @ W1.
        W1 = (Sigma_inv @ class_means.T).astype(np.float32)  # d x C
        b1 = np.zeros(C, dtype=np.float32)
        for c in range(C):
            mu_c = class_means[c]
            b1[c] = -0.5 * mu_c @ Sigma_inv @ mu_c + math.log(priors[c] + 1e-12)

        # Evaluate on tuning split
        scores = Z_tune @ W1 + b1
        pred = scores.argmax(axis=1)
        acc = (pred == y_tune).mean()
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    # Retrain on full training set with best_alpha
    # Recompute encoding parameters on full data
    enc_full = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc_full.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc_full.append(E)

    Z_full = encode(X, enc_full)

    priors_full = np.bincount(y, minlength=C).astype(np.float64) / n
    class_means_full = np.zeros((C, d), dtype=np.float64)
    for c in range(C):
        class_means_full[c] = Z_full[y == c].mean(axis=0)

    Sigma_full = np.zeros((d, d), dtype=np.float64)
    for c in range(C):
        diff = Z_full[y == c] - class_means_full[c]
        Sigma_full += diff.T @ diff
    Sigma_full /= (n - C)

    Sigma_reg = Sigma_full + best_alpha * np.eye(d)
    Sigma_inv = np.linalg.inv(Sigma_reg)
    W1 = (Sigma_inv @ class_means_full.T).astype(np.float32)  # d x C
    b1 = np.zeros(C, dtype=np.float32)
    for c in range(C):
        mu_c = class_means_full[c]
        b1[c] = -0.5 * mu_c @ Sigma_inv @ mu_c + math.log(priors_full[c] + 1e-12)

    # Build lookup tables T[j][v] = enc_full[j][v] * W1[j, :]  (since each enc is 256x1)
    T_list = []
    for j, E in enumerate(enc_full):
        # E is (256, 1), W1[j:j+1, :] is (1, C) -> (256, C)
        T_list.append((E.astype(np.float32) @ W1[j:j+1, :]).astype(np.float32))

    H = 0
    out = struct.pack("<4i", 0x54424C31, d, H, C) + np.stack(T_list).astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```