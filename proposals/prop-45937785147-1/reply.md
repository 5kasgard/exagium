```hypothesis
Family: label-lda (Linear Discriminant Analysis with shrinkage).
Why: LDA assumes Gaussian class-conditional densities with shared covariance; with shrinkage it stabilizes estimation on small datasets and often outperforms logistic regression when assumptions approximately hold. Optdigits pixels are roughly Gaussian per class.
Failure: If dev accuracy <= label-linear (0.9613) or inference cost exceeds label-linear by >20%.
```

```python
import json, struct, sys
import numpy as np

SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    np.random.seed(SEED)
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]

    # Split into train and tune (last 20% in file order)
    tune_size = n // 5
    train_size = n - tune_size
    X_train, X_tune = X[:train_size], X[train_size:]
    y_train, y_tune = y[:train_size], y[train_size:]

    X_train = X_train.astype(np.float32)
    X_tune = X_tune.astype(np.float32)

    classes = np.arange(C)
    priors = np.array([np.sum(y_train == c) for c in classes], dtype=np.float32) / train_size

    means = np.zeros((C, d), dtype=np.float32)
    for c in classes:
        means[c] = X_train[y_train == c].mean(axis=0)

    S = np.zeros((d, d), dtype=np.float32)
    for c in classes:
        Xc = X_train[y_train == c]
        if len(Xc) > 1:
            S += (Xc - means[c]).T @ (Xc - means[c])
    S /= (train_size - C)

    best_alpha = 0.0
    best_acc = -1.0
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        trace_S = np.trace(S)
        Sigma = (1 - alpha) * S + alpha * (trace_S / d) * np.eye(d, dtype=np.float32)
        Sigma += 1e-6 * np.eye(d, dtype=np.float32)
        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            continue
        W = Sigma_inv @ means.T
        b = np.zeros(C, dtype=np.float32)
        for c in classes:
            b[c] = -0.5 * means[c] @ Sigma_inv @ means[c] + np.log(priors[c] + 1e-10)
        scores = X_tune @ W + b
        preds = np.argmax(scores, axis=1)
        acc = np.mean(preds == y_tune)
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    # Retrain on full data
    X_full = X.astype(np.float32)
    priors_full = np.array([np.sum(y == c) for c in classes], dtype=np.float32) / n
    means_full = np.zeros((C, d), dtype=np.float32)
    for c in classes:
        means_full[c] = X_full[y == c].mean(axis=0)
    S_full = np.zeros((d, d), dtype=np.float32)
    for c in classes:
        Xc = X_full[y == c]
        if len(Xc) > 1:
            S_full += (Xc - means_full[c]).T @ (Xc - means_full[c])
    S_full /= (n - C)
    trace_S_full = np.trace(S_full)
    Sigma_full = (1 - best_alpha) * S_full + best_alpha * (trace_S_full / d) * np.eye(d, dtype=np.float32)
    Sigma_full += 1e-6 * np.eye(d, dtype=np.float32)
    Sigma_inv_full = np.linalg.inv(Sigma_full)
    W_full = Sigma_inv_full @ means_full.T
    b_full = np.zeros(C, dtype=np.float32)
    for c in classes:
        b_full[c] = -0.5 * means_full[c] @ Sigma_inv_full @ means_full[c] + np.log(priors_full[c] + 1e-10)

    magic = 0x4C444131  # "LDA1" little-endian
    with open(sys.argv[4], "wb") as f:
        f.write(struct.pack("<iii", magic, d, C))
        f.write(W_full.astype("<f4").tobytes())
        f.write(b_full.astype("<f4").tobytes())

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
    if (nw < 12 || hd[0] != 0x4C444131) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dimensions\n"); return 3; }
    long expected = 12 + 4L * d * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(hd + 3);
    const float *b = W + (long)d * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < d; j++) {
            float xj = (float)xi[j];
            const float *wj = W + (long)j * C;
            for (int c = 0; c < C; c++) scores[c] += xj * wj[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```