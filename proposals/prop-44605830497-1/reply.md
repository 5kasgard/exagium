```hypothesis
Family: label-lda (Linear Discriminant Analysis with pooled covariance).
Why: LDA learns a linear classifier by modeling class-conditional Gaussians with shared covariance, which often outperforms logistic regression on small, well-separated datasets like optdigits. It has identical inference cost to label-linear (one matrix-vector multiply per sample) but a different inductive bias.
Failure signal: If dev accuracy ≤ 0.9613 (label-linear's score), the Gaussian assumption adds no value here.
```

```python
import json, struct, sys
import numpy as np

SEED = 1
LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

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
    d = card["input"]["bytes"]
    C = card["classes"]
    X_bytes = b"".join(records(sys.argv[2]))
    y_bytes = b"".join(records(sys.argv[3]))
    n = len(X_bytes) // d
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(n, d).astype(np.float64)
    y = np.frombuffer(y_bytes, dtype=np.uint8).astype(np.int64)
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]
    
    best_lambda = LAMBDAS[0]
    best_acc = -1.0
    for lam in LAMBDAS:
        mu = np.zeros((C, d), dtype=np.float64)
        for c in range(C):
            mu[c] = X_train[y_train == c].mean(axis=0)
        Sigma = np.zeros((d, d), dtype=np.float64)
        for c in range(C):
            Xc = X_train[y_train == c]
            if len(Xc) > 1:
                Sigma += (Xc - mu[c]).T @ (Xc - mu[c])
        Sigma /= (len(X_train) - C)
        Sigma += lam * np.eye(d)
        W = np.linalg.solve(Sigma, mu.T).T
        priors = np.bincount(y_train, minlength=C) / len(X_train)
        b = -0.5 * np.sum(mu * W, axis=1) + np.log(priors + 1e-12)
        scores = X_tune @ W.T + b
        pred = np.argmax(scores, axis=1)
        acc = np.mean(pred == y_tune)
        if acc > best_acc:
            best_acc = acc
            best_lambda = lam
    mu = np.zeros((C, d), dtype=np.float64)
    for c in range(C):
        mu[c] = X[y == c].mean(axis=0)
    Sigma = np.zeros((d, d), dtype=np.float64)
    for c in range(C):
        Xc = X[y == c]
        if len(Xc) > 1:
            Sigma += (Xc - mu[c]).T @ (Xc - mu[c])
    Sigma /= (n - C)
    Sigma += best_lambda * np.eye(d)
    W = np.linalg.solve(Sigma, mu.T).T
    priors = np.bincount(y, minlength=C) / n
    b = -0.5 * np.sum(mu * W, axis=1) + np.log(priors + 1e-12)
    magic = 0x4C444131
    out = struct.pack("<4i", magic, d, C, 0)
    out += W.astype(np.float32).tobytes()
    out += b.astype(np.float32).tobytes()
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
    if (nw < 16 || hd[0] != 0x4C444131) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], C = hd[2];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)C * d + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(hd + 4);
    const float *b = W + (long)C * d;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            const float *wc = W + (long)c * d;
            for (int j = 0; j < d; j++) {
                sum += xi[j] * wc[j];
            }
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```