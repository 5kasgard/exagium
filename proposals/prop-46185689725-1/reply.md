```hypothesis
Family: label-lda (Linear Discriminant Analysis on z-scored pixels).
Why: Opendigits features are 64 ordinal pixel counts (0..16) with roughly Gaussian class-conditional distributions; LDA is the Bayes-optimal linear classifier under shared covariance, often reaching ~97% on this dataset. It adds no inference cost over label-linear (same 64×10 multiply-add) but learns a generative model that typically outperforms discriminative logistic regression on small, well-behaved data.
Failure signal: Dev accuracy ≤ 0.962 (no improvement over label-linear) or inference cost > label-linear.
```

```python
"""label-lda: Linear Discriminant Analysis for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports per-byte score tables: magic 0x4C444131 ('LDA1'), d, H=0, C; float32 T[d][256][C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
REG = 1e-4  # covariance regularization fraction of average variance


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

    rng = np.random.default_rng(SEED)

    # Encode exactly like label-linear: ordinal -> z-score, categorical -> one-hot
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float64)
    n, k = Z.shape

    # Class statistics
    priors = np.bincount(y, minlength=C).astype(np.float64) / n
    means = np.zeros((C, k), dtype=np.float64)
    for c in range(C):
        means[c] = Z[y == c].mean(axis=0)

    # Pooled covariance
    cov = np.zeros((k, k), dtype=np.float64)
    for c in range(C):
        diff = Z[y == c] - means[c]
        cov += diff.T @ diff
    cov /= (n - C)

    # Regularize
    avg_var = np.trace(cov) / k
    cov += REG * avg_var * np.eye(k)

    # Invert
    inv_cov = np.linalg.inv(cov)

    # LDA weights: W = inv_cov @ means.T  (k x C), then scores = Z @ W + b
    W = (inv_cov @ means.T).astype(np.float32)  # k x C
    b = np.zeros(C, dtype=np.float32)
    for c in range(C):
        mc = means[c].astype(np.float32)
        b[c] = -0.5 * mc @ W[:, c] + math.log(priors[c])

    # Build per-byte score tables T[j][v][c] = sum_{f in feature j} E_j[v][f] * W[f, c]
    off = 0
    T_blocks = []
    for E in enc:
        kj = E.shape[1]
        Wj = W[off:off + kj]  # kj x C
        Tj = (E.astype(np.float32) @ Wj).astype(np.float32)  # 256 x C
        T_blocks.append(Tj)
        off += kj
    T = np.stack(T_blocks)  # d x 256 x C

    # Write weights
    magic = 0x4C444131  # 'LDA1'
    H = 0
    out = struct.pack("<4i", magic, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-lda inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output: one byte class (argmax).
 * Weights format: magic 0x4C444131, d, H=0, C; float32 T[d][256][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x4C444131) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (H != 0 || C < 1 || C > 256) { fprintf(stderr, "bad weights config\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b = T + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(scores, b, sizeof(float) * C);
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += tj[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```