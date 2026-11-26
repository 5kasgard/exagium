```hypothesis
Family: label-poly2. A linear model on original 64 pixels plus all pairwise products of the 20 most informative pixels (selected by mutual information). Captures local feature interactions (e.g. strokes) without a hidden layer. Expect to beat label-linear (0.9613) and approach label-mlp16 (0.9770) at only ~2× linear's inference cost. If dev accuracy ≤ 0.965, the interaction set is insufficient.
```

```python
import json, math, struct, sys
import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
TOP_K = 20

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def mutual_info(X_col, y, C):
    # X_col in 0..16, y in 0..C-1
    joint = np.zeros((17, C), dtype=np.int64)
    for v, c in zip(X_col, y):
        joint[v, c] += 1
    total = joint.sum()
    px = joint.sum(axis=1) / total
    py = joint.sum(axis=0) / total
    pxy = joint / total
    mi = 0.0
    for i in range(17):
        if px[i] == 0: continue
        for j in range(C):
            if pxy[i, j] == 0: continue
            mi += pxy[i, j] * math.log(pxy[i, j] / (px[i] * py[j]))
    return mi

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]
    assert d == 64

    # mutual information per pixel
    mi = np.array([mutual_info(X[:, j], y, C) for j in range(d)], dtype=np.float64)
    top_idx = np.argsort(mi)[-TOP_K:][::-1]  # descending
    top_idx.sort()  # deterministic order

    # build expanded features: original 64 + pairwise products of top 20
    n_int = TOP_K * (TOP_K + 1) // 2
    D = d + n_int
    Z = np.empty((n, D), dtype=np.float32)
    Z[:, :d] = X.astype(np.float32)
    col = d
    for i in range(TOP_K):
        for j in range(i, TOP_K):
            Z[:, col] = X[:, top_idx[i]].astype(np.float32) * X[:, top_idx[j]].astype(np.float32)
            col += 1
    assert col == D

    # normalize each column
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (Z - mu) / sd

    # multinomial logistic regression with Adam
    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_W = (z.T @ probs) / len(yy) + L2 * W
            grad_b = probs.sum(axis=0) / len(yy)
            t += 1
            mW = 0.9 * mW + 0.1 * grad_W
            vW = 0.999 * vW + 0.001 * grad_W * grad_W
            mb = 0.9 * mb + 0.1 * grad_b
            vb = 0.999 * vb + 0.001 * grad_b * grad_b
            mW_hat = mW / (1 - 0.9 ** t)
            vW_hat = vW / (1 - 0.999 ** t)
            mb_hat = mb / (1 - 0.9 ** t)
            vb_hat = vb / (1 - 0.999 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + 1e-8)

    # save weights: magic, version, d, TOP_K, C, top_idx[TOP_K], mu[D], sd[D], W[D][C], b[C]
    magic = 0x504F4C59  # 'POLY'
    version = 1
    header = struct.pack("<5I", magic, version, d, TOP_K, C)
    top_bytes = top_idx.astype(np.uint8).tobytes()
    mu_bytes = mu.astype("<f4").tobytes()
    sd_bytes = sd.astype("<f4").tobytes()
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + top_bytes + mu_bytes + sd_bytes + W_bytes + b_bytes)

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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x504F4C59) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t version = hd[1];
    if (version != 1) { fprintf(stderr, "bad version\n"); return 3; }
    int d = hd[2], K = hd[3], C = hd[4];
    if (d != 64 || K != 20 || C != 10) { fprintf(stderr, "unexpected dims\n"); return 3; }
    const unsigned char *top_idx = wb + 20;
    const float *mu = (const float *)(top_idx + K);
    const float *sd = mu + (d + K*(K+1)/2);
    const float *W = sd + (d + K*(K+1)/2);
    const float *b = W + (long)(d + K*(K+1)/2) * C;
    long expected = 20 + K + 4L*(d + K*(K+1)/2) + 4L*(d + K*(K+1)/2) + 4L*(long)(d + K*(K+1)/2)*C + 4L*C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * (d + K*(K+1)/2));
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // original pixels
        for (int j = 0; j < d; j++) feat[j] = (float)xi[j];

        // pairwise products of top K
        int col = d;
        for (int a = 0; a < K; a++) {
            float va = (float)xi[top_idx[a]];
            for (int b = a; b < K; b++) {
                feat[col++] = va * (float)xi[top_idx[b]];
            }
        }
        int D = col;

        // normalize and compute scores
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < D; j++) {
            float v = (feat[j] - mu[j]) / sd[j];
            const float *wj = W + (long)j * C;
            for (int c = 0; c < C; c++) scores[c] += v * wj[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```