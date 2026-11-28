```hypothesis
Family: label-adjpair (adjacent-pairwise linear). Adds 112 horizontal/vertical adjacent-pixel products to the 64 z-scored pixels, then trains a 176-feature logistic regression with the same Adam/L2 recipe as label-linear. The 8×8 grid structure of optdigits makes local pairwise interactions highly informative for digit shape, so accuracy should exceed label-linear's 0.9613 while inference cost stays low (~2k FLOPs/item). Fails if dev accuracy ≤ 0.9613 or instructions/item > 2× label-linear.
```
```python
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    d = card["input"]["bytes"]
    C = card["classes"]
    w = int(round(math.sqrt(d)))
    if w * w != d:
        raise ValueError("input bytes not a perfect square")
    pairs = []
    for r in range(w):
        for c in range(w - 1):
            i = r * w + c
            pairs.append((i, i + 1))
    for r in range(w - 1):
        for c in range(w):
            i = r * w + c
            pairs.append((i, i + w))
    n_pairs = len(pairs)
    mu = X.mean(axis=0).astype(np.float64)
    sd = X.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    norm_table = ((np.arange(256, dtype=np.float64) - mu[:, None]) / sd[:, None]).astype(np.float32)
    Z = ((X.astype(np.float64) - mu) / sd).astype(np.float32)
    Z_pair = np.empty((X.shape[0], n_pairs), dtype=np.float32)
    for p, (i, j) in enumerate(pairs):
        Z_pair[:, p] = Z[:, i] * Z[:, j]
    Z_all = np.concatenate([Z, Z_pair], axis=1)
    n, k = Z_all.shape
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
            z, yy = Z_all[idx], y[idx]
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = (probs / len(yy)).astype(np.float32)
            gW = z.T @ grad + L2 * W
            gb = grad.sum(axis=0)
            grads = [gW, gb]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    W_lin = W[:d]
    W_pair = W[d:]
    magic = 0x41444A50
    header = struct.pack("<4i", magic, d, C, n_pairs)
    out = header
    out += norm_table.astype("<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += W_pair.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x41444A50) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], n_pairs = hd[3];
    if (d <= 0 || C <= 0 || C > 256 || n_pairs < 0) { fprintf(stderr, "bad weights header\n"); return 3; }
    int w = (int)(sqrt(d) + 0.5);
    if (w * w != d) { fprintf(stderr, "input dimension not a perfect square\n"); return 3; }
    int expected_pairs = 2 * w * (w - 1);
    if (n_pairs != expected_pairs) { fprintf(stderr, "pair count mismatch\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 + (long)d * C + (long)n_pairs * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *norm_table = (const float *)(hd + 4);
    const float *W_lin = norm_table + (long)d * 256;
    const float *W_pair = W_lin + (long)d * C;
    const float *bias = W_pair + (long)n_pairs * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *norm = malloc(sizeof(float) * d);
    float *scores = malloc(sizeof(float) * C);
    int *pair_i = malloc(sizeof(int) * n_pairs);
    int *pair_j = malloc(sizeof(int) * n_pairs);
    int pidx = 0;
    for (int r = 0; r < w; r++) {
        for (int c = 0; c < w - 1; c++) {
            int i = r * w + c;
            pair_i[pidx] = i;
            pair_j[pidx] = i + 1;
            pidx++;
        }
    }
    for (int r = 0; r < w - 1; r++) {
        for (int c = 0; c < w; c++) {
            int i = r * w + c;
            pair_i[pidx] = i;
            pair_j[pidx] = i + w;
            pidx++;
        }
    }
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            norm[j] = norm_table[(long)j * 256 + xi[j]];
        }
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int j = 0; j < d; j++) {
            float nj = norm[j];
            const float *w = W_lin + (long)j * C;
            for (int c = 0; c < C; c++) scores[c] += nj * w[c];
        }
        for (int p = 0; p < n_pairs; p++) {
            float prod = norm[pair_i[p]] * norm[pair_j[p]];
            const float *w = W_pair + (long)p * C;
            for (int c = 0; c < C; c++) scores[c] += prod * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```