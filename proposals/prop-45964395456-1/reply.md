```hypothesis
Family: label-adjpoly (adjacent-pair polynomial features). Why: Optdigits 8x8 pixels have strong local correlations (strokes, curves); products of adjacent z-scored pixels capture edge/corner structure that a linear model misses. Only 112 adjacent pairs (horizontal+vertical) added to 64 linear features = 176 features total. Pixel values 0..16 allow tiny 17×17 lookup tables per pair (1.3 MB weights). Expect >0.97 dev accuracy at ~6000 Ir/item (near label-linear speed). Fails if pairwise features overfit or add no discriminative power beyond linear.
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
    C = card["classes"]
    d = card["input"]["bytes"]  # 64
    n = X.shape[0]

    # Adjacent pairs for 8x8 grid: horizontal + vertical
    pairs = []
    for r in range(8):
        for c in range(7):
            pairs.append((r * 8 + c, r * 8 + c + 1))
    for r in range(7):
        for c in range(8):
            pairs.append((r * 8 + c, (r + 1) * 8 + c))
    n_pair = len(pairs)          # 112
    n_linear = d                 # 64
    n_feat = n_linear + n_pair   # 176

    # Per-pixel z-score
    mu = X.mean(axis=0).astype(np.float64)
    sd = X.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd  # n x 64

    # Build feature matrix: linear + pairwise products
    Z_feat = np.zeros((n, n_feat), dtype=np.float64)
    Z_feat[:, :n_linear] = Z
    for p, (j, k) in enumerate(pairs):
        Z_feat[:, n_linear + p] = Z[:, j] * Z[:, k]

    # Multinomial logistic regression with Adam
    rng = np.random.default_rng(SEED)
    W = (rng.uniform(-1, 1, (n_feat, C)) * math.sqrt(6.0 / n_feat)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_feat[idx].astype(np.float32), y[idx]
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = (probs / len(yy)).astype(np.float32)

            gW = z.T @ grad + L2 * W
            gb = grad.sum(axis=0)

            t += 1
            for i, (p, gr) in enumerate(zip(params, [gW, gb])):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Lookup tables: linear_T[64][17][10], pair_T[112][17][17][10]
    linear_T = np.zeros((n_linear, 17, C), dtype=np.float32)
    for j in range(n_linear):
        for val in range(17):
            z_val = (val - mu[j]) / sd[j]
            linear_T[j, val, :] = z_val * W[j, :]

    pair_T = np.zeros((n_pair, 17, 17, C), dtype=np.float32)
    for p, (j, k) in enumerate(pairs):
        for val_j in range(17):
            z_j = (val_j - mu[j]) / sd[j]
            for val_k in range(17):
                z_k = (val_k - mu[k]) / sd[k]
                pair_T[p, val_j, val_k, :] = (z_j * z_k) * W[n_linear + p, :]

    bias = b.astype(np.float32)
    pair_idx = np.array(pairs, dtype=np.int32).flatten()

    # Weights file: magic, d, n_linear, n_pair, C, pair_idx[], linear_T, pair_T, bias
    magic = 0x41444A50  # 'PJDA'
    out = struct.pack("<5i", magic, d, n_linear, n_pair, C)
    out += pair_idx.tobytes()
    out += linear_T.astype("<f4").tobytes()
    out += pair_T.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()

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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 20 || hd[0] != 0x41444A50) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], n_linear = hd[2], n_pair = hd[3], C = hd[4];
    if (d != 64 || n_linear != 64 || n_pair != 112 || C != 10) { fprintf(stderr, "unexpected dims\n"); return 3; }

    long expected = 20L + (long)n_pair * 2 * 4 + (long)n_linear * 17 * C * 4 + (long)n_pair * 17 * 17 * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    int32_t *pair_idx = (int32_t *)(wb + 20);
    float *linear_T = (float *)(wb + 20 + (long)n_pair * 2 * 4);
    float *pair_T = linear_T + (long)n_linear * 17 * C;
    float *bias = pair_T + (long)n_pair * 17 * 17 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) acc[c] = bias[c];

        for (int j = 0; j < n_linear; j++) {
            int val = xi[j];
            if (val > 16) val = 16;
            const float *t = linear_T + ((long)j * 17 + val) * C;
            for (int c = 0; c < C; c++) acc[c] += t[c];
        }

        for (int p = 0; p < n_pair; p++) {
            int j = pair_idx[2 * p];
            int k = pair_idx[2 * p + 1];
            int val_j = xi[j];
            int val_k = xi[k];
            if (val_j > 16) val_j = 16;
            if (val_k > 16) val_k = 16;
            const float *t = pair_T + ((long)p * 17 * 17 + val_j * 17 + val_k) * C;
            for (int c = 0; c < C; c++) acc[c] += t[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```