```hypothesis
Family: label-localpoly (Local Polynomial Logistic Regression). Adds pairwise products of spatially adjacent pixels (210 pairs on 8×8 grid) to the 64 raw pixels, then fits a multinomial logistic regression. Digits are defined by strokes and curves; adjacent pixel interactions capture this structure better than a linear model. The 274 features keep inference fast (~4× linear cost). If dev accuracy ≤ 0.9613 (label-linear) or Ir/item > 25000, the approach fails.
```
```python
"""label-localpoly: logistic regression on raw pixels + adjacent pairwise products (8x8 grid).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only."""
import json, math, os, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100

# fixed adjacent pairs on 8x8 grid (right, down, down-right, down-left)
PAIRS = []
for r in range(8):
    for c in range(8):
        idx = r * 8 + c
        if c < 7:
            PAIRS.append((idx, idx + 1))
        if r < 7:
            PAIRS.append((idx, idx + 8))
        if r < 7 and c < 7:
            PAIRS.append((idx, idx + 9))
        if r < 7 and c > 0:
            PAIRS.append((idx, idx + 7))
N_PAIRS = len(PAIRS)  # 210
D = 64
D_EXP = D + N_PAIRS  # 274
MAGIC = 0x4C4F4350  # "PLOC"


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def expand(X):
    """X: (n, 64) uint8 -> (n, 274) float32 expanded features (not yet standardized)."""
    n = X.shape[0]
    Z = np.empty((n, D_EXP), dtype=np.float32)
    Z[:, :D] = X.astype(np.float32)
    for p, (i, j) in enumerate(PAIRS):
        Z[:, D + p] = X[:, i].astype(np.float32) * X[:, j].astype(np.float32)
    return Z


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    assert X.shape[1] == D and C == 10

    rng = np.random.default_rng(SEED)
    Z = expand(X)
    n, k = Z.shape
    assert k == D_EXP

    # standardize per feature
    mu = Z.mean(axis=0).astype(np.float32)
    sd = Z.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Z = ((Z - mu) / sd).astype(np.float32)

    # init weights
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits, dtype=np.float32)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logits = z @ W + b
            g = softmax_grad(logits, yy)
            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(axis=0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights.bin
    # header: magic, d, n_pairs, C
    # orig_mean[d], orig_std[d]
    # pair_i[n_pairs], pair_j[n_pairs]
    # pair_mean[n_pairs], pair_std[n_pairs]
    # weights[(d+n_pairs)*C], bias[C]
    orig_mu, orig_sd = mu[:D], sd[:D]
    pair_mu, pair_sd = mu[D:], sd[D:]
    pair_i = np.array([p[0] for p in PAIRS], dtype=np.uint8)
    pair_j = np.array([p[1] for p in PAIRS], dtype=np.uint8)

    out = struct.pack("<4I", MAGIC, D, N_PAIRS, C)
    out += orig_mu.astype("<f4").tobytes()
    out += orig_sd.astype("<f4").tobytes()
    out += pair_i.tobytes()
    out += pair_j.tobytes()
    out += pair_mu.astype("<f4").tobytes()
    out += pair_sd.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()

    expected = 16 + 4*D*2 + 1*N_PAIRS*2 + 4*N_PAIRS*2 + 4*D_EXP*C + 4*C
    assert len(out) == expected, f"size mismatch {len(out)} != {expected}"
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-localpoly inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (8x8 pixels). Output record: 1 byte class 0-9. */
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
    uint32_t k = buf[*pos] | (buf[*pos + 1] << 8) | (buf[*pos + 2] << 16) | ((uint32_t)buf[*pos + 3] << 24);
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, (k >> 8) & 255, (k >> 16) & 255, (k >> 24) & 255};
    fwrite(h, 1, 4, f);
    fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    if (hd[0] != 0x4C4F4350) { fprintf(stderr, "bad magic\n"); return 3; }
    uint32_t d = hd[1], n_pairs = hd[2], C = hd[3];
    if (d != 64 || C != 10 || n_pairs != 210) { fprintf(stderr, "bad dims\n"); return 3; }
    uint32_t D_EXP = d + n_pairs;  // 274
    size_t expected = 16 + 4*d*2 + 1*n_pairs*2 + 4*n_pairs*2 + 4*D_EXP*C + 4*C;
    if ((size_t)nw != expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *orig_mu = (const float *)(wb + 16);
    const float *orig_sd = orig_mu + d;
    const uint8_t *pair_i = (const uint8_t *)(orig_sd + d);
    const uint8_t *pair_j = pair_i + n_pairs;
    const float *pair_mu = (const float *)(pair_j + n_pairs);
    const float *pair_sd = pair_mu + n_pairs;
    const float *W = (const float *)(pair_sd + n_pairs);
    const float *bias = W + (long)D_EXP * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = (float *)malloc(sizeof(float) * D_EXP);
    float *scores = (float *)malloc(sizeof(float) * C);
    if (!feat || !scores) { fprintf(stderr, "oom\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != d) { fprintf(stderr, "item %ld is %u bytes, not %u\n", i, k, d); return 3; }
        for (uint32_t j = 0; j < d; j++) {
            float v = (float)xi[j];
            feat[j] = (v - orig_mu[j]) / orig_sd[j];
        }
        for (uint32_t p = 0; p < n_pairs; p++) {
            float v = (float)xi[pair_i[p]] * (float)xi[pair_j[p]];
            feat[d + p] = (v - pair_mu[p]) / pair_sd[p];
        }
        for (uint32_t c = 0; c < C; c++) scores[c] = bias[c];
        for (uint32_t f = 0; f < D_EXP; f++) {
            float w = feat[f];
            if (w == 0.0f) continue;
            const float *wrow = W + (long)f * C;
            for (uint32_t c = 0; c < C; c++) scores[c] += w * wrow[c];
        }
        unsigned char best = 0;
        for (uint32_t c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```