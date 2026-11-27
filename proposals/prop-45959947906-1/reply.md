```hypothesis
The label-spatial family adds pairwise interactions between spatially adjacent pixels in the 8x8 grid to a linear model. This captures local stroke correlations that a purely linear model misses, while adding only 210 interaction features (vs 64 linear). Training uses the same Adam optimizer as label-linear. Inference computes normalized pixel values from a lookup table and evaluates the linear and interaction terms. This should improve accuracy over label-linear (0.9613) toward label-mlp16 (0.9770) with only a modest increase in inference cost (extra 210 multiplications per sample). If accuracy does not exceed label-linear, the spatial interactions are not informative for this task.
```
```python
"""label-spatial: linear model with spatial pairwise interactions for 8x8 digit images.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x54415053 ('SPAT'), d, C, num_pairs; float32 norm_table[d][256],
T_lin[d][256][C], b_lin[C], pairs[num_pairs][2] (uint8), W_pair[num_pairs][C]."""
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


def spatial_pairs(d=64, rows=8, cols=8):
    """Return list of (i,j) for 4-neighbor connectivity (right, down, down-right, down-left)."""
    pairs = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if c + 1 < cols:  # right
                pairs.append((idx, idx + 1))
            if r + 1 < rows:  # down
                pairs.append((idx, idx + cols))
            if r + 1 < rows and c + 1 < cols:  # down-right
                pairs.append((idx, idx + cols + 1))
            if r + 1 < rows and c - 1 >= 0:  # down-left
                pairs.append((idx, idx + cols - 1))
    return pairs


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # normalization per feature (ordinal only per card)
    mu = X.mean(axis=0).astype(np.float64)
    sd = X.std(axis=0).astype(np.float64)
    sd[sd == 0] = 1.0
    norm_table = ((np.arange(256, dtype=np.float64)[:, None] - mu) / sd).T.astype(np.float32)  # (d, 256)

    # normalized features for all samples
    Z_lin = norm_table[np.arange(d), X.T].T  # (n, d)

    # spatial interaction features
    pairs = spatial_pairs(d)
    num_pairs = len(pairs)
    Z_int = np.empty((X.shape[0], num_pairs), dtype=np.float32)
    for p, (i, j) in enumerate(pairs):
        Z_int[:, p] = Z_lin[:, i] * Z_lin[:, j]

    Z = np.concatenate([Z_lin, Z_int], axis=1).astype(np.float32)  # (n, d + num_pairs)
    n, k = Z.shape

    # initialize weights
    W = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(scores, yy):
        scores -= scores.max(axis=1, keepdims=True)
        P = np.exp(scores)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            g = softmax_grad(z @ W + b, yy)
            grad_W = z.T @ g + L2 * W
            grad_b = g.sum(0)
            grads = [grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    W_lin = W[:d]          # (d, C)
    W_pair = W[d:]         # (num_pairs, C)

    # build linear lookup table T_lin[j][val][c] = norm_table[j,val] * W_lin[j,c]
    T_lin = np.empty((d, 256, C), dtype=np.float32)
    for j in range(d):
        T_lin[j] = norm_table[j][:, None] * W_lin[j]  # (256, C)

    # write weights.bin
    MAGIC = 0x54415053  # 'SPAT' little-endian
    out = struct.pack("<4i", MAGIC, d, C, num_pairs)
    out += norm_table.astype("<f4").tobytes()
    out += T_lin.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += np.array(pairs, dtype=np.uint8).tobytes()
    out += W_pair.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-spatial inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (8x8 pixels). Output record: 1 byte class.
 * Weights format: magic 0x54415053, int32 d, C, num_pairs;
 * float32 norm_table[d][256], T_lin[d][256][C], b_lin[C];
 * uint8 pairs[num_pairs][2]; float32 W_pair[num_pairs][C]. */
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
    if (nw < 16 || hd[0] != 0x54415053) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], num_pairs = hd[3];
    if (d != 64 || C != 10 || num_pairs != 210) { fprintf(stderr, "unexpected dimensions\n"); return 3; }

    float *norm_table = (float *)(hd + 4);
    float *T_lin = norm_table + d * 256;
    float *b_lin = T_lin + d * 256 * C;
    uint8_t *pairs = (uint8_t *)(b_lin + C);
    float *W_pair = (float *)(pairs + num_pairs * 2);

    long expected = 16 + 4L * (d * 256 + d * 256 * C + C + num_pairs * 2) + 4L * num_pairs * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[10];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = b_lin[c];

        for (int j = 0; j < d; j++) {
            const float *t = T_lin + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += t[c];
        }

        for (int p = 0; p < num_pairs; p++) {
            int i1 = pairs[2 * p], i2 = pairs[2 * p + 1];
            float n1 = norm_table[i1 * 256 + xi[i1]];
            float n2 = norm_table[i2 * 256 + xi[i2]];
            float prod = n1 * n2;
            const float *wp = W_pair + (long)p * C;
            for (int c = 0; c < C; c++) scores[c] += prod * wp[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```