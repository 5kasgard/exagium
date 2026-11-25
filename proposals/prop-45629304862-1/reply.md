```hypothesis
Family: label-lowrank (low-rank factorized linear model).
Why: The 64×10 per-pixel class tables in label-linear have 6400 parameters per pixel value but only 16−17 effective values per pixel. Factorizing the tables as T[d][256][R] × U[R][C] with R=4 reduces parameters 4×, cuts memory bandwidth at inference (256 vs 640 floats loaded/item), and acts as strong regularization suited to the small optdigits training set (2867 items).
Failure signal: dev accuracy ≤ 0.9613 (label-linear) or Ir/item ≥ 5058.
```

```python
"""label-lowrank: Low-rank factorized linear model for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Exports per-pixel rank-R tables + class matrix.
Format: magic 0x4C524B4C, d, R, C; float32 T[d][256][R], U[R][C], bias[C]."""
import json, math, struct, sys

import numpy as np

R = 4
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
    d = card["input"]["bytes"]

    rng = np.random.default_rng(SEED)

    # Build per-pixel encodings (same as label-linear)
    enc = []
    dims = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)  # 256 x 1
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
        dims.append(E.shape[1])

    k = sum(dims)  # total expanded dimension
    # Low-rank factors: A[k][R], B[R][C], bias[C]
    A = (rng.uniform(-1, 1, (k, R)) * math.sqrt(6.0 / k)).astype(np.float32)
    B = (rng.uniform(-1, 1, (R, C)) * math.sqrt(6.0 / R)).astype(np.float32)
    bias = np.zeros(C, np.float32)

    params = [A, B, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Precompute Z matrix: n x k
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n = Z.shape[0]

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]  # z: B x k

            # Forward: scores = z @ A @ B + bias  (B x C)
            h = z @ A                  # B x R
            scores = h @ B + bias      # B x C

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)  # B x C

            # Backprop
            grad_B = h.T @ g + L2 * B                # R x C
            grad_bias = g.sum(axis=0)                # C
            grad_h = g @ B.T                         # B x R
            grad_A = z.T @ grad_h + L2 * A           # k x R

            grads = [grad_A, grad_B, grad_bias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export per-pixel tables T[j][256][R] = E_j @ A_j
    T_list = []
    off = 0
    for j, E in enumerate(enc):
        T_list.append(E @ A[off:off + dims[j]])  # 256 x R
        off += dims[j]
    T = np.stack(T_list, axis=0).astype("<f4")  # d x 256 x R

    # Pack weights: magic, d, R, C; T[d][256][R]; U[R][C]; bias[C]
    out = struct.pack("<4i", 0x4C524B4C, d, R, C)
    out += T.tobytes()
    out += B.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-lowrank inference: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class (argmax).
 * Weights format: magic 0x4C524B4C, d, R, C; float32 T[d][256][R], U[R][C], bias[C]. */
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
    if (nw < 16 || hd[0] != 0x4C524B4C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], R = hd[2], C = hd[3];
    if (C < 1 || C > 256 || R < 1 || R > 64 || d < 1 || d > 256) {
        fprintf(stderr, "bad dims\n"); return 3;
    }
    long expected = 16L + 4L * ((long)d * 256 * R + (long)R * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *U = T + (long)d * 256 * R;
    const float *bias = U + (long)R * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = (float *)malloc(sizeof(float) * R);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int r = 0; r < R; r++) acc[r] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * R;
            for (int r = 0; r < R; r++) acc[r] += t[r];
        }
        for (int c = 0; c < C; c++) {
            float sum = bias[c];
            for (int r = 0; r < R; r++) sum += acc[r] * U[(long)r * C + c];
            scores[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```