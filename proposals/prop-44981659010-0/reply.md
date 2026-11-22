```hypothesis
Factorization Machine (FM) with rank-4 pairwise interactions on the encoded features. Adult benefits from feature crosses (e.g., education×occupation) but full quadratic models are too large; FM captures them in O(d·k) time. With k=4 the model adds only ~28k parameters, keeping inference ~2× linear cost while targeting >0.86 dev accuracy. If accuracy ≤0.8524 or Ir/item >2000 the family is not worthwhile.
```

```python
"""Factorization Machine family for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Encoding: ordinal -> z-scored scalar, categorical -> one-hot over training values.
Model: bias + linear + rank-4 FM interactions per class.
Optimizer: Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights layout (little-endian):
  int32 magic=0x464D4C31, d, k, C
  float32 linear_table[d][256][C]
  float32 bias[C]
  float32 factor_table[d][256][k][C]"""
import json, math, struct, sys

import numpy as np

K = 4
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
    rng = np.random.default_rng(SEED)

    # Build per-feature encoders
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])          # (256,1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)                                              # (256, n_cat)

    # Concatenated encoded matrix
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, D = Z.shape

    # Model parameters on encoded space
    W = np.zeros((D, C), np.float32)
    b = np.zeros(C, np.float32)
    V = (rng.uniform(-1, 1, (D, K, C)) * math.sqrt(6.0 / D)).astype(np.float32)

    params = [W, b, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            B = len(yy)

            # Forward
            lin = z @ W + b                                            # (B,C)
            inter = np.zeros((B, C), np.float32)
            for c in range(C):
                Vc = V[:, :, c]                                        # (D,K)
                ZV = z @ Vc                                            # (B,K)
                Q = (z**2) @ (Vc**2)                                   # (B,K)
                inter[:, c] = 0.5 * np.sum(ZV**2 - Q, axis=1)
            logits = lin + inter

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            dL = probs / B                                             # (B,C)

            # Gradients
            dW = z.T @ dL + L2 * W
            db = dL.sum(axis=0)
            dV = np.zeros_like(V)
            for c in range(C):
                dLc = dL[:, c]                                         # (B,)
                Vc = V[:, :, c]                                        # (D,K)
                ZV = z @ Vc                                            # (B,K)
                # term1 = sum_b dLc[b] * ZV[b,f] * z[b,i]  -> (D,K)
                term1 = z.T @ (dLc[:, None] * ZV)
                # term2 = sum_b dLc[b] * z[b,i]^2 * Vc[i,f] -> (D,K)
                coef = dLc @ (z**2)                                    # (D,)
                term2 = coef[:, None] * Vc
                dV[:, :, c] = (term1 - term2) / B + L2 * Vc

            grads = [dW, db, dV]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)

    # Project parameters to raw-byte lookup tables
    d = X.shape[1]
    linear_table = np.zeros((d, 256, C), np.float32)
    factor_table = np.zeros((d, 256, K, C), np.float32)
    off = 0
    for j, E in enumerate(enc):
        w_slice = W[off:off + E.shape[1]]
        linear_table[j] = E.astype(np.float32) @ w_slice
        for c in range(C):
            Vc_slice = V[off:off + E.shape[1], :, c]
            factor_table[j, :, :, c] = E.astype(np.float32) @ Vc_slice
        off += E.shape[1]

    # Pack weights
    out = struct.pack("<4i", 0x464D4C31, d, K, C)
    out += linear_table.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += factor_table.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record = d bytes (d=14). Output record = 1 byte class index.
 * Weights format: int32 magic=0x464D4C31, d, k, C; float32 linear[d][256][C], bias[C], factor[d][256][k][C]. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 16L
        + 4L * d * 256 * C
        + 4L * C
        + 4L * d * 256 * K * C;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *lin = (const float *)(hd + 4);
    const float *bias = lin + (long)d * 256 * C;
    const float *fac = bias + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = bias[c];

        for (int j = 0; j < d; j++) {
            const float *lj = lin + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += lj[c];
        }

        for (int c = 0; c < C; c++) {
            float sum_f[K];
            for (int f = 0; f < K; f++) sum_f[f] = 0.0f;
            float sq_f[K];
            for (int f = 0; f < K; f++) sq_f[f] = 0.0f;

            for (int j = 0; j < d; j++) {
                const float *fj = fac + (((long)j * 256 + xi[j]) * K + 0) * C + c;
                for (int f = 0; f < K; f++) {
                    float v = fj[f * C];
                    sum_f[f] += v;
                    sq_f[f] += v * v;
                }
            }
            for (int f = 0; f < K; f++) scores[c] += 0.5f * (sum_f[f] * sum_f[f] - sq_f[f]);
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```