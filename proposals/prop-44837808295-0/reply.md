```hypothesis
Family: label-logistic (binary logistic regression with per-byte lookup tables).
Why: For binary tasks like adult, 2-class softmax is mathematically equivalent to sigmoid on a single logit (logit = w1 - w0). By parameterizing only the logit difference, we halve the parameters and inference compute (14 feature-table lookups + adds vs 28) while retaining identical expressiveness. Expect ~2x speedup (~360 Ir/item) at label-linear's 0.8524 accuracy.
Failure: If dev accuracy drops >0.002 below label-linear, the implicit weight-tying hurts; if Ir/item >550, overhead dominates the halved arithmetic.
```
```python
"""Binary logistic regression for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: ordinal z-scored, categorical one-hot over training values.
Adam lr 3e-3, L2 1e-4, minibatch 64, epochs min(100, ceil(600000 / n)).
Exported as per-byte lookup tables: int32 magic 0x4C4F4731, d, H=0, C=1;
float32 T[d][256], bias[1]."""
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
    assert card["classes"] == 2, "Binary logistic regression requires 2 classes"
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    assert set(np.unique(y)) <= {0, 1}, "Labels must be 0 or 1"
    rng = np.random.default_rng(SEED)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape
    w = (rng.uniform(-1, 1, k) * math.sqrt(6.0 / k)).astype(np.float32)
    b = np.array(0.0, dtype=np.float32)
    params = [w, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            logit = z @ w + b
            p = 1.0 / (1.0 + np.exp(-logit))
            g = ((p - yy) / len(yy)).astype(np.float32)
            grad_w = z.T @ g + L2 * w
            grad_b = np.array(g.sum(), dtype=np.float32)
            grads = [grad_w, grad_b]
            t += 1
            for i, (p_arr, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p_arr -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    T = np.zeros((X.shape[1], 256), dtype=np.float32)
    off = 0
    for j, E in enumerate(enc):
        T[j] = (E.astype(np.float32) @ w[off:off + E.shape[1]]).astype(np.float32)
        off += E.shape[1]
    out = struct.pack("<4i", 0x4C4F4731, X.shape[1], 0, 1)
    out += T.astype("<f4").tobytes()
    out += np.array([b], dtype="<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Binary logistic regression (single logit) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte: 1 if logit > 0 else 0.
 * Weights format: int32 magic 0x4C4F4731, d, H=0, C=1; float32 T[d][256], bias[1]. */
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
    if (nw < 16 || hd[0] != 0x4C4F4731) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (H != 0 || C != 1) { fprintf(stderr, "bad weights: expected H=0 C=1\n"); return 3; }
    long expected = 16 + 4L * d * 256 + 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *bias = T + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float logit = bias[0];
        for (int j = 0; j < d; j++) {
            logit += T[(long)j * 256 + xi[j]];
        }
        unsigned char best = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```