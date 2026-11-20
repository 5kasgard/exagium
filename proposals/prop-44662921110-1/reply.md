```hypothesis
Family: label-elm (Extreme Learning Machine). Fixed random first layer (ReLU) with 64 hidden units, trained output layer (logistic regression). Random features provide non-linearity to beat linear accuracy; fixed first layer makes training fast and deterministic. If dev accuracy ≤ 0.9613 (linear), the random features are insufficient for this task.
```

```python
"""label-elm: Extreme Learning Machine for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Fixed random first layer (ReLU, 64 units) with z-scored inputs; trained output layer (logistic regression, Adam).
Weights format: magic 0x454C4D31, d, D, C; float32 T[d][256][D], b[D], V[D][C], c[C]."""
import json, math, struct, sys

import numpy as np

D = 64                 # hidden units
SEED = 12345           # seed for random features
TRAIN_SEED = 1         # seed for output training
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100


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
    d, C = X.shape[1], card["classes"]

    # per-pixel z-scoring
    rng = np.random.default_rng(SEED)
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0

    # random first layer weights
    W = (rng.normal(0, 1.0 / math.sqrt(d), (D, d))).astype(np.float32)
    b = rng.normal(0, 1.0, D).astype(np.float32)

    # precompute lookup table T[d][256][D]: contribution of each byte value to each hidden unit
    T = np.empty((d, 256, D), dtype=np.float32)
    for j in range(d):
        norm_vals = (np.arange(256, dtype=np.float32) - mu[j]) / sd[j]   # (256,)
        T[j] = norm_vals[:, None] * W[:, j]  # (256, D) via broadcasting

    # compute hidden activations for all training data
    Z = (X.astype(np.float32) - mu) / sd
    H = np.maximum(0.0, Z @ W.T + b)   # (n, D)

    # train output layer (logistic regression) with Adam
    n = H.shape[0]
    rng_train = np.random.default_rng(TRAIN_SEED)
    V = (rng_train.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    c = np.zeros(C, np.float32)
    params = [V, c]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng_train.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            h, yy = H[idx], y[idx]
            g = softmax_grad(h @ V + c, yy)
            grads = [h.T @ g + L2 * V, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # pack weights: magic, d, D, C; T[d][256][D]; b[D]; V[D][C]; c[C]
    magic = 0x454C4D31  # "ELM1"
    out = struct.pack("<4i", magic, d, D, C)
    out += T.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += c.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-elm inference: ./infer weights.bin in.rec N
 * Input record: d bytes (z-scored via lookup table). Hidden layer: ReLU(T * x + b). Output: linear + argmax. */
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
    if (nw < 16 || hd[0] != 0x454C4D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b = T + (long)d * 256 * D;
    const float *V = b + D;
    const float *c = V + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * D);
    float *s = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int u = 0; u < D; u++) acc[u] = b[u];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * D;
            for (int u = 0; u < D; u++) acc[u] += t[u];
        }
        for (int u = 0; u < D; u++) if (acc[u] < 0.0f) acc[u] = 0.0f;
        for (int cls = 0; cls < C; cls++) s[cls] = c[cls];
        for (int u = 0; u < D; u++) {
            float h = acc[u];
            for (int cls = 0; cls < C; cls++) s[cls] += h * V[(long)u * C + cls];
        }
        unsigned char best = 0;
        for (int cls = 1; cls < C; cls++) if (s[cls] > s[best]) best = (unsigned char)cls;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```