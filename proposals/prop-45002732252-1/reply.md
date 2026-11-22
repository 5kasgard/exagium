```hypothesis
Family: label-maxout (2-piece maxout network with 16 hidden units).
Why: Maxout's piecewise linear convex activation captures spatial stroke patterns in 8x8 digits better than ReLU at same width; the sum-of-embeddings inference extends naturally with two embedding tables and element-wise max. Expect ~0.98 accuracy at ~2x MLP16 cost.
Failure: Dev accuracy ≤ 0.965 or Ir/item > 15000 without accuracy gain over label-mlp16.
```
```python
"""label-maxout: 2-piece maxout network for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: per-byte z-score for ordinal (like label-linear). Two embedding tables per position,
element-wise max activation, then linear output. Adam lr 3e-3, L2 1e-4, minibatch 64, epochs min(100, ceil(600000/n)).
Exported: magic 0x4D41584F, d, H, C; float32 T1[d][256][H], T2[d][256][H], b1a[H], b1b[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

H = 16
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

    W1a = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1a = np.zeros(H, np.float32)
    W1b = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1b = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1a, b1a, W1b, b1b, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]

            a1 = z @ W1a + b1a
            a2 = z @ W1b + b1b
            h = np.maximum(a1, a2)

            logits = h @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)

            gh = g @ W2.T
            mask = (a1 >= a2).astype(np.float32)
            ga1 = gh * mask
            ga2 = gh * (1.0 - mask)

            grads = [
                z.T @ ga1 + L2 * W1a,
                ga1.sum(0),
                z.T @ ga2 + L2 * W1b,
                ga2.sum(0),
                h.T @ g + L2 * W2,
                g.sum(0),
            ]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    off = 0
    T1, T2 = [], []
    for E in enc:
        w = E.shape[1]
        T1.append((E.astype(np.float32) @ W1a[off:off + w]).astype(np.float32))
        T2.append((E.astype(np.float32) @ W1b[off:off + w]).astype(np.float32))
        off += w

    out = struct.pack("<4i", 0x4D41584F, X.shape[1], H, C)
    out += np.stack(T1).astype("<f4").tobytes()
    out += np.stack(T2).astype("<f4").tobytes()
    out += b1a.astype("<f4").tobytes()
    out += b1b.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-maxout: 2-piece maxout network for label tasks. ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weights format: int32 magic 0x4D41584F, d, H, C; float32 T1[d][256][H], T2[d][256][H],
 * b1a[H], b1b[H], W2[H][C], b2[C]. */
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
    if (nw < 16 || hd[0] != 0x4D41584F) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || H < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (2L * d * 256 * H + 2L * H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T1 = (const float *)(hd + 4);
    const float *T2 = T1 + (long)d * 256 * H;
    const float *b1a = T2 + (long)d * 256 * H;
    const float *b1b = b1a + H;
    const float *W2 = b1b + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc1 = malloc(sizeof(float) * H);
    float *acc2 = malloc(sizeof(float) * H);
    float *h = malloc(sizeof(float) * H);
    float *s = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int u = 0; u < H; u++) { acc1[u] = b1a[u]; acc2[u] = b1b[u]; }

        for (int j = 0; j < d; j++) {
            const float *t1 = T1 + ((long)j * 256 + xi[j]) * H;
            const float *t2 = T2 + ((long)j * 256 + xi[j]) * H;
            for (int u = 0; u < H; u++) {
                acc1[u] += t1[u];
                acc2[u] += t2[u];
            }
        }

        for (int u = 0; u < H; u++) h[u] = acc1[u] > acc2[u] ? acc1[u] : acc2[u];

        for (int c = 0; c < C; c++) s[c] = b2[c];
        for (int u = 0; u < H; u++) {
            float hu = h[u];
            for (int c = 0; c < C; c++) s[c] += hu * W2[(long)u * C + c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```