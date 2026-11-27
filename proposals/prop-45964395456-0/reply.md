```hypothesis
Family: label-wide-deep (wide linear + deep MLP on shared folded embeddings).
Why: Adult has mixed ordinal/categorical features; linear models work well but miss interactions. A wide-and-deep architecture adds a tiny ReLU MLP (H=8) on top of the same per-feature lookup tables, capturing non-linear interactions with minimal extra parameters and inference cost. The folded embedding representation keeps inference fast (two table lookups per feature + small dense layer).
Failure signal: dev accuracy ≤ label-linear (0.8524) or instructions/item > 2× label-linear.
```

```python
"""Wide-and-deep for tabular label tasks: linear + 1-hidden-layer MLP on shared folded embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x57444C31, d, H, C; float32 T_lin[d][256][C], T_mlp[d][256][H], b_lin[C], b_mlp[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

H = 8
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

    # Build encodings and compute folded dimensions
    enc = []
    dims = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None]
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
        dims.append(E.shape[1])
    k = sum(dims)

    # Parameters on encoded space
    W_lin = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
    b_lin = np.zeros(C, np.float32)
    V = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b_mlp = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W_lin, b_lin, V, b_mlp, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    # Precompute encoded design matrix Z (n, k)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n = Z.shape[0]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits); P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # Forward
            lin_logits = z @ W_lin + b_lin
            mlp_in = z @ V + b_mlp
            mlp_hid = np.maximum(mlp_in, 0)
            mlp_logits = mlp_hid @ W2 + b2
            logits = lin_logits + mlp_logits
            # Backward
            g = softmax_grad(logits, yy)  # (batch, C)
            # MLP branch
            g_W2 = mlp_hid.T @ g + L2 * W2
            g_b2 = g.sum(0)
            g_mlp_hid = g @ W2.T
            g_mlp_in = g_mlp_hid * (mlp_in > 0)
            g_V = z.T @ g_mlp_in + L2 * V
            g_b_mlp = g_mlp_in.sum(0)
            # Linear branch
            g_W_lin = z.T @ g + L2 * W_lin
            g_b_lin = g.sum(0)
            grads = [g_W_lin, g_b_lin, g_V, g_b_mlp, g_W2, g_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Fold encodings into per-byte lookup tables
    T_lin = np.zeros((d, 256, C), np.float32)
    T_mlp = np.zeros((d, 256, H), np.float32)
    off = 0
    for j, E in enumerate(enc):
        T_lin[j] = E @ W_lin[off:off + E.shape[1]]
        T_mlp[j] = E @ V[off:off + E.shape[1]]
        off += E.shape[1]

    # Write weights
    magic = 0x57444C31  # "WDL1"
    out = struct.pack("<4i", magic, d, H, C)
    out += T_lin.astype("<f4").tobytes()
    out += T_mlp.astype("<f4").tobytes()
    out += b_lin.astype("<f4").tobytes()
    out += b_mlp.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Wide-and-deep inference: linear + 1-hidden-layer MLP on folded per-byte tables.
 * ./infer weights.bin in.rec N  -> writes N single-byte class records to stdout. */
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
    if (nw < 16 || hd[0] != 0x57444C31) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || H < 0 || H > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L
        + 4L * d * 256 * C
        + 4L * d * 256 * H
        + 4L * C
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *T_lin = p; p += (long)d * 256 * C;
    const float *T_mlp = p; p += (long)d * 256 * H;
    const float *b_lin = p; p += C;
    const float *b_mlp = p; p += H;
    const float *W2 = p; p += (long)H * C;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *lin_acc = malloc(sizeof(float) * C);
    float *mlp_acc = malloc(sizeof(float) * H);
    float *mlp_hid = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(lin_acc, b_lin, C * sizeof(float));
        memcpy(mlp_acc, b_mlp, H * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *tl = T_lin + ((long)j * 256 + xi[j]) * C;
            const float *tm = T_mlp + ((long)j * 256 + xi[j]) * H;
            for (int c = 0; c < C; c++) lin_acc[c] += tl[c];
            for (int h = 0; h < H; h++) mlp_acc[h] += tm[h];
        }
        for (int h = 0; h < H; h++) mlp_hid[h] = mlp_acc[h] > 0.0f ? mlp_acc[h] : 0.0f;
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += mlp_hid[h] * W2[(long)h * C + c];
            scores[c] = lin_acc[c] + sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```