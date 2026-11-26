```hypothesis
We propose a new family "label-mlp8": a single-hidden-layer ReLU network with 8 hidden units, using the same per-pixel z-scored lookup tables as label-linear. The small hidden layer adds non-linear feature interactions that are crucial for handwritten digits (e.g., stroke patterns) while keeping the parameter count (≈64*8 + 8*10 ≈ 592) and inference cost only marginally higher than linear (which has 640 weights). On optdigits, label-linear achieves 0.9613; label-mlp16 achieves 0.9770. A 8-unit hidden layer should interpolate between them, beating linear at a fraction of the cost of larger models. If dev accuracy does not exceed 0.9613, the family offers no advantage over label-linear.
```

```python
"""Table family with a small ReLU hidden layer (8 units) for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Exports the same binary format as the example Table family: magic, d, H, C; T[d][256][W], b1[W]; if H>0: W2[H][C], b2[C].
"""
import json, math, struct, sys

import numpy as np

H = 8                     # hidden units (new family: 8)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


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
    d = X.shape[1]

    rng = np.random.default_rng(SEED)

    # Build per-byte encoding tables (same as label-linear)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])      # (256, 1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)                                           # (256, n_cat)

    # Concatenated design matrix Z: (n, k) where k = sum of encoding widths
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape

    # Weight initialization (Xavier uniform for tanh/ReLU)
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), np.float32)
        b1 = np.zeros(C, np.float32)
        params = [W1, b1]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]

            def softmax_grad(S):
                S = S - S.max(axis=1, keepdims=True)
                P = np.exp(S)
                P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                return (P / len(yy)).astype(np.float32)

            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                g = softmax_grad(h @ W2 + b2)
                gh = (g @ W2.T) * (a > 0)
                grads = [z.T @ gh + L2 * W1, gh.sum(0), h.T @ g + L2 * W2, g.sum(0)]
            else:
                g = softmax_grad(z @ W1 + b1)
                grads = [z.T @ g + L2 * W1, g.sum(0)]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export lookup tables T[d][256][W] so inference is just table lookups + ReLU + linear
    W = H if H else C
    off = 0
    T_list = []
    for E in enc:
        # E: (256, cat_dim), W1 slice: (cat_dim, W)
        T_list.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T = np.stack(T_list)                 # (d, 256, W)

    # Pack weights.bin
    # Header: magic, d, H, C (int32 little-endian)
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    # T as float32 little-endian: d * 256 * W elements
    out += T.astype("<f4").tobytes()
    # b1: W floats
    out += b1.astype("<f4").tobytes()
    if H:
        # W2: H * C floats, b2: C floats
        out += params[2].astype("<f4").tobytes()   # W2
        out += params[3].astype("<f4").tobytes()   # b2

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-mlp8 inference: ./infer weights.bin in.rec N
 * Reads the Table family format (magic 0x54424C31, d, H, C; T[d][256][W], b1[W]; if H>0: W2[H][C], b2[C]).
 * Each input record is d bytes; output is one byte (class index 0..9). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned char *slurp(const char *path, long *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(3); }
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*size ? *size : 1);
    if (fread(buf, 1, *size, f) != (size_t)*size) { fprintf(stderr, "short read %s\n", path); exit(3); }
    fclose(f);
    return buf;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = (uint32_t)buf[*pos] | (uint32_t)buf[*pos+1]<<8 | (uint32_t)buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *rec = buf + *pos + 4;
    *pos += 4 + k;
    return rec;
}

static void put_rec(FILE *f, const unsigned char *rec, uint32_t len) {
    unsigned char hdr[4] = {len & 0xFF, (len>>8)&0xFF, (len>>16)&0xFF, (len>>24)&0xFF};
    fwrite(hdr, 1, 4, f);
    fwrite(rec, 1, len, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }

    long wsize, isize = 0;
    long N = atol(argv[3]);
    int32_t *hdr = (int32_t *)slurp(argv[1], &wsize);
    if (wsize < 16 || hdr[0] != 0x54424C31) { fprintf(stderr, "bad weights magic\n"); return 3; }

    int d = hdr[1], H = hdr[2], C = hdr[3];
    int W = H ? H : C;
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }

    long expected = 16L + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0));
    if (wsize != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hdr + 4);
    const float *b1 = T + (long)d * 256 * W;
    const float *W2 = b1 + W;
    const float *b2 = W2 + (long)H * C;

    unsigned char *ibuf = N ? slurp(argv[2], &isize) : NULL;
    long ipos = 0;

    float *acc = malloc(sizeof(float) * W);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k;
        const unsigned char *x = next_rec(ibuf, isize, &ipos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld: %u bytes, expected %d\n", i, k, d); return 3; }

        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *tp = T + ((long)j * 256 + x[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += tp[w];
        }

        const float *out_scores = acc;
        if (H) {
            for (int c = 0; c < C; c++) scores[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float hu = acc[u] > 0.0f ? acc[u] : 0.0f;
                if (hu == 0.0f) continue;
                const float *w2u = W2 + (long)u * C;
                for (int c = 0; c < C; c++) scores[c] += hu * w2u[c];
            }
            out_scores = scores;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (out_scores[c] > out_scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(acc);
    free(scores);
    if (ibuf) free(ibuf);
    free(hdr);
    return 0;
}
```