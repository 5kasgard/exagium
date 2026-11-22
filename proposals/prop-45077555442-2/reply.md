```hypothesis
Family: label-mlp2x64 (two hidden layers of 64 units, per-position 64-d embeddings).
Why: The 16-byte context likely has high-order interactions that a linear model misses; label-mlp16's 16-unit layer is too narrow. Two 64-unit layers add capacity while keeping inference ~100k Ir/item (well below label-gbdt's 131k) and weights ~1.1 MB.
Failure: Dev accuracy ≤ 0.3375 (label-linear) or inference cost > 150k Ir/item.
```

```python
"""Table family with two hidden layers (64, 64) for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x54424C32, d, H1, H2, C; float32 T[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C]."""
import json, math, struct, sys

import numpy as np

H1, H2 = 64, 64
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 1200000, 200


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

    # Build per-position one-hot encodings (same as label-linear)
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j]
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(v)
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)

    # Concatenated design matrix
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, k = Z.shape

    # Initialize weights
    W1 = (rng.uniform(-1, 1, (k, H1)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
    b3 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]

            # Forward
            a1 = z @ W1 + b1
            h1 = np.maximum(a1, 0)
            a2 = h1 @ W2 + b2
            h2 = np.maximum(a2, 0)
            logits = h2 @ W3 + b3

            # Backward
            g3 = softmax_grad(logits, yy)                 # (B, C)
            gh2 = g3 @ W3.T * (a2 > 0)                    # (B, H2)
            g2 = h1.T @ gh2 + L2 * W2                      # (H1, H2)
            gb2 = gh2.sum(0)                              # (H2,)
            gh1 = (gh2 @ W2.T) * (a1 > 0)                  # (B, H1)
            g1 = z.T @ gh1 + L2 * W1                       # (k, H1)
            gb1 = gh1.sum(0)                              # (H1,)
            g3_W = h2.T @ g3 + L2 * W3                     # (H2, C)
            gb3 = g3.sum(0)                               # (C,)

            grads = [g1, gb1, g2, gb2, g3_W, gb3]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build per-position tables T[j][byte] = E_j[byte] @ W1_slice
    T_list = []
    off = 0
    for E in enc:
        T_list.append((E @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T = np.stack(T_list)  # (d, 256, H1)

    # Write weights.bin
    d = X.shape[1]
    magic = 0x54424C32  # "TBL2"
    header = struct.pack("<4i", magic, d, H1, H2, C)
    # Layout: T[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C]
    body = (T.astype("<f4").tobytes() +
            b1.astype("<f4").tobytes() +
            W2.astype("<f4").tobytes() +
            b2.astype("<f4").tobytes() +
            W3.astype("<f4").tobytes() +
            b3.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* Table family with two hidden layers (H1, H2) for label tasks.
 * ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte (class index). */
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
    if (nw < 20 || hd[0] != 0x54424C32) { fprintf(stderr, "bad weights (magic)\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 20 + 4L * ((long)d * 256 * H1 + H1 + (long)H1 * H2 + H2 + (long)H2 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T   = (const float *)(hd + 5);
    const float *b1  = T + (long)d * 256 * H1;
    const float *W2  = b1 + H1;
    const float *b2  = W2 + (long)H1 * H2;
    const float *W3  = b2 + H2;
    const float *b3  = W3 + (long)H2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h1 = malloc(sizeof(float) * H1);
    float *h2 = malloc(sizeof(float) * H2);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // h1 = b1 + sum_j T[j][xi[j]]
        memcpy(h1, b1, sizeof(float) * H1);
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H1;
            for (int u = 0; u < H1; u++) h1[u] += t[u];
        }
        // ReLU
        for (int u = 0; u < H1; u++) if (h1[u] < 0.0f) h1[u] = 0.0f;

        // h2 = b2 + h1 @ W2
        for (int v = 0; v < H2; v++) {
            float sum = b2[v];
            for (int u = 0; u < H1; u++) sum += h1[u] * W2[(long)u * H2 + v];
            h2[v] = sum;
        }
        // ReLU
        for (int v = 0; v < H2; v++) if (h2[v] < 0.0f) h2[v] = 0.0f;

        // logits = b3 + h2 @ W3
        for (int c = 0; c < C; c++) {
            float sum = b3[c];
            for (int v = 0; v < H2; v++) sum += h2[v] * W3[(long)v * C + c];
            logits[c] = sum;
        }

        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```