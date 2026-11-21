```hypothesis
A two-hidden-layer MLP (64→64 ReLU units) with per-position byte embeddings should capture higher-order n-gram interactions that a single linear or single-hidden-layer model misses. The tinyshakespeare task benefits from modeling local character dependencies; two hidden layers allow compositional features (e.g., "th" → "e") while staying small enough to train deterministically on 24k examples. If dev accuracy does not exceed label-linear's 0.3375, the extra capacity is not justified for this data scale.
```

```python
"""Two-hidden-layer table family (64→64 ReLU) for fixed-byte label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x4D4C5032, d, H1, H2, C; float32 T1[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C]."""
import json, math, struct, sys

import numpy as np

H1, H2 = 64, 64
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
    d = X.shape[1]
    rng = np.random.default_rng(SEED)

    # One-hot encoding per position (categorical)
    enc = []
    for j in range(d):
        vals = np.unique(X[:, j])
        E = np.zeros((256, len(vals)), np.float32)
        E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)

    # Input dimension = sum of unique values per position (≤ d*256)
    k = sum(E.shape[1] for E in enc)
    # Build full design matrix Z (n, k)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n = Z.shape[0]

    # Parameters: W1 (k, H1), b1 (H1), W2 (H1, H2), b2 (H2), W3 (H2, C), b3 (C)
    W1 = (rng.uniform(-1, 1, (k, H1)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
    b3 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
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
            a1 = z @ W1 + b1
            h1 = np.maximum(a1, 0)
            a2 = h1 @ W2 + b2
            h2 = np.maximum(a2, 0)
            logits = h2 @ W3 + b3
            g3 = softmax_grad(logits, yy)                 # (B, C)
            # Backward
            grad_W3 = h2.T @ g3 + L2 * W3
            grad_b3 = g3.sum(0)
            gh2 = g3 @ W3.T * (a2 > 0)
            grad_W2 = h1.T @ gh2 + L2 * W2
            grad_b2 = gh2.sum(0)
            gh1 = gh2 @ W2.T * (a1 > 0)
            grad_W1 = z.T @ gh1 + L2 * W1
            grad_b1 = gh1.sum(0)
            grads = [grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3]
            t += 1
            for p, gr, (m, v) in zip(params, grads, state):
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Fold first layer into per-position lookup tables T1[d][256][H1]
    T1_list = []
    off = 0
    for E in enc:
        T1_list.append((E @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T1 = np.stack(T1_list)  # (d, 256, H1)

    # Write weights.bin
    magic = 0x4D4C5032  # "MPL2"
    hdr = struct.pack("<4i", magic, d, H1, H2)
    # Note: C is known from card (65), but we store it implicitly; infer.c reads it from card.
    # We'll write C as well for validation.
    hdr += struct.pack("<i", C)
    out = bytearray(hdr)
    out += T1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    out += W3.astype("<f4").tobytes()
    out += b3.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Two-hidden-layer table family (64→64 ReLU) for label tasks.
 * ./infer weights.bin in.rec N
 * Weights format (little-endian):
 *   int32 magic=0x4D4C5032, d, H1, H2, C
 *   float32 T1[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C] */
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
    if (nw < 20 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256 || H1 < 1 || H2 < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L
        + 4L * d * 256 * H1
        + 4L * H1
        + 4L * H1 * H2
        + 4L * H2
        + 4L * H2 * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *T1 = (const float *)(hd + 5);
    const float *b1 = T1 + (long)d * 256 * H1;
    const float *W2 = b1 + H1;
    const float *b2 = W2 + (long)H1 * H2;
    const float *W3 = b2 + H2;
    const float *b3 = W3 + (long)H2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc1 = malloc(sizeof(float) * H1);
    float *acc2 = malloc(sizeof(float) * H2);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Layer 1: lookup and sum
        for (int u = 0; u < H1; u++) acc1[u] = b1[u];
        for (int j = 0; j < d; j++) {
            const float *t = T1 + ((long)j * 256 + xi[j]) * H1;
            for (int u = 0; u < H1; u++) acc1[u] += t[u];
        }
        // ReLU
        for (int u = 0; u < H1; u++) if (acc1[u] < 0.0f) acc1[u] = 0.0f;

        // Layer 2: acc1 (H1) -> acc2 (H2)
        for (int v = 0; v < H2; v++) acc2[v] = b2[v];
        for (int u = 0; u < H1; u++) {
            float hu = acc1[u];
            if (hu == 0.0f) continue;
            const float *w = W2 + (long)u * H2;
            for (int v = 0; v < H2; v++) acc2[v] += hu * w[v];
        }
        // ReLU
        for (int v = 0; v < H2; v++) if (acc2[v] < 0.0f) acc2[v] = 0.0f;

        // Output layer: acc2 (H2) -> scores (C)
        for (int c = 0; c < C; c++) scores[c] = b3[c];
        for (int v = 0; v < H2; v++) {
            float hv = acc2[v];
            if (hv == 0.0f) continue;
            const float *w = W3 + (long)v * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```