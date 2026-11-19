```hypothesis
We propose a two-hidden-layer MLP with per-position byte embeddings (family "label-mlp2").
The first hidden layer has 32 units, the second 16. This adds nonlinear capacity over label-linear
while keeping inference cost low (only two small matrix multiplies). It should beat label-linear's
0.3375 dev accuracy because the extra layer can model interactions between context bytes.
If dev accuracy does not exceed 0.3375, or if inference cost exceeds 2× label-linear, the hypothesis fails.
```
```python
"""Two-hidden-layer MLP with per-position byte embeddings (label-mlp2).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x4D4C5032, d, H1, H2, C; float32 T1[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C]."""
import json, math, struct, sys

import numpy as np

H1, H2 = 32, 16
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

    # Build per-position one-hot encoders (only for values seen in training)
    enc = []
    for j in range(d):
        vals = np.unique(X[:, j])
        E = np.zeros((256, len(vals)), dtype=np.float32)
        E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)

    # Concatenated one-hot matrix Z (n x K)
    K = sum(E.shape[1] for E in enc)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n, K = Z.shape

    # Initialize weights
    W1 = (rng.uniform(-1, 1, (K, H1)) * math.sqrt(6.0 / K)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
    b3 = np.zeros(C, np.float32)

    params = [W1, b1, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]

            # Forward
            h1_pre = z @ W1 + b1
            h1 = np.maximum(h1_pre, 0)
            h2_pre = h1 @ W2 + b2
            h2 = np.maximum(h2_pre, 0)
            logits = h2 @ W3 + b3

            # Backward
            g = softmax_grad(logits, yy)
            dW3 = h2.T @ g + L2 * W3
            db3 = g.sum(0)
            dh2 = g @ W3.T
            dh2_pre = dh2 * (h2_pre > 0)
            dW2 = h1.T @ dh2_pre + L2 * W2
            db2 = dh2_pre.sum(0)
            dh1 = dh2_pre @ W2.T
            dh1_pre = dh1 * (h1_pre > 0)
            dW1 = z.T @ dh1_pre + L2 * W1
            db1 = dh1_pre.sum(0)

            grads = [dW1, db1, dW2, db2, dW3, db3]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Build per-position lookup tables T1 (256 x H1 each)
    T1_list = []
    off = 0
    for E in enc:
        T1_list.append((E @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T1 = np.stack(T1_list)  # (d, 256, H1)

    # Write weights.bin
    magic = 0x4D4C5032  # "MLP2"
    header = struct.pack("<5i", magic, d, H1, H2, C)
    out = bytearray()
    out += header
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
/* label-mlp2 inference: ./infer weights.bin in.rec N
 * Input: 16-byte records. Output: one-byte class index per record.
 * Weights format: magic 0x4D4C5032, d, H1, H2, C (int32);
 * float32 T1[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C]. */
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
    if (nw < 20 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 20 + 4L * ((long)d * 256 * H1 + H1 + (long)H1 * H2 + H2 + (long)H2 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T1 = (const float *)(hd + 5);
    const float *b1 = T1 + (long)d * 256 * H1;
    const float *W2 = b1 + H1;
    const float *b2 = W2 + (long)H1 * H2;
    const float *W3 = b2 + H2;
    const float *b3 = W3 + (long)H2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h1 = malloc(sizeof(float) * H1);
    float *h2 = malloc(sizeof(float) * H2);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // First hidden layer: sum embeddings + bias, then ReLU
        for (int u = 0; u < H1; u++) h1[u] = b1[u];
        for (int j = 0; j < d; j++) {
            const float *t = T1 + ((long)j * 256 + xi[j]) * H1;
            for (int u = 0; u < H1; u++) h1[u] += t[u];
        }
        for (int u = 0; u < H1; u++) if (h1[u] < 0.0f) h1[u] = 0.0f;

        // Second hidden layer: h1 * W2 + b2, then ReLU
        for (int v = 0; v < H2; v++) h2[v] = b2[v];
        for (int u = 0; u < H1; u++) {
            float hu = h1[u];
            if (hu == 0.0f) continue;
            const float *w = W2 + (long)u * H2;
            for (int v = 0; v < H2; v++) h2[v] += hu * w[v];
        }
        for (int v = 0; v < H2; v++) if (h2[v] < 0.0f) h2[v] = 0.0f;

        // Output layer: h2 * W3 + b3
        for (int c = 0; c < C; c++) scores[c] = b3[c];
        for (int v = 0; v < H2; v++) {
            float hv = h2[v];
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