```hypothesis
The label-emb family learns a position-specific embedding for each of the 16 context bytes (16×256×D), sums them into a dense vector, and passes it through a single ReLU layer (D→H→65). This captures pairwise interactions between context positions that the linear model misses, while using far fewer parameters (≈70k vs 266k) and fewer FLOPs per inference (≈7k vs 266k). On tinyshakespeare's 16-byte context, local character interactions are predictive. If dev accuracy does not exceed label-linear's 0.3375, the family fails.
```
```python
"""label-emb: position-specific byte embeddings + one ReLU layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

D = 32          # embedding dimension
H = 64          # hidden layer size
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 20


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = b"".join(records(sys.argv[2]))
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]
    assert d == 16 and C == 65

    rng = np.random.default_rng(SEED)

    # parameters
    scale = 1.0 / math.sqrt(D)
    emb = rng.normal(0, scale, size=(d, 256, D)).astype(np.float32)
    W1 = rng.normal(0, math.sqrt(2.0 / D), size=(D, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, math.sqrt(2.0 / H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            B = len(idx)

            # forward
            h = np.zeros((B, D), np.float32)
            for j in range(d):
                h += emb[j, xb[:, j]]
            h1 = h @ W1 + b1
            h1_relu = np.maximum(h1, 0)
            logits = h1_relu @ W2 + b2

            # softmax cross-entropy gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B

            # backward
            grad_W2 = h1_relu.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)
            grad_h1_relu = grad_logits @ W2.T
            grad_h1 = grad_h1_relu * (h1 > 0)
            grad_W1 = h.T @ grad_h1
            grad_b1 = grad_h1.sum(axis=0)
            grad_h = grad_h1 @ W1.T

            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_h)

            # L2 regularization
            grad_W2 += L2 * W2
            grad_W1 += L2 * W1
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # write weights.bin
    magic = 0x454D424C  # 'EMBL' little-endian
    header = struct.pack("<5i", magic, d, D, H, C)
    out = header
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte (class index 0..64). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned char *slurp(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(3); }
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*len ? *len : 1);
    if (fread(buf, 1, *len, f) != (size_t)*len) {
        fprintf(stderr, "short read %s\n", path);
        exit(3);
    }
    fclose(f);
    return buf;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = (uint32_t)buf[*pos] |
                 (uint32_t)buf[*pos + 1] << 8 |
                 (uint32_t)buf[*pos + 2] << 16 |
                 (uint32_t)buf[*pos + 3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *rec = buf + *pos + 4;
    *pos += 4 + k;
    return rec;
}

static void put_rec(FILE *f, const unsigned char *rec, uint32_t k) {
    unsigned char hdr[4] = {k & 255, (k >> 8) & 255, (k >> 16) & 255, (k >> 24) & 255};
    fwrite(hdr, 1, 4, f);
    fwrite(rec, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: infer weights.bin in.rec N\n");
        return 2;
    }
    long nw, nx = 0;
    long N = atol(argv[3]);
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);

    if (nw < 20 || hd[0] != 0x454D424C) {  // magic 'EMBL'
        fprintf(stderr, "bad weights\n");
        return 3;
    }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H <= 0) {
        fprintf(stderr, "bad dimensions\n");
        return 3;
    }

    long expected_floats = (long)d * 256 * D + (long)D * H + H + (long)H * C + C;
    if (nw != 20 + 4 * expected_floats) {
        fprintf(stderr, "bad weights size\n");
        return 3;
    }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * D;
    const float *b1 = W1 + (long)D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    long pos = 0;

    float *h = malloc(sizeof(float) * D);
    float *h1 = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k;
        const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) {
            fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d);
            return 3;
        }

        memset(h, 0, sizeof(float) * D);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            for (int u = 0; u < D; u++) h[u] += e[u];
        }

        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < D; v++) sum += h[v] * W1[(long)v * H + u];
            h1[u] = sum > 0.0f ? sum : 0.0f;
        }

        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h1[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(h);
    free(h1);
    free(logits);
    if (x) free(x);
    return 0;
}
```