```hypothesis
Family: label-embedding-mlp2 – a two-hidden-layer MLP with factorized input embeddings (16 positions × 256 bytes × E=16).
Why: Low-rank factorization of the first layer (embeddings @ W1) reduces parameters and regularizes, while two ReLU layers capture non-linear interactions; embeddings share strength across byte values.
Failure: Dev accuracy ≤ 0.3375 (label-linear) or inference cost >2× label-linear.
```

```python
"""label-embedding-mlp2: two-hidden-layer MLP with factorized input embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports weights as: magic 0x454D4232, d, H1, H2, C; float32 T[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 16          # embedding dimension
H1 = 64         # first hidden layer
H2 = 32         # second hidden layer
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 30

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
    assert d == 16 and C == 65

    rng = np.random.default_rng(SEED)

    # Parameters
    embeddings = [rng.normal(0, 1/math.sqrt(E), size=(256, E)).astype(np.float32) for _ in range(d)]
    W1 = rng.normal(0, math.sqrt(2.0/E), size=(E, H1)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = rng.normal(0, math.sqrt(2.0/H1), size=(H1, H2)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = rng.normal(0, math.sqrt(2.0/H2), size=(H2, C)).astype(np.float32)
    b3 = np.zeros(C, np.float32)

    params = embeddings + [W1, b1, W2, b2, W3, b3]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    n = X.shape[0]
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            B = len(idx)

            # Forward
            emb_sum = np.zeros((B, E), np.float32)
            for j in range(d):
                emb_sum += embeddings[j][Xb[:, j]]
            pre1 = emb_sum @ W1 + b1
            h1 = np.maximum(pre1, 0)
            pre2 = h1 @ W2 + b2
            h2 = np.maximum(pre2, 0)
            logits = h2 @ W3 + b3

            # Loss & softmax gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            dlogits = probs
            dlogits[np.arange(B), yb] -= 1
            dlogits /= B

            # Backward
            dW3 = h2.T @ dlogits + 2*L2*W3
            db3 = dlogits.sum(axis=0)
            dh2 = dlogits @ W3.T
            dh2[pre2 <= 0] = 0

            dW2 = h1.T @ dh2 + 2*L2*W2
            db2 = dh2.sum(axis=0)
            dh1 = dh2 @ W2.T
            dh1[pre1 <= 0] = 0

            dW1 = emb_sum.T @ dh1 + 2*L2*W1
            db1 = dh1.sum(axis=0)
            demb_sum = dh1 @ W1.T

            demb = [np.zeros_like(emb) for emb in embeddings]
            for j in range(d):
                np.add.at(demb[j], Xb[:, j], demb_sum)
                demb[j] += 2*L2*embeddings[j]

            grads = demb + [dW1, db1, dW2, db2, dW3, db3]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9*m[i] + 0.1*gr
                v[i] = 0.999*v[i] + 0.001*gr*gr
                m_hat = m[i] / (1 - 0.9**t)
                v_hat = v[i] / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Precompute T = embeddings @ W1
    T = np.stack([emb @ W1 for emb in embeddings], axis=0).astype(np.float32)  # (d, 256, H1)

    # Write weights
    magic = 0x454D4232  # 'EMB2' little-endian
    header = struct.pack("<5i", magic, d, H1, H2, C)
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
/* label-embedding-mlp2 inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D4232) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (d != 16 || C != 65 || H1 <= 0 || H2 <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H1 + H1 + (long)H1 * H2 + H2 + (long)H2 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * H1;
    const float *W2 = b1 + H1;
    const float *b2 = W2 + (long)H1 * H2;
    const float *W3 = b2 + H2;
    const float *b3 = W3 + (long)H2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc1 = malloc(sizeof(float) * H1);
    float *acc2 = malloc(sizeof(float) * H2);
    float *acc3 = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(acc1, b1, H1 * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H1;
            for (int h = 0; h < H1; h++) acc1[h] += t[h];
        }
        for (int h = 0; h < H1; h++) if (acc1[h] < 0.0f) acc1[h] = 0.0f;

        memcpy(acc2, b2, H2 * sizeof(float));
        for (int h = 0; h < H1; h++) {
            float a = acc1[h];
            if (a != 0.0f) {
                const float *w = W2 + (long)h * H2;
                for (int h2 = 0; h2 < H2; h2++) acc2[h2] += a * w[h2];
            }
        }
        for (int h2 = 0; h2 < H2; h2++) if (acc2[h2] < 0.0f) acc2[h2] = 0.0f;

        memcpy(acc3, b3, C * sizeof(float));
        for (int h2 = 0; h2 < H2; h2++) {
            float a = acc2[h2];
            if (a != 0.0f) {
                const float *w = W3 + (long)h2 * C;
                for (int c = 0; c < C; c++) acc3[c] += a * w[c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc3[c] > acc3[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```