```hypothesis
Family: label-embed-mlp. Learns per-position byte embeddings (dim 8) followed by a ReLU layer (64 units) and a linear output. The first layer is factorized as embeddings * weights, allowing precomputation of per-position lookup tables of size H at export. This captures character similarities and pairwise interactions cheaply. Should beat label-linear (0.3375) by modeling non-linear context dependencies. If dev accuracy <= 0.3375 or inference cost > 2x label-linear, the approach fails.
```

```python
"""label-embed-mlp: per-position embeddings + one ReLU layer + linear output.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hardcoded hyperparameters (chosen on tuning split)
D = 8          # embedding dimension
H = 64         # hidden layer size
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
MAGIC = 0x454D4244  # "EMBD" little-endian

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

    # Initialize parameters
    embeddings = [rng.normal(0, 0.01, (256, D)).astype(np.float32) for _ in range(d)]
    W1 = (rng.uniform(-1, 1, (d * D, H)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = embeddings + [W1, b1, W2, b2]
    param_names = [f"E{j}" for j in range(d)] + ["W1", "b1", "W2", "b2"]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # Forward: gather embeddings
            Z = np.empty((bs, d * D), np.float32)
            for j in range(d):
                Z[:, j*D:(j+1)*D] = embeddings[j][xb[:, j]]

            hidden_pre = Z @ W1 + b1
            hidden = np.maximum(hidden_pre, 0, dtype=np.float32)
            logits = hidden @ W2 + b2

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            dlogits = (probs / bs).astype(np.float32)

            # Backward
            dW2 = hidden.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)
            dhidden = dlogits @ W2.T
            dhidden_pre = dhidden * (hidden_pre > 0)
            dW1 = Z.T @ dhidden_pre + L2 * W1
            db1 = dhidden_pre.sum(axis=0)
            dZ = dhidden_pre @ W1.T

            # Embedding gradients
            dembeddings = []
            for j in range(d):
                dE = np.zeros_like(embeddings[j])
                np.add.at(dE, xb[:, j], dZ[:, j*D:(j+1)*D])
                dembeddings.append(dE)

            grads = dembeddings + [dW1, db1, dW2, db2]

            # Adam update
            t += 1
            for p, gr, (m, v) in zip(params, grads, state):
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export: precompute per-position lookup tables T[j][byte] = E[j][byte] @ W1_slice
    T = np.empty((d, 256, H), np.float32)
    for j in range(d):
        T[j] = embeddings[j] @ W1[j*D:(j+1)*D, :]

    # Pack weights: magic, d, H, C; then T (d*256*H), b1 (H), W2 (H*C), b2 (C)
    header = struct.pack("<4i", MAGIC, d, H, C)
    body = (T.astype("<f4").tobytes() +
            b1.astype("<f4").tobytes() +
            W2.astype("<f4").tobytes() +
            b2.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (d != 16 || C != 65 || H <= 0 || H > 1024) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(acc, b1, H * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *tj = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += tj[h];
        }
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;

        memcpy(scores, b2, C * sizeof(float));
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            const float *w2h = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += ah * w2h[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```