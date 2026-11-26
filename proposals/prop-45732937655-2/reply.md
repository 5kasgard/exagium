```hypothesis
Family: label-embed. Uses per-position byte embeddings (D=16) followed by a one-hidden-layer MLP (H=64) to capture sequential interactions more efficiently than one-hot linear. The embedding+MLP has ~86k params vs linear's 266k, and inference cost is similar (lookup + small matmuls). If dev accuracy does not exceed label-linear's 0.3375 at comparable or lower CPU cost, the approach fails.
```

```python
"""Embedding + one-hidden-layer MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Exports weights in the same Table format as label-linear (magic 0x54424C31)."""
import json, math, struct, sys

import numpy as np

D = 16          # embedding dimension per position
H = 64          # hidden layer size (0 for linear on embeddings)
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
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    W = H if H else C

    # parameters
    embeddings = rng.normal(0, 0.01, (d, 256, D)).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(6.0 / (d * D + W)), (d * D, W))).astype(np.float32)
    b1 = np.zeros(W, np.float32)
    if H:
        W2 = (rng.normal(0, math.sqrt(6.0 / (H + C)), (H, C))).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [embeddings, W1, b1, W2, b2]
    else:
        params = [embeddings, W1, b1]

    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            batch_idx = X[idx]          # (batch, d)
            batch_y = y[idx]            # (batch,)
            bs = len(batch_idx)

            # forward: gather embeddings
            batch_emb = np.zeros((bs, d * D), np.float32)
            for j in range(d):
                batch_emb[:, j * D:(j + 1) * D] = embeddings[j, batch_idx[:, j]]

            if H:
                hidden = batch_emb @ W1 + b1
                hidden = np.maximum(hidden, 0, dtype=np.float32)
                logits = hidden @ W2 + b2
            else:
                logits = batch_emb @ W1 + b1

            # softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), batch_y] -= 1.0
            dlogits = (probs / bs).astype(np.float32)

            # backward
            if H:
                dhidden = dlogits @ W2.T
                dhidden *= (hidden > 0).astype(np.float32)
                dW2 = hidden.T @ dlogits
                db2 = dlogits.sum(axis=0)
                dbatch_emb = dhidden @ W1.T
                dW1 = batch_emb.T @ dhidden
                db1 = dhidden.sum(axis=0)
            else:
                dW1 = batch_emb.T @ dlogits
                db1 = dlogits.sum(axis=0)
                dbatch_emb = dlogits @ W1.T

            # embedding gradients (sparse)
            dembeddings = np.zeros_like(embeddings)
            for j in range(d):
                np.add.at(dembeddings[j], batch_idx[:, j], dbatch_emb[:, j * D:(j + 1) * D])

            # L2 regularization
            dW1 += L2 * W1
            if H:
                dW2 += L2 * W2
            dembeddings += L2 * embeddings

            grads = [dembeddings, dW1, db1]
            if H:
                grads.extend([dW2, db2])

            # Adam update
            t += 1
            for i, (param, grad) in enumerate(zip(params, grads)):
                m, v = states[i]
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # fold embeddings and first layer into per-position lookup tables T
    T = []
    for j in range(d):
        T.append(embeddings[j] @ W1[j * D:(j + 1) * D])
    T = np.stack(T)  # (d, 256, W)

    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```