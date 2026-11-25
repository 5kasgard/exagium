```hypothesis
Family: label-emb-mlp. A small MLP with per-position byte embeddings (E=8) and one hidden layer (H=64 ReLU). 
This reduces parameters 6x vs label-linear (45k vs 266k) while capturing positional interactions via the hidden layer. 
Embeddings share statistics across byte values, which suits the character prediction task. 
If dev accuracy does not exceed label-linear's 0.3375, the model fails to justify its complexity.
```

```python
"""label-emb-mlp: per-position byte embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports weights in the same format as label-linear (magic 0x54424C31)."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 8          # embedding dimension per position
H = 64         # hidden layer size
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
    d = X.shape[1]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters (float32)
    embed = rng.normal(0, 1.0 / math.sqrt(E), size=(d, 256, E)).astype(np.float32)
    W1 = rng.uniform(-math.sqrt(6.0 / (d * E)), math.sqrt(6.0 / (d * E)), size=(d * E, H)).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    W2 = rng.uniform(-math.sqrt(6.0 / H), math.sqrt(6.0 / H), size=(H, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = [embed, W1, b1, W2, b2]
    adam_m = [np.zeros_like(p) for p in params]
    adam_v = [np.zeros_like(p) for p in params]

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = 0

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x_batch = X[idx]          # (B, d)
            y_batch = y[idx]          # (B,)
            B = len(idx)

            # Forward pass
            emb_concat = np.empty((B, d * E), dtype=np.float32)
            for j in range(d):
                emb_concat[:, j * E:(j + 1) * E] = embed[j][x_batch[:, j]]
            hidden_pre = emb_concat @ W1 + b1
            hidden = np.maximum(hidden_pre, 0, dtype=np.float32)
            logits = hidden @ W2 + b2

            # Softmax cross-entropy
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # Backward pass
            dlogits = probs
            dlogits[np.arange(B), y_batch] -= 1.0
            dlogits /= B

            dW2 = hidden.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)
            dh = dlogits @ W2.T
            dh[hidden_pre <= 0] = 0.0
            dW1 = emb_concat.T @ dh + L2 * W1
            db1 = dh.sum(axis=0)
            demb_concat = dh @ W1.T  # (B, d*E)

            demb = np.zeros_like(embed)
            for j in range(d):
                np.add.at(demb[j], x_batch[:, j], demb_concat[:, j * E:(j + 1) * E])

            grads = [demb, dW1, db1, dW2, db2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = adam_m[i], adam_v[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1.0 - 0.9 ** t)
                v_hat = v / (1.0 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights in label-linear format
    T = []
    for j in range(d):
        Tj = embed[j] @ W1[j * E:(j + 1) * E]   # (256, H)
        T.append(Tj.astype(np.float32))
    T_stacked = np.stack(T)                     # (d, 256, H)

    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += T_stacked.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
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