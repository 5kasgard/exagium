```hypothesis
Family: label-emb-mlp (positional embeddings + one ReLU layer).
Why: Character-level language modeling benefits from shared byte embeddings across positions (capturing similar characters) and a non-linear hidden layer to model interactions. The 16-byte context is short enough for a compact embedding (32 dims) and hidden layer (64 units) to fit in cache, giving better accuracy than linear at similar latency.
Failure signal: dev accuracy ≤ 0.3375 (label-linear) or inference > 2× label-linear ns/item.
```

```python
"""label-emb-mlp: positional byte embeddings + one ReLU layer.
train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy.
Exports: magic 0x454D424D, d=16, D=32, H=64, C=65;
float32 arrays: E[256*D], W1[(d*D)*H], b1[H], W2[H*C], b2[C]."""
import json, math, struct, sys

import numpy as np

D = 32          # embedding dimension
H = 64          # hidden units
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
    d = X.shape[1]          # 16
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # parameters
    E = (rng.normal(0, 0.01, (256, D))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * D)), (d * D, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [E, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            # forward
            emb = E[xb]                     # (bs, d, D)
            flat = emb.reshape(bs, d * D)   # (bs, d*D)
            h_pre = flat @ W1 + b1          # (bs, H)
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2            # (bs, C)

            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            # backward
            gW2 = h.T @ probs + L2 * W2
            gb2 = probs.sum(axis=0)
            gh = probs @ W2.T
            gh[h_pre <= 0] = 0
            gW1 = flat.T @ gh + L2 * W1
            gb1 = gh.sum(axis=0)
            gflat = gh @ W1.T                # (bs, d*D)
            gE = np.zeros_like(E)
            np.add.at(gE, xb.reshape(-1), gflat.reshape(-1, D))

            grads = [gE, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights
    magic = 0x454D424D  # 'EMBM'
    out = struct.pack("<5i", magic, d, D, H, C)
    out += E.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte class index. */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || D != 32 || H != 64 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }
    size_t expected = 20
        + 256L * D * 4
        + (long)d * D * H * 4
        + H * 4
        + (long)H * C * 4
        + C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E = (const float *)(hd + 5);
    const float *W1 = E + 256L * D;
    const float *b1 = W1 + (long)d * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h_pre = malloc(H * sizeof(float));
    float *h = malloc(H * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* embed and flatten */
        for (int u = 0; u < H; u++) h_pre[u] = b1[u];
        for (int j = 0; j < d; j++) {
            const float *e = E + (long)xi[j] * D;
            const float *w = W1 + (long)j * D * H;
            for (int u = 0; u < H; u++) {
                float sum = 0.0f;
                for (int v = 0; v < D; v++) sum += e[v] * w[(long)v * H + u];
                h_pre[u] += sum;
            }
        }
        /* ReLU */
        for (int u = 0; u < H; u++) h[u] = h_pre[u] > 0.0f ? h_pre[u] : 0.0f;

        /* output layer */
        for (int c = 0; c < C; c++) logits[c] = b2[c];
        for (int u = 0; u < H; u++) {
            float hu = h[u];
            if (hu == 0.0f) continue;
            const float *w = W2 + (long)u * C;
            for (int c = 0; c < C; c++) logits[c] += hu * w[c];
        }

        /* argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```