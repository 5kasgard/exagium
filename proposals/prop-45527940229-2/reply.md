```hypothesis
Family: label-mlp-emb (per-position byte embeddings + one ReLU hidden layer).
The 16-byte context benefits from learned embeddings per position, capturing positional character statistics better than one-hot linear.
A hidden size of 64 should improve accuracy over label-linear (0.3375) while keeping inference cost moderate (precomputed embedding sums).
If dev accuracy ≤ 0.34 or Ir/item > 15000, the model does not justify its complexity.
```

```python
import json, math, struct, sys

import numpy as np

H = 64
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4 : i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]  # 16
    C = card["classes"]  # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Initialize parameters
    embed = []
    for _ in range(d):
        limit = math.sqrt(6.0 / (256 + H))
        e = rng.uniform(-limit, limit, (256, H)).astype(np.float32)
        embed.append(e)
    h_bias = np.zeros(H, dtype=np.float32)
    limit2 = math.sqrt(6.0 / (H + C))
    W2 = rng.uniform(-limit2, limit2, (H, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)

    params = embed + [h_bias, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s : s + BATCH]
            xb = X[idx]  # (B, d)
            yb = y[idx]  # (B,)
            B = xb.shape[0]

            # Forward
            hidden_batch = np.broadcast_to(h_bias, (B, H)).copy()
            for j in range(d):
                hidden_batch += embed[j][xb[:, j]]
            hidden_relu = np.maximum(hidden_batch, 0)
            logits = hidden_relu @ W2 + b2
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # Backward
            dlogits = probs
            dlogits[np.arange(B), yb] -= 1.0
            dlogits /= B
            dW2 = hidden_relu.T @ dlogits + L2 * W2
            db2 = dlogits.sum(axis=0)
            dhidden_relu = dlogits @ W2.T
            dhidden = dhidden_relu * (hidden_batch > 0)
            dh_bias = dhidden.sum(axis=0)
            dembed = []
            for j in range(d):
                onehot = np.zeros((B, 256), dtype=np.float32)
                onehot[np.arange(B), xb[:, j]] = 1.0
                de = onehot.T @ dhidden + L2 * embed[j]
                dembed.append(de)

            grads = dembed + [dh_bias, dW2, db2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x454D4244  # 'DBME'
    out = struct.pack("<4i", magic, d, H, C)
    for j in range(d):
        out += embed[j].astype("<f4").tobytes()
    out += h_bias.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *embed = (const float *)(hd + 4);
    const float *h_bias = embed + (long)d * 256 * H;
    const float *W2 = h_bias + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(hidden, h_bias, sizeof(float) * H);
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) hidden[h] += e[h];
        }
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```