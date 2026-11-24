```hypothesis
Propose label-emb: an embedding-based neural net that learns a dense vector per feature value (14×256×K), sums them, then applies one ReLU hidden layer (K→H) and a linear output (H→2). This captures nonlinear feature interactions better than label-linear while avoiding the high-dimensional one-hot of label-mlp16. With K=16, H=16 it should exceed label-linear's 0.8524 dev accuracy at similar CPU cost; if dev accuracy ≤0.8524 the hypothesis fails.
```

```python
"""label-emb: embedding sum + 1-hidden-layer ReLU net for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weights format: magic 0x454D424C, d, K, H, C; float32 E[d][256][K], W1[K][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

K, H = 16, 16          # embedding dim, hidden units
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
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # embeddings: d x 256 x K
    E = (rng.normal(0, 1.0 / math.sqrt(K), (d, 256, K))).astype(np.float32)
    # hidden layer
    W1 = (rng.normal(0, math.sqrt(2.0 / K), (K, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # output layer
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [E, W1, b1, W2, b2]
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

            # forward: sum embeddings
            emb_sum = np.zeros((bs, K), np.float32)
            for j in range(d):
                emb_sum += E[j, xb[:, j]]

            # hidden
            hidden = emb_sum @ W1 + b1
            hidden_relu = np.maximum(hidden, 0)
            logits = hidden_relu @ W2 + b2

            # softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            # backward
            grad_W2 = hidden_relu.T @ probs + L2 * W2
            grad_b2 = probs.sum(axis=0)
            grad_hidden_relu = probs @ W2.T
            grad_hidden = grad_hidden_relu * (hidden > 0)
            grad_W1 = emb_sum.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(axis=0)
            grad_emb_sum = grad_hidden @ W1.T

            # scatter grad_emb_sum into E gradients
            grad_E = np.zeros_like(E)
            for j in range(d):
                np.add.at(grad_E[j], xb[:, j], grad_emb_sum)

            grads = [grad_E, grad_W1, grad_b1, grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights
    magic = 0x454D424C  # 'LBME'
    out = struct.pack("<5i", magic, d, K, H, C)
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
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index (0/1).
 * Weights format: magic 0x454D424C, d, K, H, C; float32 E[d][256][K], W1[K][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], H = hd[3], C = hd[4];
    if (C != 2 || d != 14 || K <= 0 || H <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * K + (long)K * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E = (const float *)(hd + 5);
    const float *W1 = E + (long)d * 256 * K;
    const float *b1 = W1 + (long)K * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * K);
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(emb_sum, 0, K * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = E + ((long)j * 256 + xi[j]) * K;
            for (int kk = 0; kk < K; kk++) emb_sum[kk] += e[kk];
        }

        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int kk = 0; kk < K; kk++) sum += emb_sum[kk] * W1[(long)kk * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            scores[c] = sum;
        }

        unsigned char best = scores[1] > scores[0] ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```