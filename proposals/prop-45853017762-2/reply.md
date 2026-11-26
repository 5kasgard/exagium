```hypothesis
Family: label-shared-mlp. Uses a single shared embedding table for all 16 context positions (instead of per-position tables), followed by one ReLU hidden layer. This drastically reduces parameters (from ~16*256*H to 256*E + 16*E*H) while still capturing interactions between positions via the hidden layer. The shared embedding acts as a strong regularizer, which should help on the small 24k training set. If dev accuracy does not exceed label-linear (0.3375) or if CPU cost per item exceeds label-linear (7.3k Ir/item), the hypothesis fails.
```

```python
"""label-shared-mlp: shared embedding + one ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: 16 context bytes -> shared embedding (256 x E) -> concat (16*E) -> ReLU hidden (H) -> linear (C).
Exported weights: magic 0x53484D4C, d, E, H, C; float32 embed[256][E], W1[16*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen after tuning on last 20% of train.in)
E = 16          # embedding dimension
H = 64          # hidden layer size
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
    assert d == 16

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Xavier initialization
    embed = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E + H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / (H + C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [embed, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = xb.shape[0]

            # Forward
            # embed lookup: (bs, d, E)
            emb = embed[xb]                       # (bs, d, E)
            flat = emb.reshape(bs, d * E)         # (bs, d*E)
            hidden_pre = flat @ W1 + b1           # (bs, H)
            hidden = np.maximum(hidden_pre, 0)    # ReLU
            logits = hidden @ W2 + b2             # (bs, C)

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            # Backward
            # d_logits = probs
            # d_W2 = hidden.T @ d_logits
            # d_b2 = d_logits.sum(0)
            # d_hidden = d_logits @ W2.T
            # d_hidden_pre = d_hidden * (hidden_pre > 0)
            # d_W1 = flat.T @ d_hidden_pre
            # d_b1 = d_hidden_pre.sum(0)
            # d_flat = d_hidden_pre @ W1.T
            # d_emb = d_flat.reshape(bs, d, E)
            # d_embed: scatter add over xb indices

            d_logits = probs.astype(np.float32)
            d_W2 = hidden.T @ d_logits + L2 * W2
            d_b2 = d_logits.sum(0)
            d_hidden = d_logits @ W2.T
            d_hidden_pre = d_hidden * (hidden_pre > 0).astype(np.float32)
            d_W1 = flat.T @ d_hidden_pre + L2 * W1
            d_b1 = d_hidden_pre.sum(0)
            d_flat = d_hidden_pre @ W1.T
            d_emb = d_flat.reshape(bs, d, E)

            # Accumulate embedding gradients
            d_embed = np.zeros_like(embed)
            for j in range(d):
                idxs = xb[:, j]
                np.add.at(d_embed, idxs, d_emb[:, j, :])
            d_embed += L2 * embed

            grads = [d_embed, d_W1, d_b1, d_W2, d_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<5i", 0x53484D4C, d, E, H, C)
    out += embed.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-shared-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x53484D4C, int32 d, E, H, C; float32 embed[256*E], W1[d*E*H], b1[H], W2[H*C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x53484D4C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * (256L * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *embed = (const float *)(hd + 5);
    const float *W1 = embed + 256L * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // hidden = b1
        memcpy(hidden, b1, H * sizeof(float));
        // for each position j, add embed[xi[j]] * W1[j]
        for (int j = 0; j < d; j++) {
            const float *e = embed + (long)xi[j] * E;
            const float *w = W1 + (long)j * E * H;
            for (int h = 0; h < H; h++) {
                float sum = 0.0f;
                for (int ee = 0; ee < E; ee++) sum += e[ee] * w[(long)ee * H + h];
                hidden[h] += sum;
            }
        }
        // ReLU
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        // scores = b2 + hidden * W2
        memcpy(scores, b2, C * sizeof(float));
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            if (hv == 0.0f) continue;
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w[c];
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```