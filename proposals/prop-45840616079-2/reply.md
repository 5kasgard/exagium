```hypothesis
Family: label-embed (embedding + one ReLU layer). Each of the 16 context bytes gets a learned E-dim embedding (shared across positions), concatenated to 16*E features, then a hidden layer of size H with ReLU, then linear to 65 classes. This captures byte similarities (e.g., letters near each other) that one-hot linear cannot, while keeping inference fast: 16*E*H + H*65 FLOPs per item. With E=16, H=64 we expect ~0.36 accuracy (beating label-linear's 0.3375) at ~15k Ir/item. If dev accuracy ≤ 0.34 or Ir/item > 25k, the family does not deliver.
```

```python
"""label-embed: embedding + one ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, numpy only.
Weights format (little-endian):
  int32 magic=0x454D4244, d=16, E, H, C=65
  float32 emb[256*E], W1[(d*E)*H], b1[H], W2[H*C], b2[C]"""
import json, math, struct, sys

import numpy as np

E = 16          # embedding dimension
H = 64          # hidden size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 128
SAMPLES = 600000
MAX_EPOCHS = 50


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
    assert d == 16 and C == 65

    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # parameters
    emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # embed: (B, d, E)
            emb_x = emb[xb]                   # (B, d, E)
            flat = emb_x.reshape(B, d * E)    # (B, d*E)

            # hidden
            hidden = flat @ W1 + b1           # (B, H)
            hidden_relu = np.maximum(hidden, 0)

            # output logits
            logits = hidden_relu @ W2 + b2    # (B, C)

            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_out = (probs / B).astype(np.float32)   # (B, C)

            # backprop
            grad_W2 = hidden_relu.T @ grad_out + L2 * W2
            grad_b2 = grad_out.sum(axis=0)
            grad_hidden = grad_out @ W2.T
            grad_hidden *= (hidden > 0).astype(np.float32)   # ReLU gradient
            grad_W1 = flat.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(axis=0)

            # embedding gradient: scatter-add
            grad_emb = np.zeros_like(emb)
            # grad_hidden shape (B, H), W1 shape (d*E, H) -> grad_flat = grad_hidden @ W1.T (B, d*E)
            grad_flat = grad_hidden @ W1.T                # (B, d*E)
            grad_emb_x = grad_flat.reshape(B, d, E)       # (B, d, E)
            # scatter-add: for each position j, add grad_emb_x[:,j,:] to grad_emb[xb[:,j]]
            for j in range(d):
                np.add.at(grad_emb, xb[:, j], grad_emb_x[:, j])

            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights
    magic = 0x454D4244  # 'EMBD'
    header = struct.pack("<5i", magic, d, E, H, C)
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
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: 16 raw bytes. Output record: 1 byte (class index 0..64).
 * Weights format (little-endian):
 *   int32 magic=0x454D4244, d=16, E, H, C=65
 *   float32 emb[256*E], W1[(d*E)*H], b1[H], W2[H*C], b2[C] */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * (256L * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + 256L * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* hidden = b1 */
        memcpy(hidden, b1, sizeof(float) * H);

        /* accumulate embeddings * W1 */
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * E;
            const float *w = W1 + (long)j * E * H;
            for (int h = 0; h < H; h++) {
                float sum = 0.0f;
                for (int ee = 0; ee < E; ee++) sum += e[ee] * w[(long)ee * H + h];
                hidden[h] += sum;
            }
        }

        /* ReLU */
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        /* logits = hidden * W2 + b2 */
        memcpy(logits, b2, sizeof(float) * C);
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            if (hv == 0.0f) continue;
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) logits[c] += hv * w[c];
        }

        /* argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```