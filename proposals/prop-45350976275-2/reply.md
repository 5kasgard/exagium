```hypothesis
Family: label-embed (positional byte embeddings + 1-hidden-layer MLP).
Why: The task is 16-byte context → next char (65 classes). label-linear uses 16×256 one-hot (4096 features) with no hidden layer.
A small shared embedding (4 dims per byte) reduces input to 64 dense features, adds a 16-unit ReLU layer (≈5k params total),
captures byte similarities and positional interactions, and should beat linear accuracy at similar inference cost.
Failure: If dev accuracy ≤ 0.3375 or Ir/item ≥ 2× label-linear (≈15k), the family does not improve the trade-off.
```

```python
"""label-embed: positional byte embeddings (4-d) + 1-hidden-layer ReLU MLP (16 units) → 65-class softmax.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only. Writes weights.bin with magic 0x454D4244."""
import json, math, struct, sys

import numpy as np

EMB_DIM = 4
HIDDEN = 16
SEED = 1
LR = 1e-2
L2 = 1e-4
BATCH = 64
EPOCHS = 10


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    # validation split (last 20% in file order) for potential tuning; here we just train on all
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)

    # parameters: embeddings[d, 256, EMB_DIM], W1[d*EMB_DIM, HIDDEN], b1[HIDDEN], W2[HIDDEN, C], b2[C]
    emb = (rng.normal(0, 0.01, (d, 256, EMB_DIM))).astype(np.float32)
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * EMB_DIM)), (d * EMB_DIM, HIDDEN))).astype(np.float32)
    b1 = np.zeros(HIDDEN, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / HIDDEN), (HIDDEN, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    def forward(Xb):
        # Xb: (B, d) uint8
        B = Xb.shape[0]
        # embedding lookup: (B, d, EMB_DIM)
        e = emb[np.arange(d), Xb.T].transpose(1, 0, 2).reshape(B, d * EMB_DIM)
        h = e @ W1 + b1
        h = np.maximum(h, 0, dtype=np.float32)
        logits = h @ W2 + b2
        return logits, h, e

    def loss_grad(logits, yb):
        B = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(B), yb] -= 1.0
        probs /= B
        return probs.astype(np.float32)

    for epoch in range(EPOCHS):
        order = rng.permutation(len(X_train))
        for s in range(0, len(X_train), BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_train[idx], y_train[idx]
            logits, h, e = forward(Xb)
            g = loss_grad(logits, yb)                     # (B, C)
            # grad W2, b2
            gW2 = h.T @ g + L2 * W2
            gb2 = g.sum(0)
            # grad hidden
            gh = g @ W2.T
            gh[h == 0] = 0.0
            # grad W1, b1
            gW1 = e.T @ gh + L2 * W1
            gb1 = gh.sum(0)
            # grad embeddings
            gemb = np.zeros_like(emb)
            gh_reshaped = gh.reshape(len(yb), d, EMB_DIM)
            for j in range(d):
                np.add.at(gemb[j], Xb[:, j], gh_reshaped[:, j, :])
            gemb += L2 * emb

            grads = [gemb, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * gr * gr
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # final validation accuracy (optional, not used for selection)
    if len(X_val):
        logits, _, _ = forward(X_val)
        pred = logits.argmax(1)
        acc = (pred == y_val).mean()
        print(f"val acc: {acc:.4f}", file=sys.stderr)

    # pack weights: magic, d, EMB_DIM, HIDDEN, C
    header = struct.pack("<5i", 0x454D4244, d, EMB_DIM, HIDDEN, C)
    # embeddings: (d, 256, EMB_DIM) -> flatten in order d, 256, EMB_DIM
    emb_bytes = emb.astype("<f4").tobytes()
    W1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    W2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + W1_bytes + b1_bytes + W2_bytes + b2_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format (little-endian):
 *   uint32_t magic = 0x454D4244 ('EMBD')
 *   int32_t d, emb_dim, H, C
 *   float emb[d][256][emb_dim]
 *   float W1[d*emb_dim][H], b1[H]
 *   float W2[H][C], b2[C]
 */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], emb_dim = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || emb_dim != 4 || H != 16 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }
    size_t emb_sz = (size_t)d * 256 * emb_dim;
    size_t W1_sz = (size_t)d * emb_dim * H;
    size_t W2_sz = (size_t)H * C;
    long expected = 20 + 4L * (emb_sz + W1_sz + H + W2_sz + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + emb_sz;
    const float *b1 = W1 + W1_sz;
    const float *W2 = b1 + H;
    const float *b2 = W2 + W2_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *e = malloc(sizeof(float) * d * emb_dim);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // embedding lookup
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((size_t)j * 256 + xi[j]) * emb_dim;
            memcpy(e + j * emb_dim, src, sizeof(float) * emb_dim);
        }
        // hidden layer: ReLU(e @ W1 + b1)
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * emb_dim; v++) sum += e[v] * W1[(size_t)v * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }
        // output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(size_t)u * C + c];
            logits[c] = sum;
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```