```hypothesis
We propose "label-embed-mlp": per-position byte embeddings (E=4..8) followed by one ReLU hidden layer (H=32..128). This captures local character interactions better than one-hot linear models while keeping inference fast via small dense matrix multiplies. We expect dev accuracy >38% (beating label-linear's 33.7% and approaching label-gbdt's 39.2%) at <15k Ir/item. If accuracy stays ≤34% or Ir/item exceeds 20k, the approach fails.
```

```python
"""label-embed-mlp: per-position embeddings + one ReLU layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, EPOCHS = 3e-3, 1e-4, 64, 30
E_GRID = [4, 8]
H_GRID = [32, 64, 128]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, X_val, y_val, E, H, rng, epochs=EPOCHS):
    n, d = X.shape
    C = 65
    # embeddings: (d, 256, E)
    emb = (rng.normal(0, 1.0 / math.sqrt(E), (d, 256, E))).astype(np.float32)
    k = d * E
    W1 = (rng.normal(0, math.sqrt(2.0 / k), (k, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(Xb):
        # Xb: (B, d) uint8
        B = Xb.shape[0]
        # gather embeddings
        flat = np.empty((B, k), np.float32)
        for j in range(d):
            flat[:, j*E:(j+1)*E] = emb[j, Xb[:, j]]
        h = np.maximum(flat @ W1 + b1, 0)
        logits = h @ W2 + b2
        return flat, h, logits

    def loss_grad(logits, yb):
        B = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(B), yb] -= 1.0
        return probs / B

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            flat, h, logits = forward(Xb)
            g = loss_grad(logits, yb)                 # (B, C)
            gh = g @ W2.T * (h > 0)                   # (B, H)
            gflat = gh @ W1.T                          # (B, k)
            # gradients
            gW2 = h.T @ g + L2 * W2
            gb2 = g.sum(0)
            gW1 = flat.T @ gh + L2 * W1
            gb1 = gh.sum(0)
            # embedding gradients
            gemb = np.zeros_like(emb)
            for j in range(d):
                gj = gflat[:, j*E:(j+1)*E]
                np.add.at(gemb[j], Xb[:, j], gj)
            gemb += L2 * emb
            grads = [gemb, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)

    # validation accuracy
    flat, h, logits = forward(X_val)
    pred = logits.argmax(1)
    acc = (pred == y_val).mean()
    return acc, params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    tune_size = n // 5
    train_size = n - tune_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    best_acc, best_params, best_E, best_H = -1, None, None, None
    for E in E_GRID:
        for H in H_GRID:
            rng = np.random.default_rng(SEED)
            acc, params = train_model(X_train, y_train, X_val, y_val, E, H, rng)
            if acc > best_acc:
                best_acc, best_params, best_E, best_H = acc, params, E, H
    # retrain on full data
    rng = np.random.default_rng(SEED)
    _, final_params = train_model(X, y, X_val[:1], y_val[:1], best_E, best_H, rng, epochs=EPOCHS)
    emb, W1, b1, W2, b2 = final_params
    # pack weights: magic, d, E, H, C, then embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]
    magic = 0x454D4244  # "EMBD"
    header = struct.pack("<5i", magic, d, best_E, best_H, C)
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
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E < 1 || E > 8 || H < 1 || H > 256) { fprintf(stderr, "bad header\n"); return 3; }
    long k = d * E;
    long expected = 20
        + 4L * d * 256 * E
        + 4L * k * H
        + 4L * H
        + 4L * H * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + d * 256 * E;
    const float *b1 = W1 + k * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * k);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        /* gather embeddings */
        for (int j = 0; j < d; j++) {
            const float *e = emb + (j * 256 + xi[j]) * E;
            memcpy(flat + j * E, e, E * sizeof(float));
        }
        /* hidden layer */
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int f = 0; f < k; f++) sum += flat[f] * W1[f * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        /* output layer */
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[h * C + c];
            logits[c] = sum;
        }
        /* argmax */
        best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```