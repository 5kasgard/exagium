```hypothesis
Family: label-emb-mlp (per-feature embeddings + one ReLU hidden layer).
Why: Adult has 14 byte-wide features (mixed ordinal/categorical). Linear models ignore feature interactions; a small embedding table per feature (256×E) followed by a tiny MLP (E·14 → H → 2) captures low-order interactions cheaply. With E=8, H=32 the model has ~30k parameters – fits in L1 cache, so inference should be ~2× label-linear cost while gaining accuracy.
Failure signal: dev accuracy ≤ 0.8524 or instructions/item > 1500.
```

```python
"""label-emb-mlp: per-feature embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed hyper-parameters (tuned on last 20% of train.in beforehand)
E = 8          # embedding dimension per feature
H = 32         # hidden layer size
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
    d, C = X.shape[1], card["classes"]

    # Tuning split = last 20% of train.in (file order)
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    rng = np.random.default_rng(SEED)

    # Embeddings: list of (256, E) float32, one per feature
    emb = [rng.normal(0, 0.01, (256, E)).astype(np.float32) for _ in range(d)]
    # MLP weights
    W1 = rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, math.sqrt(2.0 / H), (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = emb + [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(Xb):
        # Xb: (batch, d) uint8
        batch = Xb.shape[0]
        x = np.empty((batch, d * E), np.float32)
        for j in range(d):
            x[:, j * E:(j + 1) * E] = emb[j][Xb[:, j]]
        h = x @ W1 + b1
        np.maximum(h, 0, out=h)
        logits = h @ W2 + b2
        return x, h, logits

    def loss_grad(logits, yb):
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(yb)), yb] -= 1.0
        probs /= len(yb)
        return probs.astype(np.float32)

    best_acc, best_params = 0.0, None
    t = 0
    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(split)
        for s in range(0, split, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            x, h, logits = forward(xb)
            g = loss_grad(logits, yb)

            # Gradients
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(0)
            gh = g @ W2.T * (h > 0)
            grad_W1 = x.T @ gh + L2 * W1
            grad_b1 = gh.sum(0)
            # Embedding gradients
            grad_emb = [np.zeros((256, E), np.float32) for _ in range(d)]
            for j in range(d):
                vals, cnt = np.unique(xb[:, j], return_counts=True)
                # Accumulate gradient for each embedding row used
                for v, c in zip(vals, cnt):
                    rows = np.where(xb[:, j] == v)[0]
                    grad_emb[j][v] = gh[rows, j * E:(j + 1) * E].sum(0) + L2 * emb[j][v]

            grads = grad_emb + [grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v_ = state[i]
                m *= 0.9; m += 0.1 * gr
                v_ *= 0.999; v_ += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8)

        # Validation accuracy
        _, _, logits_va = forward(X_va)
        acc = (logits_va.argmax(1) == y_va).mean()
        if acc > best_acc:
            best_acc = acc
            best_params = [p.copy() for p in params]

    # Retrain on full training set using best hyper-params (same code, no validation)
    # We simply continue training from best_params on full data for a few epochs.
    params = best_params
    emb, W1, b1, W2, b2 = params[:d], params[d], params[d+1], params[d+2], params[d+3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(min(5, math.ceil(SAMPLES / n))):  # few epochs on full data
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            x, h, logits = forward(xb)
            g = loss_grad(logits, yb)
            grad_W2 = h.T @ g + L2 * W2
            grad_b2 = g.sum(0)
            gh = g @ W2.T * (h > 0)
            grad_W1 = x.T @ gh + L2 * W1
            grad_b1 = gh.sum(0)
            grad_emb = [np.zeros((256, E), np.float32) for _ in range(d)]
            for j in range(d):
                for v in range(256):
                    rows = np.where(xb[:, j] == v)[0]
                    if rows.size:
                        grad_emb[j][v] = gh[rows, j * E:(j + 1) * E].sum(0) + L2 * emb[j][v]
            grads = grad_emb + [grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v_ = state[i]
                m *= 0.9; m += 0.1 * gr
                v_ *= 0.999; v_ += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8)

    # Export weights.bin
    # Magic 0x454D424C ("LMBE" little-endian), d, E, H, C
    magic = 0x454D424C
    out = struct.pack("<5i", magic, d, E, H, C)
    for j in range(d):
        out += emb[j].astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (one per feature). Output: 1 byte class index. */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d < 1 || E < 1 || H < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* Concatenate embeddings */
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int j = 0; j < d; j++) {
                const float *e = emb + ((long)j * 256 + xi[j]) * E;
                const float *w = W1 + ((long)j * E + 0) * H + h;
                for (int ee = 0; ee < E; ee++) sum += e[ee] * w[ee * H];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        /* Output layer */
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```