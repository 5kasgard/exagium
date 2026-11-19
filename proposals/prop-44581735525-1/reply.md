```hypothesis
Family: label-mlp2 (two hidden layers of 16 and 8 ReLU units). The optdigits 8x8 images benefit from hierarchical features; a second hidden layer can combine first-layer patterns (strokes) into digit parts, improving over single-layer MLP. Training uses early stopping on the tuning split (last 20% of train) to avoid overfitting. If dev accuracy does not exceed label-mlp16 (0.9770) or cost exceeds label-mlp16 (8685 Ir/item), the family does not justify its complexity.
```

```python
"""Table family with two ReLU hidden layers (H1=16, H2=8) for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic: seed 1, single thread. Uses tuning split (last 20%) for early stopping.
Encoding: ordinal z-scored per byte. Exported as per-byte lookup tables for first layer only.
Weights format: int32 magic 0x54424C32, d, H1, H2, C; float32 T1[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C]."""
import json, math, struct, sys

import numpy as np

H1, H2 = 16, 8
SEED, LR, L2, BATCH, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 200
PATIENCE = 15  # early stopping patience on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X_all.shape[1]

    # split: last 20% for tuning
    n = X_all.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X_all[:split], y_all[:split]
    X_va, y_va = X_all[split:], y_all[split:]

    rng = np.random.default_rng(SEED)

    # encoding on training set only
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X_tr[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X_tr[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)

    def encode(X):
        return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

    Z_tr = encode(X_tr)
    Z_va = encode(X_va)
    n_tr, k = Z_tr.shape

    # parameters: W1[k, H1], b1[H1]; W2[H1, H2], b2[H2]; W3[H2, C], b3[C]
    W1 = (rng.uniform(-1, 1, (k, H1)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
    b3 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(Z):
        a1 = Z @ W1 + b1
        h1 = np.maximum(a1, 0)
        a2 = h1 @ W2 + b2
        h2 = np.maximum(a2, 0)
        out = h2 @ W3 + b3
        return a1, h1, a2, h2, out

    def loss_grad(out, y):
        # softmax cross-entropy gradient
        out = out - out.max(axis=1, keepdims=True)
        exp = np.exp(out)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(y)), y] -= 1.0
        return (probs / len(y)).astype(np.float32)

    best_acc = -1.0
    best_params = None
    patience = 0
    t = 0

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_tr[idx], y_tr[idx]
            a1, h1, a2, h2, out = forward(z)
            g3 = loss_grad(out, yy)
            # backprop
            g_h2 = g3 @ W3.T * (a2 > 0)
            g2 = g_h2
            g_W3 = h2.T @ g3 + L2 * W3
            g_b3 = g3.sum(0)
            g_h1 = g_h2 @ W2.T * (a1 > 0)
            g1 = g_h1
            g_W2 = h1.T @ g_h2 + L2 * W2
            g_b2 = g_h2.sum(0)
            g_W1 = z.T @ g_h1 + L2 * W1
            g_b1 = g_h1.sum(0)
            grads = [g_W1, g_b1, g_W2, g_b2, g_W3, g_b3]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # validation accuracy
        _, _, _, _, out_va = forward(Z_va)
        preds = out_va.argmax(axis=1)
        acc = (preds == y_va).mean()
        if acc > best_acc:
            best_acc = acc
            best_params = [p.copy() for p in params]
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    # use best parameters
    W1, b1, W2, b2, W3, b3 = best_params

    # build lookup tables for first layer: T1[d][256][H1]
    off = 0
    T1_list = []
    for E in enc:
        T1_list.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T1 = np.stack(T1_list)  # d x 256 x H1

    # pack weights
    out_bytes = struct.pack("<5i", 0x54424C32, d, H1, H2, C)
    out_bytes += T1.astype("<f4").tobytes()
    out_bytes += b1.astype("<f4").tobytes()
    out_bytes += W2.astype("<f4").tobytes()
    out_bytes += b2.astype("<f4").tobytes()
    out_bytes += W3.astype("<f4").tobytes()
    out_bytes += b3.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out_bytes)


if __name__ == "__main__":
    main()
```

```c
/* Two-hidden-layer table family (H1, H2 ReLU) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). */
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
    if (nw < 20 || hd[0] != 0x54424C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256 || H1 < 1 || H2 < 1) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H1 + H1 + (long)H1 * H2 + H2 + (long)H2 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T1 = (const float *)(hd + 5);
    const float *b1 = T1 + (long)d * 256 * H1;
    const float *W2 = b1 + H1;
    const float *b2 = W2 + (long)H1 * H2;
    const float *W3 = b2 + H2;
    const float *b3 = W3 + (long)H2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc1 = malloc(sizeof(float) * H1);
    float *acc2 = malloc(sizeof(float) * H2);
    float *acc3 = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(acc1, b1, sizeof(float) * H1);
        for (int j = 0; j < d; j++) {
            const float *t = T1 + ((long)j * 256 + xi[j]) * H1;
            for (int h = 0; h < H1; h++) acc1[h] += t[h];
        }
        for (int h = 0; h < H1; h++) if (acc1[h] < 0.0f) acc1[h] = 0.0f;

        memcpy(acc2, b2, sizeof(float) * H2);
        for (int h1 = 0; h1 < H1; h1++) {
            float a = acc1[h1];
            const float *w = W2 + (long)h1 * H2;
            for (int h2 = 0; h2 < H2; h2++) acc2[h2] += a * w[h2];
        }
        for (int h2 = 0; h2 < H2; h2++) if (acc2[h2] < 0.0f) acc2[h2] = 0.0f;

        memcpy(acc3, b3, sizeof(float) * C);
        for (int h2 = 0; h2 < H2; h2++) {
            float a = acc2[h2];
            const float *w = W3 + (long)h2 * C;
            for (int c = 0; c < C; c++) acc3[c] += a * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc3[c] > acc3[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```