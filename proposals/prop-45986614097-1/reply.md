```hypothesis
Propose a one-hidden-layer MLP family (label-mlp) with ReLU and per-pixel lookup tables for the first layer. Hidden size is tuned on the last 20% of train.in over {8,16,32,64}. Optdigits is known to benefit from mild nonlinearity; label-linear (96.1%) is linear, while label-mlp16 (97.7%) shows a hidden layer helps. A well-tuned small hidden layer should beat 96.1% at similar inference cost (label-mlp16: 8685 Ir/item). If dev accuracy ≤ 96.5%, the added complexity is not justified.
```

```python
"""label-mlp: one-hidden-layer MLP with per-pixel lookup tables, tuned on the tuning split.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
H_CHOICES = [8, 16, 32, 64]  # hidden sizes to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def make_encoding(X, types):
    """Return list of encoding matrices E_j (256 x dim_j) and total dim k."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])  # (256,1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def encode_batch(X, enc):
    """X (n, d) -> Z (n, k) by concatenating E_j[X[:,j]]."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def train_model(X, y, enc, H, rng):
    """Train on given data, return params (W1,b1,W2,b2)."""
    Z = encode_batch(X, enc)
    n, k = Z.shape
    C = 10
    # init
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # forward
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            # softmax grad
            logits -= logits.max(axis=1, keepdims=True)
            P = np.exp(logits); P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            g = (P / len(yy)).astype(np.float32)
            # backprop
            gh = (g @ W2.T) * (a > 0)
            grads = [
                z.T @ gh + L2 * W1,
                gh.sum(0),
                h.T @ g + L2 * W2,
                g.sum(0)
            ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params


def eval_model(X, y, enc, params):
    W1, b1, W2, b2 = params
    Z = encode_batch(X, enc)
    a = Z @ W1 + b1
    h = np.maximum(a, 0)
    logits = h @ W2 + b2
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def export_weights(enc, params, d, H, C):
    """Pack weights into bytes: magic, d, H, C, T[d][256][H], b1[H], W2[H][C], b2[C]."""
    W1, b1, W2, b2 = params
    # Build T: for each pixel j, T_j = E_j @ W1_j where W1_j is slice of W1 for that pixel's encoding dim.
    T_list = []
    off = 0
    for E in enc:
        dim = E.shape[1]
        T_j = (E.astype(np.float32) @ W1[off:off + dim]).astype(np.float32)  # (256, H)
        T_list.append(T_j)
        off += dim
    T = np.stack(T_list)  # (d, 256, H)
    magic = 0x4D4C5031  # "MLP1"
    out = struct.pack("<4i", magic, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    return out


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # tuning split: last 20% in file order
    n_tune = int(0.2 * n)
    train_idx = np.arange(n - n_tune)
    tune_idx = np.arange(n - n_tune, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    enc = make_encoding(X, types)  # use all data for encoding stats (as in label-linear)
    best_H, best_acc = None, -1.0
    for H in H_CHOICES:
        rng = np.random.default_rng(SEED + H)
        params = train_model(X_train, y_train, enc, H, rng)
        acc = eval_model(X_tune, y_tune, enc, params)
        if acc > best_acc:
            best_acc, best_H = acc, H
    # retrain on full data with best_H
    rng = np.random.default_rng(SEED)
    params = train_model(X, y, enc, best_H, rng)
    weights = export_weights(enc, params, d, best_H, C)
    open(sys.argv[4], "wb").write(weights)


if __name__ == "__main__":
    main()
```

```c
/* label-mlp inference: ./infer weights.bin in.rec N
 * Weights format: int32 magic(0x4D4C5031), d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C].
 * Input record: d bytes (pixel counts). Output record: 1 byte (class 0..9). */
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
    if (nw < 16 || hd[0] != 0x4D4C5031) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) hidden[h] += t[h];
        }
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = hidden[h];
            for (int c = 0; c < C; c++) scores[c] += hv * W2[(long)h * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```