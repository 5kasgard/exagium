```hypothesis
Family: label-mlp-tiny (one hidden layer, 8–32 units, tuned on the 20% tail).
Why: optdigits is small (2867 train) and low-dimensional (64 ordinal features); a tiny MLP adds just enough non-linearity to beat linear without overfitting, and the table-lookup inference stays extremely fast.
Failure signal: dev accuracy ≤ 0.9613 (label-linear) or Ir/item > 6000 without accuracy gain.
```

```python
"""label-mlp-tiny: tiny MLP with one hidden layer (H tuned on last 20% of train).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x54424C32, d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
CANDIDATE_H = [4, 8, 16, 32]          # hidden sizes to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode(X, types, fit=True, enc=None):
    """Return (Z, enc_list). Z is float32 [n, k]. enc_list holds per-feature tables."""
    n, d = X.shape
    if fit:
        enc = []
        for j, t in enumerate(types):
            v = X[:, j].astype(np.float64)
            if t == "ordinal":
                mu, sd = v.mean(), v.std() or 1.0
                enc.append(((np.arange(256) - mu) / sd)[:, None])          # 256 x 1
            else:   # not used here, but keep generic
                vals = np.unique(X[:, j])
                E = np.zeros((256, len(vals))); E[vals, np.arange(len(vals))] = 1.0
                enc.append(E)
    Z_parts = [E[X[:, j]] for j, E in enumerate(enc)]
    Z = np.concatenate(Z_parts, axis=1).astype(np.float32)
    return Z, enc


def train_mlp(Z, y, H, C, rng, epochs):
    n, k = Z.shape
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
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


def accuracy(Z, y, params):
    W1, b1, W2, b2 = params
    h = np.maximum(Z @ W1 + b1, 0)
    logits = h @ W2 + b2
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]

    n = len(X)
    split = int(0.8 * n)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)

    # tuning split (last 20% in file order)
    tr_idx, tu_idx = idx[:split], idx[split:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_tu, y_tu = X[tu_idx], y[tu_idx]

    # encode on training split
    Z_tr, enc = encode(X_tr, types, fit=True)
    Z_tu, _ = encode(X_tu, types, fit=False, enc=enc)

    best_H, best_acc, best_params = None, -1.0, None
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))

    for H in CANDIDATE_H:
        params = train_mlp(Z_tr, y_tr, H, C, rng, epochs)
        acc = accuracy(Z_tu, y_tu, params)
        if acc > best_acc:
            best_acc, best_H, best_params = acc, H, params

    # final training on all data with best_H
    Z_all, enc = encode(X, types, fit=True)
    final_params = train_mlp(Z_all, y, best_H, C, rng, min(MAX_EPOCHS, math.ceil(SAMPLES / n)))
    W1, b1, W2, b2 = final_params

    # build per-feature lookup tables T[j][v][h] = E_j[v] @ W1[offset:offset+E_j.shape[1], :]
    d = X.shape[1]
    T = np.zeros((d, 256, best_H), dtype=np.float32)
    off = 0
    for j, E in enumerate(enc):
        T[j] = (E.astype(np.float32) @ W1[off:off + E.shape[1], :]).astype(np.float32)
        off += E.shape[1]

    # write weights.bin
    magic = 0x54424C32
    header = struct.pack("<4i", magic, d, best_H, C)
    body = T.astype("<f4").tobytes() + b1.astype("<f4").tobytes() + W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-mlp-tiny inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixels). Output record: 1 byte (class 0-9). */
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
    if (nw < 16 || hd[0] != 0x54424C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 64) { fprintf(stderr, "bad header\n"); return 3; }
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