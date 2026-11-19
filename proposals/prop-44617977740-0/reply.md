```hypothesis
A tiny MLP with 2–8 hidden ReLU units (H ∈ {2,4,8}) adds minimal non-linearity to the linear baseline.
Tuning H on the last 20% of train.in (by file order) then retraining on all data should beat
label-linear (0.8524) on dev while keeping inference cost close to linear (only a few extra multiply-adds).
If the best tuned H does not exceed 0.8524 accuracy on dev, the hypothesis fails.
```
```python
"""Tiny MLP family for label tasks with fixed-size byte items.
Chooses H ∈ {2,4,8} on the last 20% of train.in (file order), then trains on all data.
Exports same format as label-linear (magic 0x54424C31, d, H, C, T, b1, [W2, b2])."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
H_CANDIDATES = (2, 4, 8)   # tuned on the last 20% of train.in
TUNE_SEED = 12345          # fixed seed for tuning loop


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode(X, types, rng):
    """Return list of encoding matrices E_j (256 x dim_j) and the concatenated Z."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    return enc, Z


def train_mlp(Z, y, C, H, rng, n_epochs):
    """Train MLP with given H, return params (W1, b1, W2, b2)."""
    n, k = Z.shape
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), np.float32)
        b1 = np.zeros(C, np.float32)
        params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            def sg(S):
                S = S - S.max(axis=1, keepdims=True)
                P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                return (P / len(yy)).astype(np.float32)
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                g = sg(h @ W2 + b2)
                gh = (g @ W2.T) * (a > 0)
                grads = [z.T @ gh + L2 * W1, gh.sum(0), h.T @ g + L2 * W2, g.sum(0)]
            else:
                g = sg(z @ W1 + b1)
                grads = [z.T @ g + L2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params


def evaluate(Z, y, params, H, C):
    """Return accuracy on (Z, y)."""
    if H:
        W1, b1, W2, b2 = params
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
    else:
        W1, b1 = params
        logits = Z @ W1 + b1
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def pack_weights(card, enc, params, H, C):
    """Serialize to the exact binary format expected by infer.c."""
    d = card["input"]["bytes"]
    W = H if H else C
    off, T = 0, []
    for E in enc:
        dim = E.shape[1]
        T.append((E.astype(np.float32) @ params[0][off:off + dim]).astype(np.float32))
        off += dim
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += np.stack(T).astype("<f4").tobytes()
    out += params[1].astype("<f4").tobytes()          # b1
    if H:
        out += params[2].astype("<f4").tobytes()      # W2
        out += params[3].astype("<f4").tobytes()      # b2
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = len(y)
    split = int(0.8 * n)          # last 20% for tuning
    X_tr, X_tune = X[:split], X[split:]
    y_tr, y_tune = y[:split], y[split:]

    # Build encodings on the 80% training part (for tuning)
    enc_tr, Z_tr = encode(X_tr, card["input"]["types"], np.random.default_rng(SEED))
    # Apply same encodings to tune split
    Z_tune = np.concatenate([E[X_tune[:, j]] for j, E in enumerate(enc_tr)], axis=1).astype(np.float32)

    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(y_tr)))

    best_H, best_acc, best_params = 0, -1.0, None
    for H in H_CANDIDATES:
        rng = np.random.default_rng(TUNE_SEED + H)
        params = train_mlp(Z_tr, y_tr, C, H, rng, n_epochs)
        acc = evaluate(Z_tune, y_tune, params, H, C)
        if acc > best_acc:
            best_acc, best_H, best_params = acc, H, params

    # Retrain on full data with best_H using encodings from full data
    enc_full, Z_full = encode(X, card["input"]["types"], np.random.default_rng(SEED))
    n_epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng_full = np.random.default_rng(SEED)
    final_params = train_mlp(Z_full, y, C, best_H, rng_full, n_epochs_full)

    weights = pack_weights(card, enc_full, final_params, best_H, C)
    open(sys.argv[4], "wb").write(weights)


if __name__ == "__main__":
    main()
```
```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```