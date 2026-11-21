```hypothesis
Family: label-mlp-tuned (tiny MLP with 4-16 hidden units, tuned on last 20% of train).
Why: optdigits benefits from non-linear pixel interactions; a small ReLU MLP adds minimal inference cost (≈600 FLOPs) over linear (640 FLOPs) while typically reaching >97% accuracy.
Failure: if dev accuracy ≤ 0.9613 (label-linear) or instruction count exceeds label-linear by >20%.
```
```python
"""label-mlp-tuned: tiny MLP with hyperparameter search on tuning split.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid
H_CHOICES = [4, 8, 16]
LR_CHOICES = [1e-3, 3e-3, 1e-2]
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
SEED = 1


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoders(X, types):
    """Return list of encoding matrices (256, dim_j) for each feature."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None].astype(np.float32))
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def encode(X, enc):
    """Apply encoders to X (n, d) -> Z (n, sum dim_j)."""
    parts = [E[X[:, j]] for j, E in enumerate(enc)]
    return np.concatenate(parts, axis=1).astype(np.float32)


def train_model(Ztr, ytr, Zval, yval, k, H, C, lr, l2, batch, max_epochs, rng):
    """Train MLP on Ztr, validate on Zval. Return (params, val_acc)."""
    n, _ = Ztr.shape
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

    def softmax_grad(S, yy):
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Ztr[idx], ytr[idx]
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                g = softmax_grad(h @ W2 + b2, yy)
                gh = (g @ W2.T) * (a > 0)
                grads = [
                    z.T @ gh + l2 * W1,
                    gh.sum(0),
                    h.T @ g + l2 * W2,
                    g.sum(0)
                ]
            else:
                g = softmax_grad(z @ W1 + b1, yy)
                grads = [z.T @ g + l2 * W1, g.sum(0)]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # validation accuracy
    if H:
        a = Zval @ W1 + b1
        h = np.maximum(a, 0)
        scores = h @ W2 + b2
    else:
        scores = Zval @ W1 + b1
    pred = scores.argmax(axis=1)
    acc = (pred == yval).mean()
    return params, acc


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]

    # split: last 20% for tuning
    n = len(X)
    n_tune = n // 5
    n_train = n - n_tune
    X_train, y_train = X[:n_train], y[:n_train]
    X_tune, y_tune = X[n_train:], y[n_train:]

    # build encoders on training part only
    enc_train = build_encoders(X_train, types)
    Z_train = encode(X_train, enc_train)
    Z_tune = encode(X_tune, enc_train)
    k = Z_train.shape[1]

    # hyperparameter search
    best_acc = -1.0
    best_params = None
    best_H = 0
    best_lr = 0.0
    base_rng = np.random.default_rng(SEED)

    for idx, (H, lr) in enumerate([(h, lr) for h in H_CHOICES for lr in LR_CHOICES]):
        rng = np.random.default_rng(SEED + idx + 1)  # deterministic per config
        params, acc = train_model(Z_train, y_train, Z_tune, y_tune, k, H, C, lr, L2, BATCH, MAX_EPOCHS, rng)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_H = H
            best_lr = lr

    # retrain on all data with best config
    enc_all = build_encoders(X, types)
    Z_all = encode(X, enc_all)
    k_all = Z_all.shape[1]
    rng_final = np.random.default_rng(SEED + 999)
    final_params, _ = train_model(Z_all, y, Z_all, y, k_all, best_H, C, best_lr, L2, BATCH, MAX_EPOCHS, rng_final)

    # export weights in label-linear format (magic, d, H, C, T, b1, [W2, b2])
    d = X.shape[1]
    H = best_H
    W = H if H else C
    # compute per-byte lookup tables T
    off = 0
    T_list = []
    for E in enc_all:
        dim = E.shape[1]
        # E: (256, dim), W1: (k, W) -> take rows off:off+dim
        T_list.append((E @ final_params[0][off:off + dim]).astype(np.float32))
        off += dim
    T = np.stack(T_list)  # (d, 256, W)

    b1 = final_params[1].astype(np.float32)
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        W2 = final_params[2].astype(np.float32)
        b2 = final_params[3].astype(np.float32)
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-mlp-tuned inference: ./infer weights.bin in.rec N
 * Input record: d bytes (pixels). Output record: 1 byte (class 0..9).
 * Weight format: magic 0x54424C31, int32 d, H, C; float32 T[d][256][W], b1[W];
 * if H>0: float32 W2[H][C], b2[C] (W = H, else W = C). */
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