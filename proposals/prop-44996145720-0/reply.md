```hypothesis
Family: label-linear-te (Target Encoded Linear). Replaces one-hot/z-score encoding with per-column target encoding (smoothed P(y=1|value)), yielding 14 dense features. Logistic regression on these captures non-linear bin/category effects that linear-on-one-hot misses, while keeping inference identical to label-linear (14×256×2 lookup + 28 mul-adds). Should beat label-linear's 0.8524 dev accuracy at same ~720 Ir/item. Failure: accuracy ≤ 0.8524 or instructions not ~720 Ir/item.
```
```python
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
ALPHAS = [1, 5, 10, 20, 50, 100, 200]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_lr(Z, y, C, rng, epochs, batch, lr, l2):
    n, k = Z.shape
    W = np.zeros((k, C), np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z[idx], y[idx]
            S = z @ W + b
            S = S - S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            P[np.arange(len(yy)), yy] -= 1.0
            G = (P / len(yy)).astype(np.float32)
            grads = [z.T @ G + l2 * W, G.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W, b


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]
    n_train = int(0.8 * n)
    rng = np.random.default_rng(SEED)

    # Tuning split (last 20% in file order)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_tu, y_tu = X[n_train:], y[n_train:]
    global_mean = y_tr.mean()

    best_alpha, best_acc = ALPHAS[0], -1.0
    for alpha in ALPHAS:
        # Compute target encodings on training portion
        te = np.zeros((d, 256), np.float32)
        for j in range(d):
            for v in range(256):
                mask = (X_tr[:, j] == v)
                cnt = mask.sum()
                if cnt:
                    te[j, v] = (y_tr[mask].sum() + alpha * global_mean) / (cnt + alpha)
                else:
                    te[j, v] = global_mean
        Z_tr = te[X_tr].astype(np.float32)  # (n_train, d)
        Z_tu = te[X_tu].astype(np.float32)
        W, b = train_lr(Z_tr, y_tr, C, rng, min(MAX_EPOCHS, math.ceil(SAMPLES / n_train)), BATCH, LR, L2)
        # Evaluate on tuning
        S = Z_tu @ W + b
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        pred = P.argmax(axis=1)
        acc = (pred == y_tu).mean()
        if acc > best_acc:
            best_acc, best_alpha = acc, alpha

    # Retrain on full data with best_alpha
    te = np.zeros((d, 256), np.float32)
    for j in range(d):
        for v in range(256):
            mask = (X[:, j] == v)
            cnt = mask.sum()
            if cnt:
                te[j, v] = (y[mask].sum() + best_alpha * global_mean) / (cnt + best_alpha)
            else:
                te[j, v] = global_mean
    Z = te[X].astype(np.float32)
    W, b = train_lr(Z, y, C, rng, min(MAX_EPOCHS, math.ceil(SAMPLES / n)), BATCH, LR, L2)

    # Build T[j, v, c] = W[j, c] * te[j, v]
    T = np.zeros((d, 256, C), np.float32)
    for j in range(d):
        for v in range(256):
            T[j, v] = W[j] * te[j, v]

    H = 0
    out = struct.pack("<4i", 0x54424C31, d, H, C) + T.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Table family (linear) for label tasks: ./infer weights.bin in.rec N
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