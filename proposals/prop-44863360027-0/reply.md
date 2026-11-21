```hypothesis
Family: label-fm (Factorization Machine). Captures pairwise feature interactions (e.g., education×occupation) crucial for tabular data with O(k·d) parameters, bridging linear and MLP. Uses same ordinal/categorical encoding as label-linear; trains with Adam on logistic loss. Latent dimension k tuned on last 20% of train.in (k∈{4,8,16,32}), then retrained on all data. Target: dev accuracy ≥0.857 (>label-linear 0.8524, approaching label-mlp16 0.8591) at ≤1500 Ir/item. Failure: accuracy ≤0.853 or cost >2000 Ir/item.
```
```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: ordinal z-scored per bin, categorical one-hot over training values.
FM: w0[C] + w[d][C]·x + 0.5·∑_f((∑_i v[i,f,c]·x_i)² - ∑_i v[i,f,c]²·x_i²).
Tunes k on validation split (last 20%), then trains on all. Adam lr=1e-2, L2=1e-4, batch=64.
Weights: magic=0x464D3130, d, k, C; float32 w0[C], w[d][C], V[d][k][C] (row-major)."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 100
K_CANDIDATES = [4, 8, 16, 32]
VAL_FRAC = 0.2


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoders(card, X):
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def encode(enc, X):
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def fm_forward(Z, w0, w, V, k, C):
    n, d = Z.shape
    scores = np.zeros((n, C), dtype=np.float32)
    for c in range(C):
        scores[:, c] = w0[c] + Z @ w[:, c]
        for f in range(k):
            v = V[:, f, c]
            vx = Z @ v
            v2x2 = Z @ (v * v)
            scores[:, c] += 0.5 * (vx * vx - v2x2)
    return scores


def softmax_grad(scores, y):
    P = np.exp(scores - scores.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    P[np.arange(len(y)), y] -= 1.0
    return (P / len(y)).astype(np.float32)


def fm_backward(Z, g, w, V, k, C, L2):
    n, d = Z.shape
    gw0 = g.sum(axis=0)
    gw = Z.T @ g + L2 * w
    gV = np.zeros_like(V)
    for c in range(C):
        gc = g[:, c:c+1]
        for f in range(k):
            v = V[:, f, c:c+1]
            vx = Z @ v
            gV[:, f, c] = (Z * vx).sum(axis=0) * gc.ravel() + L2 * V[:, f, c]
    return gw0, gw, gV


def train_fm(Z, y, k, C, epochs, rng):
    d = Z.shape[1]
    w0 = np.zeros(C, np.float32)
    w = (rng.uniform(-0.01, 0.01, (d, C)) * math.sqrt(2.0 / d)).astype(np.float32)
    V = (rng.uniform(-0.01, 0.01, (d, k, C)) * math.sqrt(2.0 / (d * k))).astype(np.float32)
    params = [w0, w, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for s in range(0, len(y), BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            scores = fm_forward(z, w0, w, V, k, C)
            g = softmax_grad(scores, yy)
            gw0, gw, gV = fm_backward(z, g, w, V, k, C, L2)
            grads = [gw0, gw, gV]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return w0, w, V


def eval_acc(Z, y, w0, w, V, k, C):
    scores = fm_forward(Z, w0, w, V, k, C)
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d_in = card["input"]["bytes"]

    enc = build_encoders(card, X)
    Z = encode(enc, X)
    n, d = Z.shape

    split = int(n * (1 - VAL_FRAC))
    Z_tr, Z_val = Z[:split], Z[split:]
    y_tr, y_val = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_k, best_acc = K_CANDIDATES[0], -1.0
    for k in K_CANDIDATES:
        w0, w, V = train_fm(Z_tr, y_tr, k, C, MAX_EPOCHS // 2, np.random.default_rng(SEED))
        acc = eval_acc(Z_val, y_val, w0, w, V, k, C)
        if acc > best_acc:
            best_acc, best_k = acc, k

    rng_final = np.random.default_rng(SEED)
    w0, w, V = train_fm(Z, y, best_k, C, MAX_EPOCHS, rng_final)

    magic = 0x464D3130
    out = struct.pack("<4i", magic, d, best_k, C)
    out += w0.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: 1 byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || k < 1 || k > 64 || d < 1) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)C + (long)d * C + (long)d * k * C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *w0 = (const float *)(hd + 4);
    const float *w = w0 + C;
    const float *V = w + (long)d * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *vx = malloc(sizeof(float) * k);
    float *v2x2 = malloc(sizeof(float) * k);

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = w0[c];
        for (int c = 0; c < C; c++) {
            const float *wc = w + (long)c * d;
            for (int j = 0; j < d; j++) scores[c] += wc[j] * xi[j];
        }

        for (int c = 0; c < C; c++) {
            const float *Vc = V + (long)c * d * k;
            for (int f = 0; f < k; f++) vx[f] = v2x2[f] = 0.0f;
            for (int j = 0; j < d; j++) {
                float xj = xi[j];
                if (xj == 0.0f) continue;
                const float *vjf = Vc + (long)j * k;
                for (int f = 0; f < k; f++) {
                    float v = vjf[f];
                    vx[f] += v * xj;
                    v2x2[f] += v * v * xj;
                }
            }
            for (int f = 0; f < k; f++) scores[c] += 0.5f * (vx[f] * vx[f] - v2x2[f]);
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```