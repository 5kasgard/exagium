```hypothesis
We propose a factorization machine (FM) with class-specific latent vectors (label-fm). FM captures pairwise feature interactions efficiently via low-rank factorization, which is crucial for tabular data like Adult where interactions (e.g., education × occupation) boost accuracy. The model adds minimal inference overhead (O(d·k·C)) over linear. If dev accuracy does not exceed label-linear (0.8524) or inference cost exceeds label-mlp16 (2325 Ir/item), the approach fails.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Encodes each byte value directly with a linear weight and a latent vector per class.
Trains with Adam on cross-entropy, tunes latent dimension k on last 20% of train.in."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
K_CHOICES = [4, 8, 16]  # latent dimensions to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(d, C, k, seen, rng):
    bias = np.zeros(C, np.float32)
    linear = np.zeros((d, 256, C), np.float32)
    latent = np.zeros((d, 256, k, C), np.float32)
    for j in range(d):
        for v in seen[j]:
            linear[j, v] = rng.normal(0, 0.01, C)
            latent[j, v] = rng.normal(0, 0.01, (k, C))
    return bias, linear, latent


def forward(X, bias, linear, latent, C, k):
    n = X.shape[0]
    scores = np.tile(bias, (n, 1))
    # linear part
    for j in range(X.shape[1]):
        v = X[:, j]
        scores += linear[j, v]
    # interaction part
    for c in range(C):
        sum_u = np.zeros((n, k), np.float32)
        sq_u = np.zeros((n, k), np.float32)
        for j in range(X.shape[1]):
            v = X[:, j]
            u = latent[j, v, :, c]
            sum_u += u
            sq_u += u * u
        inter = 0.5 * (np.sum(sum_u * sum_u, axis=1) - np.sum(sq_u, axis=1))
        scores[:, c] += inter
    return scores


def loss_grad(scores, y):
    n = scores.shape[0]
    max_s = scores.max(axis=1, keepdims=True)
    exp_s = np.exp(scores - max_s)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return probs / n  # dL/dscores


def accuracy(X, y, bias, linear, latent, C, k):
    scores = forward(X, bias, linear, latent, C, k)
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def train_model(X_tr, y_tr, X_val, y_val, d, C, k, rng):
    seen = [np.unique(X_tr[:, j]) for j in range(d)]
    bias, linear, latent = init_params(d, C, k, seen, rng)
    params = [bias, linear, latent]
    shapes = [p.shape for p in params]
    state = [(np.zeros(s, np.float32), np.zeros(s, np.float32)) for s in shapes]
    t = 0
    n = len(X_tr)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            b = len(idx)
            # forward
            scores = forward(Xb, bias, linear, latent, C, k)
            # loss gradient
            g_scores = loss_grad(scores, yb)
            # gradients
            g_bias = g_scores.sum(axis=0)
            g_linear = np.zeros_like(linear)
            g_latent = np.zeros_like(latent)
            # linear grad
            for j in range(d):
                v = Xb[:, j]
                np.add.at(g_linear[j], v, g_scores)
            # latent grad
            for c in range(C):
                gc = g_scores[:, c]
                sum_u = np.zeros((b, k), np.float32)
                for j in range(d):
                    v = Xb[:, j]
                    sum_u += latent[j, v, :, c]
                for j in range(d):
                    v = Xb[:, j]
                    u = latent[j, v, :, c]
                    gu = gc[:, None] * (sum_u - u)
                    np.add.at(g_latent[j, :, :, c], v, gu)
            # L2
            g_bias += L2 * bias
            g_linear += L2 * linear
            g_latent += L2 * latent
            grads = [g_bias, g_linear, g_latent]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    val_acc = accuracy(X_val, y_val, bias, linear, latent, C, k)
    return val_acc, bias, linear, latent


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = len(X)
    split = int(0.8 * n)
    idx_tr, idx_val = np.arange(split), np.arange(split, n)
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_val, y_val = X[idx_val], y[idx_val]

    rng = np.random.default_rng(SEED)
    best_k, best_acc = K_CHOICES[0], -1.0
    for k in K_CHOICES:
        # Use a fresh seed for each k to keep runs independent but deterministic
        rng_k = np.random.default_rng(SEED + k)
        acc, _, _, _ = train_model(X_tr, y_tr, X_val, y_val, d, C, k, rng_k)
        if acc > best_acc:
            best_acc, best_k = acc, k

    # Retrain on full data with best_k
    rng_full = np.random.default_rng(SEED)
    _, bias, linear, latent = train_model(X, y, X_val, y_val, d, C, best_k, rng_full)

    # Save weights
    magic = 0x464D544C  # "LTMF"
    out = struct.pack("<4i", magic, d, C, best_k)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += latent.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte (class index). */
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
    if (nw < 16 || hd[0] != 0x464D544C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], C = hd[2], k = hd[3];
    if (C < 1 || C > 256 || k < 1 || k > 64 || d < 1 || d > 256) {
        fprintf(stderr, "bad weights dims\n"); return 3;
    }
    long expected = 16L + 4L * C + 4L * d * 256 * C + 4L * d * 256 * k * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *latent = linear + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *sum_u = malloc(sizeof(float) * k);
    float *sq_u = malloc(sizeof(float) * k);

    for (long i = 0; i < N; i++) {
        uint32_t klen; const unsigned char *xi = next_rec(x, nx, &pos, &klen);
        if ((int)klen != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, klen, d); return 3; }

        memcpy(scores, bias, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            uint8_t v = xi[j];
            const float *w = linear + ((long)j * 256 + v) * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }

        for (int c = 0; c < C; c++) {
            memset(sum_u, 0, k * sizeof(float));
            memset(sq_u, 0, k * sizeof(float));
            for (int j = 0; j < d; j++) {
                uint8_t v = xi[j];
                const float *u = latent + (((long)j * 256 + v) * k + 0) * C + c;
                for (int a = 0; a < k; a++) {
                    float val = u[a * C];
                    sum_u[a] += val;
                    sq_u[a] += val * val;
                }
            }
            float inter = 0.0f;
            for (int a = 0; a < k; a++) inter += sum_u[a] * sum_u[a] - sq_u[a];
            scores[c] += 0.5f * inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```