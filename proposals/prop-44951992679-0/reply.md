```hypothesis
Factorization Machine (FM) with low-rank embeddings for pairwise interactions. The adult dataset has 14 mixed ordinal/categorical features; FM captures feature interactions efficiently with O(d*K) inference cost. It should beat label-linear (0.8524) by modeling interactions while staying faster than label-mlp16 (2325 Ir/item). Failure if dev accuracy ≤0.8524 or Ir/item >2500.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Chooses embedding dim K on tuning split (last 20%),
then trains on all data. Exports: magic 0x464D3132, d, K, C; float32 bias[C], w[d][256][C], e[d][256][K][C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
K_CHOICES = [4, 8, 16]  # tuning grid


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    return exp / exp.sum(axis=1, keepdims=True)


def fm_forward(X, bias, w, e, K, C):
    """X: (N, d) uint8. Returns scores (N, C) float32."""
    N, d = X.shape
    scores = np.tile(bias, (N, 1))
    for j in range(d):
        v = X[:, j]
        scores += w[j, v, :]
    for c in range(C):
        for k in range(K):
            sum_e = np.zeros(N, np.float32)
            sum_e2 = np.zeros(N, np.float32)
            for j in range(d):
                v = X[:, j]
                ev = e[j, v, k, c]
                sum_e += ev
                sum_e2 += ev * ev
            scores[:, c] += 0.5 * (sum_e * sum_e - sum_e2)
    return scores


def fm_gradients(X, y, scores, bias, w, e, K, C, L2):
    """Compute gradients for a batch. Returns (gbias, gw, ge)."""
    N, d = X.shape
    probs = softmax(scores)
    probs[np.arange(N), y] -= 1.0
    probs /= N

    gbias = probs.sum(axis=0).astype(np.float32)
    gw = np.zeros_like(w)
    ge = np.zeros_like(e)

    for j in range(d):
        v = X[:, j]
        for c in range(C):
            gw[j, v, c] += probs[:, c].sum()
    # L2 on w
    gw += L2 * w

    for c in range(C):
        pc = probs[:, c]
        for k in range(K):
            sum_e = np.zeros(N, np.float32)
            for j in range(d):
                v = X[:, j]
                sum_e += e[j, v, k, c]
            for j in range(d):
                v = X[:, j]
                ev = e[j, v, k, c]
                ge[j, v, k, c] += np.dot(pc, sum_e - ev)
    # L2 on e
    ge += L2 * e
    return gbias, gw, ge


def adam_update(params, grads, states, t, lr):
    for i, (p, g) in enumerate(zip(params, grads)):
        m, v = states[i]
        m *= 0.9; m += 0.1 * g
        v *= 0.999; v += 0.001 * g * g
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def train_fm(X, y, K, epochs, rng):
    N, d = X.shape
    C = 2
    # init
    bias = np.zeros(C, np.float32)
    w = (rng.uniform(-0.01, 0.01, (d, 256, C)) * math.sqrt(6.0 / (d * 256))).astype(np.float32)
    e = (rng.uniform(-0.01, 0.01, (d, 256, K, C)) * math.sqrt(6.0 / (d * 256 * K))).astype(np.float32)
    params = [bias, w, e]
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(epochs):
        order = rng.permutation(N)
        for s in range(0, N, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            scores = fm_forward(Xb, bias, w, e, K, C)
            gbias, gw, ge = fm_gradients(Xb, yb, scores, bias, w, e, K, C, L2)
            t += 1
            adam_update(params, [gbias, gw, ge], states, t, LR)
    return bias, w, e


def evaluate(X, y, bias, w, e, K, C):
    scores = fm_forward(X, bias, w, e, K, C)
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)

    # split: last 20% for tuning
    n = len(X)
    n_tune = n // 5
    X_tr, y_tr = X[:-n_tune], y[:-n_tune]
    X_tune, y_tune = X[-n_tune:], y[-n_tune:]

    best_K, best_acc = K_CHOICES[0], -1.0
    for K in K_CHOICES:
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))
        rng = np.random.default_rng(SEED + K)
        bias, w, e = train_fm(X_tr, y_tr, K, epochs, rng)
        acc = evaluate(X_tune, y_tune, bias, w, e, K, C)
        if acc > best_acc:
            best_acc, best_K = acc, K

    # final training on all data
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng = np.random.default_rng(SEED + best_K)
    bias, w, e = train_fm(X, y, best_K, epochs, rng)

    # write weights.bin
    magic = 0x464D3132  # "FM12"
    header = struct.pack("<4I", magic, d, best_K, C)
    out = header
    out += bias.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += e.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (0 or 1). */
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || K < 1) { fprintf(stderr, "bad header\n"); return 3; }

    size_t expected = 16
        + (size_t)C * 4
        + (size_t)d * 256 * C * 4
        + (size_t)d * 256 * K * C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *w = bias + C;
    const float *e = w + (size_t)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) scores[c] = bias[c];

        // linear part
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            const float *wj = w + ((size_t)j * 256 + v) * C;
            for (int c = 0; c < C; c++) scores[c] += wj[c];
        }

        // FM part
        for (int c = 0; c < C; c++) {
            for (int kk = 0; kk < K; kk++) {
                float sum_e = 0.0f, sum_e2 = 0.0f;
                for (int j = 0; j < d; j++) {
                    int v = xi[j];
                    float ev = e[((size_t)j * 256 + v) * K * C + kk * C + c];
                    sum_e += ev;
                    sum_e2 += ev * ev;
                }
                scores[c] += 0.5f * (sum_e * sum_e - sum_e2);
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```