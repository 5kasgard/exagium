```hypothesis
Family: label-factorized
Reason: A factorized linear model learns shared byte embeddings (256 x D) and position-specific classifiers (16 x D x C), capturing similarities between characters (e.g., letters vs punctuation) while drastically reducing parameters vs. label-linear's one-hot 4096 x C. This should improve generalization on the modest 24k training set and yield higher dev accuracy at lower inference cost (16*D*C multiply-adds per item).
Failure signal: Dev accuracy ≤ 0.3375 (label-linear) or inference cost ≥ 7359.9 Ir/item.
```

```python
import json, math, struct, sys

import numpy as np

# hyperparameters
SEED = 1
LR = 0.01
BATCH = 64
EPOCHS_TUNE = 30
EPOCHS_FULL = 100
L2 = 1e-5
D_CANDIDATES = [8, 16, 32, 64]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_eval(Xtr, ytr, Xte, yte, D, C, rng, epochs):
    n, d = Xtr.shape
    # init
    E = (rng.normal(0, 1/math.sqrt(D), (256, D))).astype(np.float32)
    W = (rng.normal(0, 1/math.sqrt(D), (d, D, C))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [E, W, b]
    # adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = Xtr[idx], yb[idx]
            # forward
            # logits = sum_j (E[xb[:,j]] @ W[j]) + b
            logits = np.zeros((len(yb), C), np.float32)
            for j in range(d):
                e = E[xb[:, j]]          # (B, D)
                logits += e @ W[j]       # (B, C)
            logits += b
            # softmax + cross-entropy gradient
            maxlog = logits.max(axis=1, keepdims=True)
            exp = np.exp(logits - maxlog)
            probs = exp / exp.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            probs /= len(yb)
            # gradients
            g_b = probs.sum(axis=0)
            g_W = np.zeros_like(W)
            g_E = np.zeros_like(E)
            for j in range(d):
                e = E[xb[:, j]]
                g_W[j] = e.T @ probs
                # scatter-add to g_E
                # we can do a loop over batch for simplicity (B small)
                for bi in range(len(yb)):
                    g_E[xb[bi, j]] += probs[bi] @ W[j].T
            g_W += L2 * W
            g_E += L2 * E
            grads = [g_E, g_W, g_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    # eval
    logits = np.zeros((len(yte), C), np.float32)
    for j in range(d):
        logits += E[Xte[:, j]] @ W[j]
    logits += b
    pred = logits.argmax(axis=1)
    return (pred == yte).mean(), params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = len(y)
    tune_size = n // 5
    train_size = n - tune_size
    Xtr, ytr = X[:train_size], y[:train_size]
    Xte, yte = X[train_size:], y[train_size:]

    rng = np.random.default_rng(SEED)
    best_acc, best_D, best_params = -1, None, None
    for D in D_CANDIDATES:
        rng_cand = np.random.default_rng(SEED + D)  # deterministic per D
        acc, params = train_eval(Xtr, ytr, Xte, yte, D, C, rng_cand, EPOCHS_TUNE)
        if acc > best_acc:
            best_acc, best_D, best_params = acc, D, params
    # retrain on full data
    rng_full = np.random.default_rng(SEED + best_D + 1000)
    _, final_params = train_eval(X, y, X[:1], y[:1], best_D, C, rng_full, EPOCHS_FULL)
    E, W, b = final_params
    # write weights.bin
    # magic 0x46435446 ("FCTF" for factorized), d, D, C
    out = struct.pack("<4i", 0x46435446, d, best_D, C)
    out += E.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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
    if (nw < 16 || hd[0] != 0x46435446) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (256L * D + 16L * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *W = E + 256L * D;
    const float *b = W + 16L * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    float *emb = malloc(sizeof(float) * D);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(logits, b, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = E + (long)xi[j] * D;
            const float *wj = W + (long)j * D * C;
            for (int c = 0; c < C; c++) {
                float sum = 0.0f;
                for (int u = 0; u < D; u++) sum += e[u] * wj[(long)u * C + c];
                logits[c] += sum;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```