```hypothesis
We propose "label-emb", a linear model on learned feature embeddings. Each of the 14 features (ordinal and categorical) gets a D-dimensional embedding (D tuned on the 20% tuning split). The embeddings are concatenated and fed to a linear output layer. This captures category similarities and reduces parameters vs. one-hot, while keeping inference cost identical to label-linear (one table lookup per feature). We expect it to beat label-linear's 0.8524 dev accuracy; if it does not, the hypothesis is falsified.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
TUNE_EPOCHS = 20
FULL_EPOCHS = 50


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_embedding(X, y, D, LR, L2, epochs, seed):
    n, d = X.shape
    C = 2
    rng = np.random.default_rng(seed)
    E = [rng.normal(0, 1.0 / np.sqrt(D), size=(256, D)).astype(np.float32) for _ in range(d)]
    W = rng.normal(0, 1.0 / np.sqrt(d * D), size=(d * D, C)).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = E + [W, b]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb = X[idx]
            yb = y[idx]
            B = len(yb)

            logits = np.zeros((B, C), np.float32)
            for j in range(d):
                Ej = E[j][Xb[:, j]]
                Wj = W[j * D:(j + 1) * D]
                logits += Ej @ Wj
            logits += b

            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dlogits = probs / B

            db = dlogits.sum(axis=0)
            dW = np.zeros_like(W)
            dE = [np.zeros_like(E[j]) for j in range(d)]
            for j in range(d):
                Ej = E[j][Xb[:, j]]
                Wj = W[j * D:(j + 1) * D]
                dW[j * D:(j + 1) * D] = Ej.T @ dlogits + L2 * Wj
                dEj = dlogits @ Wj.T
                np.add.at(dE[j], Xb[:, j], dEj)
                dE[j] += L2 * E[j]

            grads = dE + [dW, db]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] *= 0.9
                m[i] += 0.1 * gr
                v[i] *= 0.999
                v[i] += 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    T = np.zeros((d, 256, C), np.float32)
    for j in range(d):
        T[j] = E[j] @ W[j * D:(j + 1) * D]
    return T, b


def evaluate(T, b, X, y):
    n = X.shape[0]
    logits = np.zeros((n, 2), np.float32)
    for j in range(14):
        logits += T[j, X[:, j]]
    logits += b
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X)
    tune_size = n // 5
    X_train, X_tune = X[:n - tune_size], X[n - tune_size:]
    y_train, y_tune = y[:n - tune_size], y[n - tune_size:]

    best_acc = -1.0
    best_params = None
    for D in [4, 8, 16, 32]:
        for LR in [1e-3, 3e-3, 1e-2]:
            for L2 in [1e-4, 1e-5]:
                print(f"Trying D={D}, LR={LR}, L2={L2}", file=sys.stderr)
                T, b = train_embedding(X_train, y_train, D, LR, L2, TUNE_EPOCHS, SEED)
                acc = evaluate(T, b, X_tune, y_tune)
                print(f"  Tune accuracy: {acc:.4f}", file=sys.stderr)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (D, LR, L2)

    print(f"Best: D={best_params[0]}, LR={best_params[1]}, L2={best_params[2]}, acc={best_acc:.4f}", file=sys.stderr)
    D, LR, L2 = best_params
    T, b = train_embedding(X, y, D, LR, L2, FULL_EPOCHS, SEED)

    magic = 0x454D4231  # 'EMB1'
    out = struct.pack("<4i", magic, d, 0, C)
    out += T.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (H != 0 || C != 2 || d != 14) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *b = T + (long)d * 256 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float scores[2];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        scores[0] = b[0]; scores[1] = b[1];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * C;
            scores[0] += t[0];
            scores[1] += t[1];
        }
        unsigned char best = (scores[1] > scores[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```