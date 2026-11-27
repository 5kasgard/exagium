```hypothesis
Factorization Machine (FM) with low-rank embeddings captures pairwise feature interactions, which are crucial for tabular data like Adult. With d=14 features and rank k=4, inference cost is O(d^2 k) ≈ 784 multiply-adds per item, far below GBDT/MLP. The model is trained with Adam and L2 regularization, hyperparameters (k, lr, L2) selected on the last 20% tuning split. If dev accuracy ≤ 0.8524 (label-linear) or instructions/item exceed ~1500, the family fails its goal.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid for tuning split search
GRID = [
    {"k": 2, "lr": 0.01, "l2": 1e-4},
    {"k": 2, "lr": 0.003, "l2": 1e-4},
    {"k": 2, "lr": 0.001, "l2": 1e-4},
    {"k": 4, "lr": 0.01, "l2": 1e-4},
    {"k": 4, "lr": 0.003, "l2": 1e-4},
    {"k": 4, "lr": 0.001, "l2": 1e-4},
    {"k": 8, "lr": 0.01, "l2": 1e-4},
    {"k": 8, "lr": 0.003, "l2": 1e-4},
    {"k": 8, "lr": 0.001, "l2": 1e-4},
]
SEED = 12345
BATCH = 256
EPOCHS_TUNE = 15
EPOCHS_FULL = 30
MAGIC = 0x464D0001  # 'FM' + version


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(d, k, rng):
    w0 = np.float32(0.0)
    w = np.zeros((d, 256), dtype=np.float32)
    v = rng.uniform(-0.01, 0.01, (d, 256, k)).astype(np.float32)
    return w0, w, v


def forward(X, w0, w, v, k):
    n = X.shape[0]
    linear = np.full(n, w0, dtype=np.float32)
    for i in range(X.shape[1]):
        linear += w[i, X[:, i]]
    pairwise = np.zeros(n, dtype=np.float32)
    for i in range(X.shape[1]):
        vi = v[i, X[:, i]]  # (n, k)
        for j in range(i + 1, X.shape[1]):
            vj = v[j, X[:, j]]
            pairwise += np.sum(vi * vj, axis=1)
    return linear + pairwise


def predict(X, w0, w, v, k):
    scores = forward(X, w0, w, v, k)
    return (scores > 0).astype(np.int64)


def accuracy(X, y, w0, w, v, k):
    return np.mean(predict(X, w0, w, v, k) == y)


def train_fm(X, y, k, lr, l2, epochs, rng):
    n, d = X.shape
    w0, w, v = init_params(d, k, rng)
    # Adam state
    m_w0, v_w0 = 0.0, 0.0
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_v = np.zeros_like(v)
    v_v = np.zeros_like(v)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi = X[idx]
            yi = y[idx]
            bsz = len(idx)

            # Forward
            linear = np.full(bsz, w0, dtype=np.float32)
            for i in range(d):
                linear += w[i, xi[:, i]]
            pairwise = np.zeros(bsz, dtype=np.float32)
            for i in range(d):
                vi = v[i, xi[:, i]]
                for j in range(i + 1, d):
                    vj = v[j, xi[:, j]]
                    pairwise += np.sum(vi * vj, axis=1)
            score = linear + pairwise

            # Logistic loss gradient
            p = 1.0 / (1.0 + np.exp(-score))
            grad_score = (p - yi).astype(np.float32) / bsz

            # Gradients
            gw0 = np.sum(grad_score)
            gw = np.zeros_like(w)
            gv = np.zeros_like(v)

            for i in range(d):
                gw[i, xi[:, i]] += grad_score
            for i in range(d):
                vi = v[i, xi[:, i]]
                for j in range(i + 1, d):
                    vj = v[j, xi[:, j]]
                    gv[i, xi[:, i]] += grad_score[:, None] * vj
                    gv[j, xi[:, j]] += grad_score[:, None] * vi

            # L2 regularization
            gw += l2 * w
            gv += l2 * v

            t += 1
            # Adam updates
            m_w0 = beta1 * m_w0 + (1 - beta1) * gw0
            v_w0 = beta2 * v_w0 + (1 - beta2) * gw0 * gw0
            m_w = beta1 * m_w + (1 - beta1) * gw
            v_w = beta2 * v_w + (1 - beta2) * gw * gw
            m_v = beta1 * m_v + (1 - beta1) * gv
            v_v = beta2 * v_v + (1 - beta2) * gv * gv

            m_w0_hat = m_w0 / (1 - beta1 ** t)
            v_w0_hat = v_w0 / (1 - beta2 ** t)
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            m_v_hat = m_v / (1 - beta1 ** t)
            v_v_hat = v_v / (1 - beta2 ** t)

            w0 -= lr * m_w0_hat / (np.sqrt(v_w0_hat) + eps)
            w -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
            v -= lr * m_v_hat / (np.sqrt(v_v_hat) + eps)

    return w0, w, v


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    n = X.shape[0]

    # Split: last 20% for tuning
    tune_size = n // 5
    train_X, train_y = X[:-tune_size], y[:-tune_size]
    tune_X, tune_y = X[-tune_size:], y[-tune_size:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_model = None

    for hp in GRID:
        k, lr, l2 = hp["k"], hp["lr"], hp["l2"]
        # Use a deterministic sub-seed for each grid point
        sub_rng = np.random.default_rng(SEED + k * 1000 + int(lr * 10000))
        w0, w, v = train_fm(train_X, train_y, k, lr, l2, EPOCHS_TUNE, sub_rng)
        acc = accuracy(tune_X, tune_y, w0, w, v, k)
        if acc > best_acc:
            best_acc = acc
            best_params = (k, lr, l2)
            best_model = (w0, w, v)

    # Retrain on full data with best hyperparameters
    k, lr, l2 = best_params
    final_rng = np.random.default_rng(SEED)
    w0, w, v = train_fm(X, y, k, lr, l2, EPOCHS_FULL, final_rng)

    # Write weights
    out = struct.pack("<4i", MAGIC, d, k, 2)  # C=2
    out += struct.pack("<f", w0)
    for i in range(d):
        out += w[i].astype("<f4").tobytes()
        out += v[i].astype("<f4").tobytes()
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
    if (nw < 20 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d < 1 || k < 1 || k > 8) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected = 20 + (long)d * (256 + 256 * k) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 5); // after magic,d,k,C,w0
    float w0 = p[0];
    const float *w = p + 1;
    const float *v = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        float score = w0;
        // linear term
        for (int j = 0; j < d; j++) score += w[(long)j * 256 + xi[j]];

        // pairwise interactions
        for (int j = 0; j < d; j++) {
            const float *vj = v + ((long)j * 256 + xi[j]) * k;
            for (int l = j + 1; l < d; l++) {
                const float *vl = v + ((long)l * 256 + xi[l]) * k;
                for (int f = 0; f < k; f++) score += vj[f] * vl[f];
            }
        }

        out_byte = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```