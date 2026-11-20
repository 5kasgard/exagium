```hypothesis
Family: label-embmlp. Uses per-pixel value embeddings (64 pixels × 17 values × D) summed, then ReLU and linear layer. Captures non-linear pixel responses while keeping inference fast (64×D adds + D×10 mul-adds). Expect to beat label-linear (96.1%) and approach label-mlp16 (97.7%) with D=8 at lower cost (~6000 Ir/item). If dev accuracy < 96.5% or cost > label-mlp16, hypothesis fails.
```

```python
"""label-embmlp: per-pixel embeddings + ReLU + linear.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Tunes embedding size D on last 20% of train, then trains on all.
Weights format: magic 0x454D424D, d, D, C, num_values; then embeddings[d][num_values][D], b1[D], W2[D][C], b2[C] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR_GRID = [0.01, 0.003, 0.001]
L2_GRID = [1e-4, 1e-5]
D_GRID = [8, 16, 32]
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, C, D, lr, l2, epochs, rng):
    n, d = X.shape
    num_values = 17  # 0..16
    # embeddings: (d, num_values, D)
    E = (rng.uniform(-1, 1, (d, num_values, D)) * math.sqrt(6.0 / (num_values * D))).astype(np.float32)
    b1 = np.zeros(D, np.float32)
    W2 = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [E, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)
            # forward
            h = np.tile(b1, (bs, 1))
            for j in range(d):
                h += E[j, xb[:, j], :]
            h_relu = np.maximum(h, 0)
            scores = h_relu @ W2 + b2
            # softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs
            # backward
            dW2 = h_relu.T @ probs + l2 * W2
            db2 = probs.sum(0)
            dh_relu = probs @ W2.T
            dh = dh_relu * (h > 0)
            db1 = dh.sum(0)
            dE = np.zeros_like(E)
            for j in range(d):
                # accumulate dh[i] into dE[j, xb[i,j], :]
                np.add.at(dE[j], xb[:, j], dh)
            dE += l2 * E
            grads = [dE, db1, dW2, db2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def eval_model(X, y, params):
    E, b1, W2, b2 = params
    n, d = X.shape
    h = np.tile(b1, (n, 1))
    for j in range(d):
        h += E[j, X[:, j], :]
    h_relu = np.maximum(h, 0)
    scores = h_relu @ W2 + b2
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 64
    # tuning split: last 20%
    n = X.shape[0]
    split = n - n // 5
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    best_acc, best_params, best_cfg = -1, None, None
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for D in D_GRID:
        for lr in LR_GRID:
            for l2 in L2_GRID:
                params = train_model(X_train, y_train, C, D, lr, l2, epochs, rng)
                acc = eval_model(X_val, y_val, params)
                if acc > best_acc:
                    best_acc, best_params, best_cfg = acc, params, (D, lr, l2)
    # retrain on all data with best config
    D, lr, l2 = best_cfg
    final_params = train_model(X, y, C, D, lr, l2, epochs, rng)
    E, b1, W2, b2 = final_params
    # write weights
    num_values = 17
    magic = 0x454D424D
    out = struct.pack("<5i", magic, d, D, C, num_values)
    out += E.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 64 bytes (pixels 0..16). Output: one byte class.
 * Weights format: magic 0x454D424D, d, D, C, num_values; then embeddings[d][num_values][D], b1[D], W2[D][C], b2[C] (float32 LE). */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3], num_values = hd[4];
    if (d != 64 || num_values != 17 || C < 1 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * num_values * D + D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *E = (const float *)(hd + 5);
    const float *b1 = E + (long)d * num_values * D;
    const float *W2 = b1 + D;
    const float *b2 = W2 + (long)D * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(h, b1, sizeof(float) * D);
        for (int j = 0; j < d; j++) {
            const float *e = E + ((long)j * num_values + xi[j]) * D;
            for (int u = 0; u < D; u++) h[u] += e[u];
        }
        for (int u = 0; u < D; u++) if (h[u] < 0.0f) h[u] = 0.0f;
        for (int c = 0; c < C; c++) {
            float s = b2[c];
            for (int u = 0; u < D; u++) s += h[u] * W2[(long)u * C + c];
            scores[c] = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```