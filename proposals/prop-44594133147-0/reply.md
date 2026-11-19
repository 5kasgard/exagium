```hypothesis
We propose label-embmlp: an embedding-based MLP that learns a dense vector (size E) for each of the 256 values of every feature, concatenates the 14 embeddings, and passes them through one ReLU hidden layer (size H) to the two output classes. This captures feature interactions more efficiently than one-hot linear models and should exceed label-linear's 0.8524 dev accuracy while staying faster than label-mlp16 (which uses one-hot + hidden-16). If dev accuracy ≤0.8524 or inference cost (Ir/item) exceeds label-mlp16's 2325, the hypothesis fails.
```

```python
"""label-embmlp: embedding MLP for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: per-feature-value embeddings (E) -> concat -> ReLU(H) -> linear(C).
Hyperparameters (E,H) chosen on last 20% of train.in (tuning split), then trained on all train."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH = 3e-3, 1e-4, 64
EPOCHS_TUNE = 20
EPOCHS_FULL = 100
CONFIGS = [(4, 16), (8, 32), (16, 64)]  # (E, H) pairs to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


def forward(X, params, d, E, H, C):
    """X: (n, d) uint8. Returns logits (n, C)."""
    emb, W1, b1, W2, b2 = params
    n = X.shape[0]
    # gather embeddings: (n, d, E)
    flat = emb[X.reshape(-1), np.arange(d)[:, None], :].reshape(n, d * E)  # (n, d*E)
    h = flat @ W1 + b1
    np.maximum(h, 0, out=h)
    logits = h @ W2 + b2
    return logits


def loss_and_grad(X, y, params, d, E, H, C):
    """Returns loss, grads for params list [emb, W1, b1, W2, b2]."""
    emb, W1, b1, W2, b2 = params
    n = X.shape[0]
    # forward
    flat = emb[X.reshape(-1), np.arange(d)[:, None], :].reshape(n, d * E)  # (n, d*E)
    h_pre = flat @ W1 + b1
    h = np.maximum(h_pre, 0)
    logits = h @ W2 + b2
    # softmax + cross entropy
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
    # add L2
    loss += L2 * (np.sum(W1**2) + np.sum(W2**2) + np.sum(emb**2))
    # backward
    grad_logits = probs
    grad_logits[np.arange(n), y] -= 1.0
    grad_logits /= n
    grad_W2 = h.T @ grad_logits + 2 * L2 * W2
    grad_b2 = grad_logits.sum(axis=0)
    grad_h = grad_logits @ W2.T
    grad_h[h_pre <= 0] = 0.0
    grad_W1 = flat.T @ grad_h + 2 * L2 * W1
    grad_b1 = grad_h.sum(axis=0)
    grad_flat = grad_h @ W1.T  # (n, d*E)
    # gradient for embeddings: scatter-add
    grad_emb = np.zeros_like(emb)
    grad_flat_reshaped = grad_flat.reshape(n, d, E)
    # vectorized scatter: for each feature j, add grad_flat_reshaped[:,j,:] to grad_emb[X[:,j], j, :]
    for j in range(d):
        np.add.at(grad_emb[:, j, :], X[:, j], grad_flat_reshaped[:, j, :])
    grad_emb += 2 * L2 * emb
    return loss, [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]


def train_loop(X, y, params, d, E, H, C, epochs, rng):
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            loss, grads = loss_and_grad(xb, yb, params, d, E, H, C)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    tune_size = n // 5
    train_idx = np.arange(n - tune_size)
    tune_idx = np.arange(n - tune_size, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_cfg = None

    for E, H in CONFIGS:
        # init
        emb = rng.uniform(-0.1, 0.1, (d, 256, E)).astype(np.float32)
        W1 = rng.uniform(-0.1, 0.1, (d * E, H)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = rng.uniform(-0.1, 0.1, (H, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
        # train on train split
        params = train_loop(X_train, y_train, params, d, E, H, C, EPOCHS_TUNE, rng)
        # evaluate on tune split
        logits = forward(X_tune, params, d, E, H, C)
        acc = accuracy(logits, y_tune)
        if acc > best_acc:
            best_acc = acc
            best_params = [p.copy() for p in params]
            best_cfg = (E, H)

    # retrain on full data with best config
    E, H = best_cfg
    emb, W1, b1, W2, b2 = best_params
    params = [emb, W1, b1, W2, b2]
    params = train_loop(X, y, params, d, E, H, C, EPOCHS_FULL, rng)
    emb, W1, b1, W2, b2 = params

    # pack weights: magic, d, E, H, C; then emb[d,256,E], W1[d*E,H], b1[H], W2[H,C], b2[C]
    magic = 0x454D4244  # 'DBME' little-endian
    out = struct.pack("<5i", magic, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index (0 or 1). */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d <= 0 || E <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + (long)j * E, src, E * sizeof(float));
        }

        // hidden = relu(concat @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) sum += concat[k] * W1[(long)k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // logits = hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```