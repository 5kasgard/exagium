```hypothesis
We propose "label-emblinear": per-position byte embeddings (size E) concatenated and fed to a linear classifier. This replaces the 4096-d one-hot input of label-linear with 16*E dense features, capturing character similarities and reducing parameters. It should generalize better than label-linear on this character-level task while keeping inference extremely fast (embedding lookups + one matrix-vector multiply). If dev accuracy does not exceed label-linear's 0.3375 or inference cost exceeds ~7.3k Ir/item, the approach fails.
```

```python
"""label-emblinear: per-position embeddings + linear classifier.
train.py: deterministic, single-thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all train.in."""
import json, math, os, struct, sys

import numpy as np

# Force single-threaded BLAS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

SEED = 1
SAMPLES = 600000
MAX_EPOCHS = 100
BATCH = 64
L2 = 1e-4
E_GRID = [16, 32, 64]
LR_GRID = [0.01, 0.003]

MAGIC = 0x454D424C  # 'L','B','M','E' little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(d, E, C, rng):
    """Return list of parameter arrays: [emb_0,...,emb_{d-1}, W, b] all float32."""
    params = []
    for _ in range(d):
        # embeddings: (256, E)
        emb = rng.normal(0, 1.0 / math.sqrt(E), (256, E)).astype(np.float32)
        params.append(emb)
    # W: (d*E, C)
    W = (rng.uniform(-1, 1, (d * E, C)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    params.append(W)
    # b: (C,)
    b = np.zeros(C, np.float32)
    params.append(b)
    return params


def forward(X, params, d, E):
    """X: (n, d) uint8. Returns scores (n, C) and concatenated embeddings (n, d*E) for backprop."""
    n = X.shape[0]
    embs = params[:-2]
    W = params[-2]
    b = params[-1]
    X_emb = np.empty((n, d * E), np.float32)
    for j in range(d):
        X_emb[:, j * E:(j + 1) * E] = embs[j][X[:, j]]
    scores = X_emb @ W + b
    return scores, X_emb


def loss_grad(scores, y):
    """Cross-entropy gradient w.r.t scores. Returns (n, C) float32."""
    n = scores.shape[0]
    scores_max = scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return (probs / n).astype(np.float32)


def accuracy(scores, y):
    return (scores.argmax(axis=1) == y).mean()


def train_loop(X, y, params, state, lr, l2, batch, rng, t, epochs):
    n = X.shape[0]
    d = len(params) - 2
    E = params[0].shape[1]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            scores, X_emb = forward(xb, params, d, E)
            g = loss_grad(scores, yb)
            # gradients for W and b
            dW = X_emb.T @ g + l2 * params[-2]
            db = g.sum(axis=0)
            # gradients for embeddings
            dX_emb = g @ params[-2].T  # (batch, d*E)
            dembs = []
            for j in range(d):
                demb = np.zeros_like(params[j])
                part = dX_emb[:, j * E:(j + 1) * E]
                np.add.at(demb, xb[:, j], part)
                dembs.append(demb + l2 * params[j])
            grads = dembs + [dW, db]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return t


def evaluate(X, y, params, d, E):
    scores, _ = forward(X, params, d, E)
    return accuracy(scores, y)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)

    n = len(X_all)
    val_size = n // 5
    train_size = n - val_size
    X_train, X_val = X_all[:train_size], X_all[train_size:]
    y_train, y_val = y_all[:train_size], y_all[train_size:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_E = None
    best_lr = None

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / train_size))

    for E in E_GRID:
        for lr in LR_GRID:
            params = init_params(d, E, C, rng)
            state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
            t = 0
            t = train_loop(X_train, y_train, params, state, lr, L2, BATCH, rng, t, epochs)
            acc = evaluate(X_val, y_val, params, d, E)
            if acc > best_acc:
                best_acc = acc
                best_params = [p.copy() for p in params]
                best_E = E
                best_lr = lr

    # Retrain on full data with best hyperparameters
    params = best_params
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    t = train_loop(X_all, y_all, params, state, best_lr, L2, BATCH, rng, t, epochs_full)

    # Write weights.bin
    E = best_E
    embs = params[:-2]
    W = params[-2]
    b = params[-1]
    out = struct.pack("<4i", MAGIC, d, E, C)
    for j in range(d):
        out += embs[j].astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emblinear inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte class index.
 * Weights format: magic(0x454D424C), d, E, C (int32 each), then
 *   embeddings[d][256][E] (float32, row-major per position),
 *   W[d*E][C] (float32, row-major),
 *   b[C] (float32). */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_emb = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(x_emb + j * E, src, E * sizeof(float));
        }

        // Linear layer
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int k = 0; k < d * E; k++) {
            float xv = x_emb[k];
            const float *w = W + (long)k * C;
            for (int c = 0; c < C; c++) scores[c] += xv * w[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```