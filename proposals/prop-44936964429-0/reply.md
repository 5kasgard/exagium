```hypothesis
Family: label-emb-linear — linear classifier on top of learned per-feature embeddings (size E=4).  
Why: The adult dataset has 8 categorical and 6 ordinal (255-bin) features. One-hot encoding (label-linear) creates a wide sparse input; dense embeddings share statistical strength across categories/bins and reduce parameter count. A linear model on 14×4=56 dimensions is far smaller than label-mlp16 (which adds a hidden layer) and should train faster while capturing non-linear ordinal effects via embedding lookup.  
Failure signal: dev accuracy ≤ 0.8524 (label-linear) or instructions/item ≥ 719 (label-linear cost).
```

```python
"""label-emb-linear: linear classifier on learned per-feature embeddings.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Format: magic 0x454D424C, int32 d, E, C; float32 embeddings[d][256][E]; float32 W[d*E][C]; float32 b[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid searched on the last 20% of train.in (tuning split)
GRID = [
    {"E": 2, "lr": 1e-3, "l2": 1e-4, "batch": 64, "samples": 500000},
    {"E": 4, "lr": 1e-3, "l2": 1e-4, "batch": 64, "samples": 500000},
    {"E": 8, "lr": 1e-3, "l2": 1e-4, "batch": 64, "samples": 500000},
]
SEED = 1
MAX_EPOCHS = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, idx_tr, idx_va, hp, rng):
    n_tr = len(idx_tr)
    d = X.shape[1]
    C = 2
    E = hp["E"]
    lr = hp["lr"]
    l2 = hp["l2"]
    batch = hp["batch"]
    target_samples = hp["samples"]
    epochs = min(MAX_EPOCHS, math.ceil(target_samples / n_tr))

    # Parameters
    emb = [rng.normal(0, 1/math.sqrt(E), (256, E)).astype(np.float32) for _ in range(d)]
    W = rng.normal(0, 1/math.sqrt(d*E), (d*E, C)).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    params = emb + [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, batch):
            idx = idx_tr[order[s:s+batch]]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward: gather embeddings
            cat = np.empty((B, d*E), np.float32)
            for j in range(d):
                cat[:, j*E:(j+1)*E] = emb[j][xb[:, j]]

            logits = cat @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = probs / B

            # Gradients
            grad_W = cat.T @ grad_logits + l2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_cat = grad_logits @ W.T

            # Scatter gradients back to embeddings
            grad_emb = [np.zeros_like(e) for e in emb]
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_cat[:, j*E:(j+1)*E])
            for j in range(d):
                grad_emb[j] += l2 * emb[j]

            grads = grad_emb + [grad_W, grad_b]

            # Adam update
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * g
                v *= 0.999; v += 0.001 * g * g
                m_hat = m / (1 - 0.9**t)
                v_hat = v / (1 - 0.999**t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Validation accuracy
    cat_va = np.empty((len(idx_va), d*E), np.float32)
    for j in range(d):
        cat_va[:, j*E:(j+1)*E] = emb[j][X[idx_va, j]]
    logits_va = cat_va @ W + b
    preds = logits_va.argmax(axis=1)
    acc = (preds == y[idx_va]).mean()
    return acc, emb, W, b


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(y)
    split = int(0.8 * n)
    idx_tr, idx_va = np.arange(split), np.arange(split, n)

    rng = np.random.default_rng(SEED)
    best_acc, best_hp, best_params = -1, None, None
    for hp in GRID:
        acc, emb, W, b = train_model(X, y, idx_tr, idx_va, hp, np.random.default_rng(SEED))
        if acc > best_acc:
            best_acc, best_hp, best_params = acc, hp, (emb, W, b)

    # Retrain on full data with best hyperparameters
    rng = np.random.default_rng(SEED)
    _, emb, W, b = train_model(X, y, np.arange(n), np.array([], dtype=int), best_hp, rng)

    d, E, C = X.shape[1], best_hp["E"], card["classes"]
    out = struct.pack("<4i", 0x454D424C, d, E, C)
    for j in range(d):
        out += emb[j].astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-linear inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class index (0 or 1).
 * Weights format: magic 0x454D424C, int32 d, E, C; float32 emb[d][256][E]; float32 W[d*E][C]; float32 b[C]. */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C != 2 || d != 14) { fprintf(stderr, "unexpected d or C\n"); return 3; }
    long expected = 16L + 4L * ( (long)d * 256 * E + (long)d * E * C + C );
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = calloc(d * E, sizeof(float));
    float logits[2];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(acc, 0, d * E * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) acc[j*E + e] = src[e];
        }
        logits[0] = b[0]; logits[1] = b[1];
        for (int k = 0; k < d*E; k++) {
            float a = acc[k];
            logits[0] += a * W[k*2 + 0];
            logits[1] += a * W[k*2 + 1];
        }
        unsigned char best = (logits[1] > logits[0]) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    free(acc);
    return 0;
}
```