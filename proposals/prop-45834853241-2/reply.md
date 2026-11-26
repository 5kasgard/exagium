```hypothesis
Family: label-embed (factorized linear model with per-position byte embeddings).
Why: The linear model (label-linear) uses a full one-hot weight matrix of size 16*256*65 ≈ 266k parameters, which overfits on 24k samples. Replacing it with low-rank embeddings (16*256*D + 16*D*65) acts as structured regularization, capturing position-specific byte statistics while drastically reducing parameters. A small embedding dimension D=8 yields ~41k parameters, enough to model interactions via the linear layer on concatenated embeddings. Hyperparameters (D, lr, epochs) are chosen on the last 20% tuning split, then retrained on all data.
Failure indication: If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds label-linear significantly.
```

```python
"""label-embed: factorized linear model with per-position byte embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters tuned on last 20% of train.in (file order), then retrained on all.
Format: magic 0x454D4245, int32 d, D, C; float32 embeddings[d][256][D], V[d*D][C], bias[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
SAMPLES = 600000
MAX_EPOCHS = 100
BATCH = 64
L2 = 1e-4

# Hyperparameter grid for tuning split
D_GRID = [4, 8, 16, 32]
LR_GRID = [1e-3, 3e-3, 1e-2]
TUNE_EPOCHS = 20  # epochs for tuning (less than full)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, d, C, D, lr, epochs, rng, l2=L2, batch=BATCH):
    n = X.shape[0]
    # init embeddings: d x 256 x D
    E = (rng.normal(0, 0.01, (d, 256, D))).astype(np.float32)
    # init V: (d*D) x C
    V = (rng.normal(0, 0.01, (d * D, C))).astype(np.float32)
    bias = np.zeros(C, np.float32)
    params = [E, V, bias]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            # forward: embeddings lookup and concat
            emb = np.concatenate([E[j, xb[:, j]] for j in range(d)], axis=1)  # batch x (d*D)
            logits = emb @ V + bias
            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = (probs / len(yb)).astype(np.float32)
            # gradients
            grad_V = emb.T @ grad_logits + l2 * V
            grad_bias = grad_logits.sum(axis=0)
            grad_emb = grad_logits @ V.T  # batch x (d*D)
            # split grad_emb per position
            grad_E = np.zeros_like(E)
            for j in range(d):
                start = j * D
                end = (j + 1) * D
                g = grad_emb[:, start:end]
                # scatter-add into grad_E[j]
                np.add.at(grad_E[j], xb[:, j], g)
            grad_E += l2 * E
            grads = [grad_E, grad_V, grad_bias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return E, V, bias

def evaluate(X, y, E, V, bias, d, D, C):
    emb = np.concatenate([E[j, X[:, j]] for j in range(d)], axis=1)
    logits = emb @ V + bias
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # split: last 20% for tuning
    n_train = int(0.8 * n)
    idx = np.arange(n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    # hyperparameter search
    best_acc = -1
    best_cfg = None
    best_model = None
    for D in D_GRID:
        for lr in LR_GRID:
            rng = np.random.default_rng(SEED + D * 1000 + int(lr * 10000))
            E, V, bias = train_model(X[train_idx], y[train_idx], d, C, D, lr, TUNE_EPOCHS, rng)
            acc = evaluate(X[val_idx], y[val_idx], E, V, bias, d, D, C)
            if acc > best_acc:
                best_acc = acc
                best_cfg = (D, lr)
                best_model = (E.copy(), V.copy(), bias.copy())

    # retrain on full data with best config
    D, lr = best_cfg
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng = np.random.default_rng(SEED)
    E, V, bias = train_model(X, y, d, C, D, lr, epochs, rng)

    # write weights
    magic = 0x454D4245  # 'EMBE'
    out = struct.pack("<4i", magic, d, D, C)
    out += E.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index.
 * Weights format: magic 0x454D4245, int32 d, D, C; float32 E[d][256][D], V[d*D][C], bias[C]. */
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
    if (nw < 16 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *V = E + (long)d * 256 * D;
    const float *bias = V + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb = malloc(sizeof(float) * d * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *src = E + ((long)j * 256 + xi[j]) * D;
            float *dst = emb + j * D;
            memcpy(dst, src, D * sizeof(float));
        }

        // linear layer: scores = emb @ V + bias
        for (int c = 0; c < C; c++) scores[c] = bias[c];
        for (int p = 0; p < d * D; p++) {
            float ep = emb[p];
            if (ep == 0.0f) continue;
            const float *vc = V + (long)p * C;
            for (int c = 0; c < C; c++) scores[c] += ep * vc[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```