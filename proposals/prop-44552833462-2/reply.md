```hypothesis
We propose a factorized linear model ("label-emb") that replaces the full one-hot linear layer with per-position embeddings of dimension D followed by a shared linear layer. This reduces parameters from ~266k (label-linear) to ~82k (D=16) and acts as a regularized low-rank approximation, which should generalize better on the small tinyshakespeare training set (24k examples). Inference requires a small matrix-vector product (16*D x 65) that remains fast. If validation accuracy on the tuning split does not exceed label-linear's 0.3375, or if instruction count per item exceeds label-linear significantly, the approach fails.
```

```python
"""Factorized linear model (label-emb) for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Trains per-position embeddings (dim D) + linear layer to C classes.
Chooses D on last 20% of train.in (tuning split), then retrains on all data.
Exports weights: magic 0x454D4244, d, D, C; float32 embeddings[d][256][D], W[(d*D)][C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS_TUNE = 10
EPOCHS_FULL = 30
D_CANDIDATES = [4, 8, 16, 32]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, d, C, D, epochs, rng):
    # embeddings: list of d arrays (256, D)
    E = [(rng.uniform(-1, 1, (256, D)) * math.sqrt(6.0 / 256)).astype(np.float32) for _ in range(d)]
    W = (rng.uniform(-1, 1, (d * D, C)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [p for e in E for p in [e]] + [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)

            # forward
            emb = np.concatenate([E[j][xb[:, j]] for j in range(d)], axis=1)  # (B, d*D)
            logits = emb @ W + b   # (B, C)
            # softmax + cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = (probs / len(yb)).astype(np.float32)  # (B, C)

            # gradients
            grad_W = emb.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_emb = grad_logits @ W.T  # (B, d*D)
            grad_E = []
            for j in range(d):
                g = grad_emb[:, j*D:(j+1)*D]  # (B, D)
                # accumulate gradient for each embedding row
                ge = np.zeros((256, D), np.float32)
                np.add.at(ge, xb[:, j], g)
                ge += L2 * E[j]
                grad_E.append(ge)

            grads = grad_E + [grad_W, grad_b]

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    return E, W, b


def evaluate(E, W, b, X, y, d, D, C):
    emb = np.concatenate([E[j][X[:, j]] for j in range(d)], axis=1)
    logits = emb @ W + b
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)

    n_all = X_all.shape[0]
    split = int(0.8 * n_all)  # first 80% train, last 20% tune
    X_train, y_train = X_all[:split], y_all[:split]
    X_tune, y_tune = X_all[split:], y_all[split:]

    rng = np.random.default_rng(SEED)
    best_D, best_acc = D_CANDIDATES[0], -1.0

    for D in D_CANDIDATES:
        E, W, b = train_model(X_train, y_train, d, C, D, EPOCHS_TUNE, rng)
        acc = evaluate(E, W, b, X_tune, y_tune, d, D, C)
        if acc > best_acc:
            best_acc, best_D = acc, D

    # retrain on all data with best_D
    rng = np.random.default_rng(SEED)  # reset seed for reproducibility
    E, W, b = train_model(X_all, y_all, d, C, best_D, EPOCHS_FULL, rng)

    # pack weights
    # magic 0x454D4244, d, D, C
    out = struct.pack("<4i", 0x454D4244, d, best_D, C)
    for j in range(d):
        out += E[j].astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorized linear model (label-emb) inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: one byte class index.
 * Weights format: magic 0x454D4244, int32 d, D, C; float32 embeddings[d][256][D], W[(d*D)][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)d * 256 * D;
    const float *b = W + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    float *emb = malloc(sizeof(float) * d * D);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // copy bias
        memcpy(scores, b, C * sizeof(float));

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = E + ((long)j * 256 + xi[j]) * D;
            memcpy(emb + j * D, e, D * sizeof(float));
        }

        // scores += emb @ W
        for (int row = 0; row < d * D; row++) {
            float v = emb[row];
            if (v == 0.0f) continue;
            const float *wrow = W + (long)row * C;
            for (int c = 0; c < C; c++) scores[c] += v * wrow[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(scores); free(emb); if (x) free(x);
    return 0;
}
```