```hypothesis
Propose label-fm: a factorization machine with class-specific linear weights and shared low-rank embeddings (rank k=4) for pairwise interactions. The adult dataset has many categorical features where interactions (e.g., education × occupation) are known to be predictive; FM captures these efficiently. If dev accuracy does not exceed label-linear (0.8524) by at least 0.005, the hypothesis is falsified.
```

```python
"""Factorization Machine family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: int32 magic 0x464D4C31, d, k, C; float32 w[C][d][256], b[C], e[d][256][k]."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (tuned on last 20% of train.in)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
K_GRID = [2, 4, 8]  # embedding ranks to try

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_fm(X, y, C, d, k, rng, n_epochs):
    """Train FM on (X,y) for n_epochs, return (w,b,e)."""
    n = X.shape[0]
    # Initialize parameters
    w = (rng.uniform(-0.01, 0.01, (C, d, 256))).astype(np.float32)
    b = np.zeros(C, np.float32)
    e = (rng.uniform(-0.01, 0.01, (d, 256, k))).astype(np.float32)
    params = [w, b, e]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d) uint8
            yb = y[idx]          # (B,)
            B = len(idx)

            # Forward: compute scores (B, C)
            # Interaction term (shared)
            sum_e = np.zeros((B, k), np.float32)
            sum_sq = np.zeros((B, k), np.float32)
            for j in range(d):
                v = xb[:, j]
                emb = e[j, v, :]          # (B, k)
                sum_e += emb
                sum_sq += emb * emb
            inter = 0.5 * (np.sum(sum_e * sum_e, axis=1) - np.sum(sum_sq, axis=1))  # (B,)

            scores = np.zeros((B, C), np.float32)
            for c in range(C):
                lin = b[c] + np.sum(w[c, np.arange(d), xb.T], axis=0)
                scores[:, c] = lin + inter

            # Softmax cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            G = probs / B                       # (B, C)
            H = G.sum(axis=1)                   # (B,) gradient w.r.t. interaction

            # Gradients
            grad_w = np.zeros_like(w)
            grad_b = np.zeros_like(b)
            grad_e = np.zeros_like(e)

            # Linear part gradients
            for c in range(C):
                grad_b[c] = G[:, c].sum()
                for j in range(d):
                    v = xb[:, j]
                    np.add.at(grad_w[c, j], v, G[:, c])

            # Embedding gradients
            for j in range(d):
                v = xb[:, j]
                for f in range(k):
                    ef = e[j, :, f]                 # (256,)
                    sf = sum_e[:, f]                # (B,)
                    # dL/de[j,v,f] = sum_i H[i] * (sf[i] - ef[v_i]) * I(v_i==v)
                    # We can accumulate by iterating over batch
                    for i in range(B):
                        vi = v[i]
                        grad_e[j, vi, f] += H[i] * (sf[i] - ef[vi])

            # L2 regularization
            grad_w += L2 * w
            grad_e += L2 * e

            grads = [grad_w, grad_b, grad_e]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v_ = state[i]
                m *= 0.9; m += 0.1 * gr
                v_ *= 0.999; v_ += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v_ / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    return w, b, e

def evaluate(w, b, e, X, y, C, d, k):
    n = X.shape[0]
    correct = 0
    for s in range(0, n, 1024):
        xb = X[s:s+1024]
        yb = y[s:s+1024]
        B = len(xb)
        sum_e = np.zeros((B, k), np.float32)
        sum_sq = np.zeros((B, k), np.float32)
        for j in range(d):
            v = xb[:, j]
            emb = e[j, v, :]
            sum_e += emb
            sum_sq += emb * emb
        inter = 0.5 * (np.sum(sum_e * sum_e, axis=1) - np.sum(sum_sq, axis=1))
        scores = np.zeros((B, C), np.float32)
        for c in range(C):
            lin = b[c] + np.sum(w[c, np.arange(d), xb.T], axis=0)
            scores[:, c] = lin + inter
        pred = scores.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]

    # Split: last 20% for tuning
    n = X.shape[0]
    split = int(0.8 * n)
    X_train, X_tune = X[:split], X[split:]
    y_train, y_tune = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_k = K_GRID[0]
    best_params = None

    # Grid search over k
    for k in K_GRID:
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
        w, b, e = train_fm(X_train, y_train, C, d, k, rng, epochs)
        acc = evaluate(w, b, e, X_tune, y_tune, C, d, k)
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_params = (w.copy(), b.copy(), e.copy())

    # Retrain on full train with best k
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    w, b, e = train_fm(X, y, C, d, best_k, rng, epochs)

    # Write weights.bin
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, best_k, C)
    out += w.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    out += e.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || k < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)C * d * 256 + C + (long)d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *w = (const float *)(hd + 4);
    const float *b = w + (long)C * d * 256;
    const float *e = b + C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_e = malloc(sizeof(float) * k);
    float *sum_sq = malloc(sizeof(float) * k);
    float *score = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }

        // Linear part
        for (int c = 0; c < C; c++) score[c] = b[c];
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            for (int c = 0; c < C; c++) {
                score[c] += w[((long)c * d + j) * 256 + v];
            }
        }

        // Interaction term
        memset(sum_e, 0, k * sizeof(float));
        memset(sum_sq, 0, k * sizeof(float));
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *emb = e + ((long)j * 256 + v) * k;
            for (int f = 0; f < k; f++) {
                float val = emb[f];
                sum_e[f] += val;
                sum_sq[f] += val * val;
            }
        }
        float inter = 0.0f;
        for (int f = 0; f < k; f++) inter += sum_e[f] * sum_e[f] - sum_sq[f];
        inter *= 0.5f;
        for (int c = 0; c < C; c++) score[c] += inter;

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```