```hypothesis
Family: label-emb-linear (low-rank factorized linear model with learned embeddings).
Why: Adult has high-cardinality categorical columns (e.g., native-country) where a full one-hot linear model overfits; a shared low-dimensional embedding per column acts as a regularized factorized weight matrix, capturing similarity between categories while keeping inference fast (two small dense matmuls per item).
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or inference cost exceeds label-linear by >2×, the extra parameters do not justify themselves.
```

```python
"""label-emb-linear: factorized linear model with per-column embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weight format (little-endian):
  int32 magic=0x4C424D45, d, E, C
  float32 T[d][256][E]   # embedding tables
  float32 b1[E]          # embedding bias
  float32 W_out[E][C]    # output weights
  float32 b_out[C]       # output bias
"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (fixed, except E chosen on tuning split)
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
E_CANDIDATES = [8, 16, 32]  # embedding sizes to try on tuning split
MAGIC = 0x4C424D45  # "EMBL"

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(d, E, C, rng):
    # T: d x 256 x E, initialized with Xavier uniform
    scale = math.sqrt(6.0 / (256 + E))
    T = rng.uniform(-scale, scale, (d, 256, E)).astype(np.float32)
    b1 = np.zeros(E, np.float32)
    W_out = rng.uniform(-scale, scale, (E, C)).astype(np.float32)
    b_out = np.zeros(C, np.float32)
    return T, b1, W_out, b_out

def forward(X, T, b1, W_out, b_out):
    # X: (n, d) uint8
    n, d = X.shape
    hidden = np.broadcast_to(b1, (n, E)).copy()  # (n, E)
    for j in range(d):
        hidden += T[j, X[:, j]]  # T[j] is (256, E), X[:,j] is (n,)
    logits = hidden @ W_out + b_out  # (n, C)
    return hidden, logits

def loss_and_grad(X, y, hidden, logits, T, b1, W_out, b_out, L2):
    n = X.shape[0]
    # softmax cross-entropy gradient
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n  # (n, C)

    dW_out = hidden.T @ probs + L2 * W_out  # (E, C)
    db_out = probs.sum(axis=0)              # (C,)
    dhidden = probs @ W_out.T                # (n, E)

    db1 = dhidden.sum(axis=0)                # (E,)
    dT = np.zeros_like(T)
    for j in range(X.shape[1]):
        np.add.at(dT[j], X[:, j], dhidden)
    dT += L2 * T
    dT /= n
    db1 /= n
    dW_out /= n
    db_out /= n

    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean() + 0.5 * L2 * (
        np.sum(T*T) + np.sum(W_out*W_out)
    )
    return loss, dT, db1, dW_out, db_out

def accuracy(X, y, T, b1, W_out, b_out):
    _, logits = forward(X, T, b1, W_out, b_out)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def train_split(X, y, E, epochs, rng):
    n, d = X.shape
    C = 2
    T, b1, W_out, b_out = init_params(d, E, C, rng)
    params = [T, b1, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            hidden, logits = forward(xb, T, b1, W_out, b_out)
            loss, dT, db1, dW_out, db_out = loss_and_grad(xb, yb, hidden, logits, T, b1, W_out, b_out, L2)
            grads = [dT, db1, dW_out, db_out]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return T, b1, W_out, b_out

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_all = X_all.shape[0]
    d = card["input"]["bytes"]
    C = card["classes"]

    # Tuning split: last 20%
    n_tune = n_all // 5
    X_train, y_train = X_all[:-n_tune], y_all[:-n_tune]
    X_tune, y_tune = X_all[-n_tune:], y_all[-n_tune:]

    rng_base = np.random.default_rng(SEED)
    best_E, best_acc = E_CANDIDATES[0], -1.0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))

    for E in E_CANDIDATES:
        rng = np.random.default_rng(SEED + E)  # deterministic per candidate
        T, b1, W_out, b_out = train_split(X_train, y_train, E, epochs, rng)
        acc = accuracy(X_tune, y_tune, T, b1, W_out, b_out)
        if acc > best_acc:
            best_acc, best_E = acc, E

    # Retrain on all data with best_E
    rng_final = np.random.default_rng(SEED)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n_all))
    T, b1, W_out, b_out = train_split(X_all, y_all, best_E, epochs_full, rng_final)

    # Write weights
    out = struct.pack("<4i", MAGIC, d, best_E, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-linear inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class index. */
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
    if (nw < 16 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d < 1 || E < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * E;
    const float *W_out = b1 + E;
    const float *b_out = W_out + (long)E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * E);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(hidden, b1, E * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *emb = T + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) hidden[e] += emb[e];
        }

        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int e = 0; e < E; e++) sum += hidden[e] * W_out[(long)e * C + c];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```