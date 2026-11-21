```hypothesis
Family: label-emb-mlp — per-position character embeddings (size E) + one ReLU hidden layer (size H) + linear output.
Why: The 16-byte context is a sequence of categorical tokens; embeddings capture token similarity and reduce input dimension from 16×256 to 16×E, allowing a smaller H for the same capacity. The inference path folds embeddings into the same per-position lookup tables as label-linear, so CPU cost is identical to label-linear with H>0. We expect higher accuracy than label-linear (0.3375) with H≤64.
Failure: If dev accuracy ≤0.34 with H≤64 after tuning E∈{8,16,32}, the embeddings don't help enough.
```

```python
"""label-emb-mlp: per-position embeddings + one ReLU layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only. Tunes E,H on last 20% of train, then trains on all."""
import json, math, struct, sys
import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
EMB_GRID = [8, 16, 32]
HID_GRID = [32, 64, 128]
TUNE_FRAC = 0.2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(d, C, E, H, rng):
    # embeddings: (d, 256, E)
    emb = (rng.normal(0, 1.0 / math.sqrt(E), (d, 256, E))).astype(np.float32)
    # W1: (d*E, H)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E + H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: (H, C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / (H + C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return emb, W1, b1, W2, b2

def forward(emb, W1, b1, W2, b2, X, y, H):
    n, d = X.shape
    # X: (n, d) uint8
    # embed: (n, d, E)
    e = emb[np.arange(d), X]          # (n, d, E)
    e = e.reshape(n, -1)              # (n, d*E)
    a = e @ W1 + b1                   # (n, H)
    h = np.maximum(a, 0)
    logits = h @ W2 + b2              # (n, C)
    # softmax + cross-entropy
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
    # L2 on all params except biases
    l2 = L2 * (np.sum(emb**2) + np.sum(W1**2) + np.sum(W2**2))
    return loss + l2, probs, e, h, a

def backward(e, h, a, probs, y, W2, emb, W1, X, d, E, H):
    n = X.shape[0]
    # dlogits = probs - onehot
    dlogits = probs.copy()
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n
    # dW2, db2
    dW2 = h.T @ dlogits + 2 * L2 * W2
    db2 = dlogits.sum(0)
    # dh
    dh = dlogits @ W2.T
    # dReLU
    da = dh * (a > 0)
    # dW1, db1
    dW1 = e.T @ da + 2 * L2 * W1
    db1 = da.sum(0)
    # de (n, d*E)
    de = da @ W1.T
    # demb (d, 256, E)
    demb = np.zeros_like(emb)
    for j in range(d):
        # de[:, j*E:(j+1)*E] corresponds to position j
        ej = de[:, j*E:(j+1)*E]
        Xj = X[:, j]
        # accumulate
        np.add.at(demb[j], Xj, ej)
    demb += 2 * L2 * emb
    return demb, dW1, db1, dW2, db2

def evaluate(emb, W1, b1, W2, b2, X, y):
    n = X.shape[0]
    e = emb[np.arange(X.shape[1]), X].reshape(n, -1)
    h = np.maximum(e @ W1 + b1, 0)
    logits = h @ W2 + b2
    preds = logits.argmax(1)
    return (preds == y).mean()

def train_split(emb, W1, b1, W2, b2, X, y, rng, steps):
    n = X.shape[0]
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    d, E = emb.shape[0], emb.shape[2]
    H = W1.shape[1]
    for _ in range(steps):
        idx = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch = idx[s:s+BATCH]
            Xb, yb = X[batch], y[batch]
            loss, probs, e, h, a = forward(emb, W1, b1, W2, b2, Xb, yb, H)
            demb, dW1, db1, dW2, db2 = backward(e, h, a, probs, yb, W2, emb, W1, Xb, d, E, H)
            grads = [demb, dW1, db1, dW2, db2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]
    tune_n = int(n * TUNE_FRAC)
    train_n = n - tune_n
    X_train, y_train = X[:train_n], y[:train_n]
    X_tune, y_tune = X[train_n:], y[train_n:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_params = None
    best_E, best_H = None, None

    # hyperparameter search
    for E in EMB_GRID:
        for H in HID_GRID:
            emb, W1, b1, W2, b2 = init_params(d, C, E, H, rng)
            # quick train: min(MAX_EPOCHS, ceil(SAMPLES/train_n)) epochs
            epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / train_n))
            train_split(emb, W1, b1, W2, b2, X_train, y_train, rng, epochs)
            acc = evaluate(emb, W1, b1, W2, b2, X_tune, y_tune)
            if acc > best_acc:
                best_acc = acc
                best_E, best_H = E, H
                best_params = (emb.copy(), W1.copy(), b1.copy(), W2.copy(), b2.copy())

    # retrain on all data with best hyperparameters
    emb, W1, b1, W2, b2 = best_params
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    train_split(emb, W1, b1, W2, b2, X, y, rng, epochs)

    # fold embeddings into table T: (d, 256, H)
    T = np.zeros((d, 256, best_H), np.float32)
    for j in range(d):
        T[j] = emb[j] @ W1[j*best_E:(j+1)*best_E]

    # write weights: magic, d, H, C, E, T, b1, W2, b2
    magic = 0x454D424C  # 'L','B','M','E' little-endian
    out = struct.pack("<5i", magic, d, best_H, C, best_E)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Weights format: int32 magic(0x454D424C), d, H, C, E; float32 T[d][256][H], b1[H], W2[H][C], b2[C].
 * Each input record is d bytes; output is one byte (class index). */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], E = hd[4];
    if (C < 1 || C > 256 || d != 16 || H < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: %ld != %ld\n", nw, expected); return 3; }
    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *sc  = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int h = 0; h < H; h++) acc[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += t[h];
        }
        for (int c = 0; c < C; c++) sc[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = acc[h] > 0.0f ? acc[h] : 0.0f;
            if (hv == 0.0f) continue;
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) sc[c] += hv * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```