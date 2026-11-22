```hypothesis
Family: label-emb (positional embeddings + linear output).  
Why: The 16-byte context is a sequence where nearby positions interact (e.g., "th" -> "e"). One-hot linear models cannot share statistical strength across characters; embeddings map each byte to a dense vector, capturing similarities (letters vs punctuation) and reducing parameters (16*256*E + 16*E*C vs 16*256*C). Position-specific output weights let the model weight recent positions more heavily. With E=32 the model has ~164k params (vs 266k for label-linear) and ~34k multiply-adds/item, so it should be faster and more accurate.  
Failure signal: dev accuracy ≤ 0.3375 or instructions/item ≥ label-linear's 7359.
```

```python
"""label-emb: positional embeddings + linear output for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameter E (embedding dim) chosen on last 20% of train.in, then retrain on all.
Format: magic 0x4C454D42, int32 d, E, C; float32 emb[d][256][E], W[d][E][C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
E_CANDIDATES = (16, 32, 64)  # tuned on validation split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def split_data(X, y, val_frac=0.2):
    n = X.shape[0]
    val_size = int(n * val_frac)
    idx = np.arange(n)
    # last 20% in file order = validation
    tr_idx, val_idx = idx[:-val_size], idx[-val_size:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def init_params(d, C, E, rng):
    emb = [rng.uniform(-0.1, 0.1, (256, E)).astype(np.float32) for _ in range(d)]
    W = [rng.uniform(-0.1, 0.1, (E, C)).astype(np.float32) for _ in range(d)]
    b = np.zeros(C, np.float32)
    return emb, W, b


def forward(X, emb, W, b):
    """X: (B, d) uint8 -> logits (B, C)"""
    B, d = X.shape
    C = b.shape[0]
    logits = np.zeros((B, C), np.float32)
    for j in range(d):
        e = emb[j][X[:, j]]           # (B, E)
        logits += e @ W[j]            # (B, C)
    logits += b
    return logits


def loss_and_grad(X, y, emb, W, b):
    B, d = X.shape
    C = b.shape[0]
    logits = forward(X, emb, W, b)
    # softmax + cross-entropy
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits, dtype=np.float32)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    probs /= B
    # gradients
    grad_emb = [np.zeros_like(e) for e in emb]
    grad_W = [np.zeros_like(w) for w in W]
    grad_b = probs.sum(axis=0).astype(np.float32)
    for j in range(d):
        e = emb[j][X[:, j]]           # (B, E)
        grad_W[j] = e.T @ probs       # (E, C)
        # grad_emb[j][x] += probs_i @ W[j].T
        # vectorized: for each unique byte in column j
        for byte in range(256):
            mask = (X[:, j] == byte)
            if np.any(mask):
                grad_emb[j][byte] = probs[mask].T @ W[j]   # (E,)
    # L2 regularization
    for j in range(d):
        grad_W[j] += L2 * W[j]
        grad_emb[j] += L2 * emb[j]
    loss = -np.log(np.clip(probs[np.arange(B), y] + 1, 1e-12, 1.0)).mean()
    return loss, grad_emb, grad_W, grad_b


def train_loop(X, y, emb, W, b, rng, epochs):
    n = X.shape[0]
    params = [p for pair in zip(emb, W) for p in pair] + [b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            _, ge, gW, gb = loss_and_grad(Xb, yb, emb, W, b)
            grads = [g for pair in zip(ge, gW) for g in pair] + [gb]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)


def eval_acc(X, y, emb, W, b):
    logits = forward(X, emb, W, b)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    rng = np.random.default_rng(SEED)

    # split for tuning
    X_tr, y_tr, X_val, y_val = split_data(X, y, 0.2)
    n_tr = X_tr.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))

    best_E, best_acc = E_CANDIDATES[0], -1.0
    for E in E_CANDIDATES:
        emb, W, b = init_params(d, C, E, rng)
        train_loop(X_tr, y_tr, emb, W, b, rng, epochs)
        acc = eval_acc(X_val, y_val, emb, W, b)
        if acc > best_acc:
            best_acc, best_E = acc, E

    # retrain on all data with best_E
    emb, W, b = init_params(d, C, best_E, rng)
    n_all = X.shape[0]
    epochs_all = min(MAX_EPOCHS, math.ceil(SAMPLES / n_all))
    train_loop(X, y, emb, W, b, rng, epochs_all)

    # write weights.bin
    magic = 0x4C454D42  # "LEMB"
    out = struct.pack("<4i", magic, d, best_E, C)
    for j in range(d):
        out += emb[j].astype("<f4").tobytes()
    for j in range(d):
        out += W[j].astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x4C454D42, int32 d, E, C; float32 emb[d][256][E], W[d][E][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x4C454D42) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256 || E < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    float *tmp = malloc(sizeof(float) * E);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) logits[c] = b[c];

        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            const float *Wj = W + (long)j * E * C;
            for (int c = 0; c < C; c++) {
                float sum = 0.0f;
                for (int u = 0; u < E; u++) sum += e[u] * Wj[(long)u * C + c];
                logits[c] += sum;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(logits); free(tmp);
    return 0;
}
```