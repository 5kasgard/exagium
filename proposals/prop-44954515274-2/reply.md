```hypothesis
Family: label-mlp-tuned (one-hidden-layer MLP with ReLU, hidden size H chosen on tuning split).
Why: The linear model (H=0) cannot capture interactions between context bytes; a hidden layer can. The existing label-mlp16 uses fixed H=16 which is too small for this task. Tuning H up to 512 should find a better capacity trade-off.
Failure: If best H=0 or dev accuracy ≤ 0.3375 (label-linear), the hypothesis is wrong.
```

```python
"""Table family with tuned hidden size for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each byte position as one-hot over training values. Trains a one-hidden-layer MLP
with ReLU (H hidden units, H=0 means linear) using Adam. Chooses H on the last 20% of train.in,
then retrains on all data. Exports in the same format as label-linear (magic 0x54424C31)."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (fixed except H which is tuned)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
H_CANDIDATES = [0, 16, 32, 64, 128, 256, 512]  # include 0 (linear) and 16 (label-mlp16)


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoders(X, types):
    """Return list of encoder matrices (256 x num_vals) for each column."""
    enc = []
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None].astype(np.float32))
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def encode(X, encoders):
    """Concatenate one-hot (or ordinal) encodings for each column."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(encoders)], axis=1).astype(np.float32)


def softmax_grad(logits, y):
    """Gradient of cross-entropy w.r.t. logits."""
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return (probs / len(y)).astype(np.float32)


def train_model(Z, y, H, C, epochs, batch, lr, l2, rng):
    """Train and return parameters (W1, b1, W2, b2). If H=0, W2,b2 are None."""
    n, k = Z.shape
    W = H if H else C
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), np.float32)
        b1 = np.zeros(C, np.float32)
        params = [W1, b1]

    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z[idx], y[idx]
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                g = softmax_grad(h @ W2 + b2, yy)
                gh = (g @ W2.T) * (a > 0)
                grads = [z.T @ gh + l2 * W1, gh.sum(0), h.T @ g + l2 * W2, g.sum(0)]
            else:
                g = softmax_grad(z @ W1 + b1, yy)
                grads = [z.T @ g + l2 * W1, g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params


def evaluate(Z, y, params, H):
    """Return accuracy."""
    W1, b1 = params[0], params[1]
    if H:
        W2, b2 = params[2], params[3]
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
    else:
        logits = Z @ W1 + b1
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def write_weights(encoders, params, H, C, d, out_path):
    """Write weights.bin in the label-linear format."""
    W = H if H else C
    W1, b1 = params[0], params[1]
    # Build T tables: for each position, 256 x W
    T_list = []
    off = 0
    for E in encoders:
        num_vals = E.shape[1]
        T_list.append((E @ W1[off:off + num_vals]).astype(np.float32))
        off += num_vals
    T = np.stack(T_list)  # d x 256 x W
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        W2, b2 = params[2], params[3]
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    open(out_path, "wb").write(out)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    types = card["input"]["types"]

    n = len(X)
    n_tune = n // 5
    n_train = n - n_tune
    X_train, y_train = X[:n_train], y[:n_train]
    X_tune, y_tune = X[n_train:], y[n_train:]

    # Tune H on the tuning split
    best_H, best_acc = 0, -1.0
    for H in H_CANDIDATES:
        rng = np.random.default_rng(SEED + H)
        encoders = build_encoders(X_train, types)
        Z_train = encode(X_train, encoders)
        Z_tune = encode(X_tune, encoders)
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))
        params = train_model(Z_train, y_train, H, C, epochs, BATCH, LR, L2, rng)
        acc = evaluate(Z_tune, y_tune, params, H)
        print(f"H={H:4d}  tune_acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc, best_H = acc, H

    print(f"Selected H={best_H} with tune_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best_H
    rng = np.random.default_rng(SEED)
    encoders = build_encoders(X, types)
    Z = encode(X, encoders)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    params = train_model(Z, y, best_H, C, epochs, BATCH, LR, L2, rng)
    write_weights(encoders, params, best_H, C, d, sys.argv[4])


if __name__ == "__main__":
    main()
```

```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```