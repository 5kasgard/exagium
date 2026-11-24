```hypothesis
Family: label-mlp-search – a one-hidden-layer MLP (ReLU) with per-position byte embeddings, where the hidden size is chosen by validation on the last 20% of the training data (the tuning split) and then retrained on the full set.
Why: Character-level language modeling requires capturing nonlinear interactions between context bytes (e.g., "th" → "e"). A linear model cannot do this; a small MLP adds modest compute while enabling such interactions. The tuning-split search picks a good capacity (H ∈ {0, 16, 32, 64, 128}) without overfitting.
Failure signal: If the best hidden size is 0 (i.e., linear) or the final dev accuracy does not exceed label-linear’s 33.75%, the extra capacity does not help on this task.
```

```python
"""label-mlp-search: one-hidden-layer MLP with tuning-split hidden-size selection.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (except H which is searched)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
H_OPTIONS = [0, 16, 32, 64, 128]  # 0 = linear baseline


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoders(X, types):
    """Return list of encoder matrices E_j (256 x nvals_j) for each position."""
    enc = []
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def encode(X, enc):
    """Transform X (n x d) to dense Z (n x k) using encoders."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def softmax_grad(logits, y):
    """Return gradient of cross-entropy w.r.t. logits (n x C)."""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return (probs / len(y)).astype(np.float32)


def forward(Z, params, H):
    """Compute logits from Z and params. Returns logits (n x C) and hidden pre-act (n x H) if H>0."""
    W1, b1 = params[0], params[1]
    if H == 0:
        return Z @ W1 + b1, None
    W2, b2 = params[2], params[3]
    a = Z @ W1 + b1
    h = np.maximum(a, 0)
    return h @ W2 + b2, a


def accuracy(logits, y):
    return np.mean(np.argmax(logits, axis=1) == y)


def train_model(Z_train, y_train, Z_val, y_val, H, C, epochs):
    """Train on Z_train, validate on Z_val. Return (params, val_acc)."""
    n, k = Z_train.shape
    rng = np.random.default_rng(SEED)
    # Initialize parameters
    if H == 0:
        W1 = np.zeros((k, C), np.float32)
        b1 = np.zeros(C, np.float32)
        params = [W1, b1]
    else:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_train[idx], y_train[idx]
            logits, a = forward(z, params, H)
            g = softmax_grad(logits, yy)
            if H == 0:
                grads = [z.T @ g + L2 * W1, g.sum(0)]
            else:
                W2 = params[2]
                gh = g @ W2.T
                gh *= (a > 0)  # ReLU derivative
                grads = [
                    z.T @ gh + L2 * W1,
                    gh.sum(0),
                    np.maximum(a, 0).T @ g + L2 * W2,
                    g.sum(0),
                ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    # Validation
    val_logits, _ = forward(Z_val, params, H)
    val_acc = accuracy(val_logits, y_val)
    return params, val_acc


def pack_weights(enc, params, H, C, d):
    """Serialize weights to the label-linear format (magic, d, H, C, T, b1, [W2, b2])."""
    W = H if H else C
    off, T = 0, []
    for E in enc:
        T.append((E @ params[0][off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += np.stack(T).astype("<f4").tobytes()
    out += params[1].astype("<f4").tobytes()  # b1
    if H:
        out += params[2].astype("<f4").tobytes()  # W2
        out += params[3].astype("<f4").tobytes()  # b2
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]

    # Split: last 20% for tuning
    n = len(X)
    n_train = int(0.8 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    # Build encoders on training split only
    enc = build_encoders(X_train, card["input"]["types"])
    Z_train = encode(X_train, enc)
    Z_val = encode(X_val, enc)

    # Hyperparameter search over H
    best_H, best_acc, best_params = 0, -1.0, None
    for H in H_OPTIONS:
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))
        params, val_acc = train_model(Z_train, y_train, Z_val, y_val, H, C, epochs)
        if val_acc > best_acc:
            best_acc, best_H, best_params = val_acc, H, params

    # Retrain on full data with best_H
    enc_full = build_encoders(X, card["input"]["types"])
    Z_full = encode(X, enc_full)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    final_params, _ = train_model(Z_full, y, Z_full, y, best_H, C, epochs_full)  # val unused

    # Write weights
    out = pack_weights(enc_full, final_params, best_H, C, d)
    open(sys.argv[4], "wb").write(out)


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