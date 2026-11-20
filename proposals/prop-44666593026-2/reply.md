```hypothesis
Family: label-embed-mlp. A shallow MLP with per-position byte embeddings (size H) summed, then a ReLU hidden layer of size H, then output. Hyperparameters (H, learning rate) chosen on the tuning split (last 20% of train). Expects to beat label-linear by capturing pairwise interactions via the hidden layer. If validation accuracy does not exceed label-linear by >0.5%, the added complexity is not justified.
```

```python
"""label-embed-mlp: per-position embeddings summed -> ReLU hidden layer -> output.
Hyperparameters (H, lr) tuned on last 20% of train.in (file order), then retrained on all train.
Deterministic (seed 1), single-threaded, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning
H_CHOICES = [32, 64, 128]
LR_CHOICES = [1e-3, 3e-3, 1e-2]
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
SEED = 1
MAGIC = 0x454D4C50  # "EMLP"

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def encode_features(X, types):
    """Return (enc_list, Z) where enc_list[j] is (256, nvals_j) and Z is (n, sum nvals_j)."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j]
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None].astype(np.float32)
        else:
            vals = np.unique(v)
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
        enc.append(E)
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    return enc, Z

def softmax_cross_entropy_grad(logits, y):
    """Return gradient of softmax cross-entropy w.r.t logits, shape (n, C)."""
    n = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return probs / n

def train_model(Z, y, H, lr, l2, batch, samples, max_epochs, seed, Z_val=None, y_val=None):
    """Train on (Z, y), optionally validate on (Z_val, y_val). Return (params, val_acc)."""
    n, k = Z.shape
    C = int(y.max()) + 1
    rng = np.random.default_rng(seed)

    # He initialization for embeddings (W1) and hidden/output layers
    W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(max_epochs, math.ceil(samples / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z[idx], y[idx]

            # Forward
            a = z @ W1 + b1
            h = np.maximum(a, 0)
            logits = h @ W2 + b2

            # Backward
            g = softmax_cross_entropy_grad(logits, yy)  # (batch, C)
            gh = g @ W2.T * (a > 0)                     # (batch, H)

            grads = [
                z.T @ gh + l2 * W1,                     # dW1
                gh.sum(axis=0),                         # db1
                h.T @ g + l2 * W2,                      # dW2
                g.sum(axis=0),                          # db2
            ]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    val_acc = 0.0
    if Z_val is not None:
        a_val = Z_val @ W1 + b1
        h_val = np.maximum(a_val, 0)
        logits_val = h_val @ W2 + b2
        preds = logits_val.argmax(axis=1)
        val_acc = (preds == y_val).mean()

    return params, val_acc

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]

    # Split: last 20% for validation (file order)
    n = len(X)
    val_size = n // 5
    train_size = n - val_size
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Encode using training-set unique values
    enc, Z_train = encode_features(X_train, types)
    Z_val = np.concatenate([enc[j][X_val[:, j]] for j in range(d)], axis=1).astype(np.float32)

    best_val_acc = -1.0
    best_params = None
    best_H = None
    best_lr = None

    for H in H_CHOICES:
        for lr in LR_CHOICES:
            params, val_acc = train_model(Z_train, y_train, H, lr, L2, BATCH, SAMPLES, MAX_EPOCHS, SEED, Z_val, y_val)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = params
                best_H = H
                best_lr = lr

    # Retrain on full data with best hyperparameters
    enc_full, Z_full = encode_features(X, types)
    params, _ = train_model(Z_full, y, best_H, best_lr, L2, BATCH, SAMPLES, MAX_EPOCHS, SEED)
    W1, b1, W2, b2 = params

    # Build per-position lookup tables T (d, 256, H)
    T = []
    off = 0
    for j, E in enumerate(enc_full):
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    T = np.stack(T)  # (d, 256, H)

    # Write weights.bin
    out = struct.pack("<4i", MAGIC, d, best_H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic(0x454D4C50), d, H, C; then float32 T[d][256][H], b1[H], W2[H][C], b2[C]. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    if (nw < 16 || hd[0] != 0x454D4C50) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int h = 0; h < H; h++) acc[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += t[h];
        }

        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = acc[h] > 0.0f ? acc[h] : 0.0f;
            if (hv == 0.0f) continue;
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```