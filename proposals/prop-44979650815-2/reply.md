```hypothesis
We propose a new family "label-mlp-tuned" that extends the table family with a single ReLU hidden layer and selects its size (32, 64, 128) and learning rate on the tuning split (last 20% of train). The one-hot linear model (label-linear) cannot capture interactions between the 16 context positions; a hidden layer enables such interactions. Tuning on the held-out split prevents overfitting and chooses a good capacity. If dev accuracy does not exceed label-linear's 0.3375, or if inference cost (Ir/item) exceeds label-linear by more than 2×, the hypothesis fails.
```

```python
"""Tuned MLP for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 42), single thread, stdlib + numpy.
Chooses hidden size and learning rate on tuning split (last 20% of train),
then trains on all train. Exports table-family weights (magic 0x54424C31)."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters
SEED = 42
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# Tuning grid
HIDDEN_SIZES = [32, 64, 128]
LEARNING_RATES = [1e-3, 3e-3]


def load_records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def prepare_xy(records_x, records_y, card):
    X = np.frombuffer(b"".join(records_x), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records_y], dtype=np.int64)
    return X, y


def build_encoders(X, card):
    """Return list of encoding matrices (256 x n_vals) for each position."""
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:  # categorical
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def encode(X, enc):
    """Apply encoders to X (n x d) -> Z (n x total_dim)."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def softmax_grad(logits, y):
    """Gradient of cross-entropy with softmax: (P - one_hot) / n."""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return (probs / len(y)).astype(np.float32)


def train_model(Z, y, enc, H, lr, l2, batch, max_epochs, seed, C):
    """Train on Z, return params (W1, b1, W2, b2)."""
    n, k = Z.shape
    rng = np.random.default_rng(seed)
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
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z[idx], y[idx]
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                g = softmax_grad(h @ W2 + b2, yy)
                gh = (g @ W2.T) * (a > 0)
                grads = [
                    z.T @ gh + l2 * W1,
                    gh.sum(0),
                    h.T @ g + l2 * W2,
                    g.sum(0)
                ]
            else:
                g = softmax_grad(z @ W1 + b1, yy)
                grads = [
                    z.T @ g + l2 * W1,
                    g.sum(0)
                ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def evaluate(Z, y, params, H, C):
    """Return accuracy on Z, y."""
    if H:
        W1, b1, W2, b2 = params
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
    else:
        W1, b1 = params
        logits = Z @ W1 + b1
    preds = logits.argmax(axis=1)
    return (preds == y).mean()


def build_tables(enc, params, H):
    """Build per-position lookup tables T for inference."""
    W1 = params[0]
    off, T = 0, []
    for E in enc:
        T.append((E @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    return T


def write_weights(card, enc, params, H, out_path):
    d = card["input"]["bytes"]
    C = card["classes"]
    W = H if H else C
    T = build_tables(enc, params, H)
    b1 = params[1].astype("<f4")
    magic = 0x54424C31
    out = struct.pack("<4i", magic, d, H, C)
    out += np.stack(T).astype("<f4").tobytes()
    out += b1.tobytes()
    if H:
        W2 = params[2].astype("<f4")
        b2 = params[3].astype("<f4")
        out += W2.tobytes()
        out += b2.tobytes()
    open(out_path, "wb").write(out)


def main():
    card = json.load(open(sys.argv[1]))
    train_x = load_records(sys.argv[2])
    train_y = load_records(sys.argv[3])
    X_full, y_full = prepare_xy(train_x, train_y, card)
    n_full = len(X_full)
    # Split: last 20% for tuning
    split = int(0.8 * n_full)
    X_train, y_train = X_full[:split], y_full[:split]
    X_tune, y_tune = X_full[split:], y_full[split:]
    C = card["classes"]

    # Build encoders on training split only
    enc_train = build_encoders(X_train, card)
    Z_train = encode(X_train, enc_train)
    Z_tune = encode(X_tune, enc_train)

    n_train = len(X_train)
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))

    best_acc = -1.0
    best_cfg = None
    best_params = None

    # Grid search
    for idx, (H, lr) in enumerate([(h, lr) for h in HIDDEN_SIZES for lr in LEARNING_RATES]):
        cfg_seed = SEED + idx + 1
        params = train_model(Z_train, y_train, enc_train, H, lr, L2, BATCH, max_epochs, cfg_seed, C)
        acc = evaluate(Z_tune, y_tune, params, H, C)
        print(f"Tuning: H={H}, lr={lr:.0e} -> acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_cfg = (H, lr)
            best_params = params

    print(f"Best config: H={best_cfg[0]}, lr={best_cfg[1]:.0e}, tune_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    enc_full = build_encoders(X_full, card)
    Z_full = encode(X_full, enc_full)
    n_full = len(X_full)
    max_epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n_full))
    final_params = train_model(Z_full, y_full, enc_full, best_cfg[0], best_cfg[1], L2, BATCH, max_epochs_full, SEED, C)

    write_weights(card, enc_full, final_params, best_cfg[0], sys.argv[4])


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