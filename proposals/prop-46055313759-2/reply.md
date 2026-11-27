```hypothesis
Family: label-tiedmlp (tied-embedding MLP). Uses a shared embedding table for all 16 context positions, sums the embeddings, then passes through one ReLU hidden layer to the 65-class output. Tied embeddings drastically reduce parameters vs. per-position tables, forcing the model to learn position-invariant byte representations while still capturing order via the sum. Should beat label-linear by modeling non-linear interactions at similar inference cost. Failure: if validation accuracy on the tuning split does not exceed label-linear's 0.3375.
```

```python
"""label-tiedmlp: tied-embedding MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid for tuning split (last 20% of train.in)
GRID = [
    {"E": 16, "H": 64, "lr": 3e-3},
    {"E": 32, "H": 64, "lr": 3e-3},
    {"E": 32, "H": 32, "lr": 3e-3},
    {"E": 64, "H": 64, "lr": 1e-3},
]
SEED = 1
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
TUNE_EPOCHS = 10  # epochs for each grid search run


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def split_data(X, y, tune_frac=0.2):
    n = X.shape[0]
    tune_n = int(n * tune_frac)
    return X[:-tune_n], y[:-tune_n], X[-tune_n:], y[-tune_n:]


def train_model(X, y, X_val, y_val, C, d, E, H, lr, epochs, rng):
    n, k = X.shape  # k == d
    # Parameters
    emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / E)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            # Forward
            # Sum embeddings for the 16 positions
            h = np.zeros((len(xb), E), np.float32)
            for p in range(d):
                h += emb[xb[:, p]]
            a = h @ W1 + b1
            h_relu = np.maximum(a, 0)
            logits = h_relu @ W2 + b2
            # Loss gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            dlogits = (probs / len(yb)).astype(np.float32)
            # Backward
            dW2 = h_relu.T @ dlogits + L2 * W2
            db2 = dlogits.sum(0)
            dh_relu = dlogits @ W2.T
            da = dh_relu * (a > 0)
            dW1 = h.T @ da + L2 * W1
            db1 = da.sum(0)
            dh = da @ W1.T
            # Embedding gradients
            demb = np.zeros_like(emb)
            for p in range(d):
                np.add.at(demb, xb[:, p], dh)
            demb += L2 * emb
            grads = [demb, dW1, db1, dW2, db2]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    # Validation accuracy
    h_val = np.zeros((X_val.shape[0], E), np.float32)
    for p in range(d):
        h_val += emb[X_val[:, p]]
    a_val = h_val @ W1 + b1
    h_relu_val = np.maximum(a_val, 0)
    logits_val = h_relu_val @ W2 + b2
    preds = logits_val.argmax(axis=1)
    acc = (preds == y_val).mean()
    return acc, params


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)

    # Split tuning set (last 20% in file order)
    X_tr, y_tr, X_val, y_val = split_data(X_all, y_all, 0.2)

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_cfg = None

    for cfg in GRID:
        rng_cfg = np.random.default_rng(SEED + hash(str(cfg)) % 2**32)
        acc, params = train_model(X_tr, y_tr, X_val, y_val, C, d,
                                  cfg["E"], cfg["H"], cfg["lr"], TUNE_EPOCHS, rng_cfg)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_cfg = cfg

    # Retrain on full data with best config
    rng_full = np.random.default_rng(SEED + 12345)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / X_all.shape[0]))
    _, final_params = train_model(X_all, y_all, X_val[:1], y_val[:1], C, d,
                                  best_cfg["E"], best_cfg["H"], best_cfg["lr"],
                                  epochs_full, rng_full)

    emb, W1, b1, W2, b2 = final_params
    E, H = best_cfg["E"], best_cfg["H"]

    # Write weights.bin
    # Magic: 0x5449454D ("TIEM" little-endian), d, E, H, C
    magic = 0x5449454D
    header = struct.pack("<5i", magic, d, E, H, C)
    out = bytearray()
    out += header
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-tiedmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 20 || hd[0] != 0x5449454D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20L + 4L * ((long)256 * E + (long)E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + 256L * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = (float *)malloc(sizeof(float) * E);
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, sizeof(float) * E);
        for (int p = 0; p < d; p++) {
            const float *e = emb + (long)xi[p] * E;
            for (int j = 0; j < E; j++) h[j] += e[j];
        }
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int j = 0; j < E; j++) sum += h[j] * W1[(long)j * H + u];
            hidden[u] = sum > 0.0f ? sum : 0.0f;
        }
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += hidden[u] * W2[(long)u * C + c];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```