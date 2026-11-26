```hypothesis
Family: label-embedmlp — per-position byte embeddings + one ReLU hidden layer.
Why: The linear model uses 4096 one-hot features; learned embeddings (D=8) reduce to 128 dimensions while preserving positional information, and a hidden layer (H=64) captures non-linear interactions. This should improve accuracy over label-linear (0.3375) with similar inference cost (~10k ops/item). If validation accuracy on the tuning split does not exceed label-linear, the approach fails.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid for tuning split (last 20% of train.in)
GRID = [
    {"D": 4, "H": 32, "lr": 1e-3, "L2": 1e-4, "batch": 64, "epochs": 100},
    {"D": 8, "H": 64, "lr": 3e-3, "L2": 1e-4, "batch": 64, "epochs": 100},
    {"D": 8, "H": 32, "lr": 1e-3, "L2": 1e-4, "batch": 64, "epochs": 100},
]
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(d, D, H, C, rng):
    # embeddings: list of (256, D) per position
    E = [(rng.uniform(-1, 1, (256, D)) * math.sqrt(6.0 / 256)).astype(np.float32) for _ in range(d)]
    # hidden layer: (d*D, H)
    W1 = (rng.uniform(-1, 1, (d * D, H)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # output layer: (H, C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return E, W1, b1, W2, b2

def forward(X, E, W1, b1, W2, b2, d, D, H):
    n = X.shape[0]
    # concatenate embeddings
    emb = np.empty((n, d * D), np.float32)
    for j in range(d):
        emb[:, j*D:(j+1)*D] = E[j][X[:, j]]
    h = emb @ W1 + b1
    h = np.maximum(h, 0)
    logits = h @ W2 + b2
    return emb, h, logits

def loss_grad(logits, y):
    n = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return probs.astype(np.float32) / n

def evaluate(X, y, E, W1, b1, W2, b2, d, D, H):
    _, _, logits = forward(X, E, W1, b1, W2, b2, d, D, H)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def train_model(X, y, X_val, y_val, cfg, rng):
    d = X.shape[1]
    C = int(y.max()) + 1
    D, H = cfg["D"], cfg["H"]
    lr, L2, batch, epochs = cfg["lr"], cfg["L2"], cfg["batch"], cfg["epochs"]
    n = X.shape[0]
    E, W1, b1, W2, b2 = init_params(d, D, H, C, rng)
    # Adam state
    params = [W1, b1, W2, b2] + E
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(min(epochs, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            xb, yb = X[idx], y[idx]
            emb, h, logits = forward(xb, E, W1, b1, W2, b2, d, D, H)
            g = loss_grad(logits, yb)
            # backprop output layer
            gW2 = h.T @ g + L2 * W2
            gb2 = g.sum(0)
            gh = g @ W2.T * (h > 0)
            # backprop hidden layer
            gW1 = emb.T @ gh + L2 * W1
            gb1 = gh.sum(0)
            # backprop embeddings
            g_emb = gh @ W1.T  # (batch, d*D)
            gE = [np.zeros((256, D), np.float32) for _ in range(d)]
            for j in range(d):
                np.add.at(gE[j], xb[:, j], g_emb[:, j*D:(j+1)*D])
            grads = [gW1, gb1, gW2, gb2] + gE
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    val_acc = evaluate(X_val, y_val, E, W1, b1, W2, b2, d, D, H)
    return val_acc, (E, W1, b1, W2, b2)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(X)
    val_size = n // 5
    X_train, y_train = X[:n-val_size], y[:n-val_size]
    X_val, y_val = X[n-val_size:], y[n-val_size:]

    rng = np.random.default_rng(SEED)
    best_acc, best_cfg, best_params = -1, None, None
    for cfg in GRID:
        # deterministic training for each config
        cfg_rng = np.random.default_rng(SEED + hash(str(cfg)) % 2**32)
        acc, params = train_model(X_train, y_train, X_val, y_val, cfg, cfg_rng)
        if acc > best_acc:
            best_acc, best_cfg, best_params = acc, cfg, params

    # Retrain on full data with best config
    final_rng = np.random.default_rng(SEED)
    _, final_params = train_model(X, y, X[:1], y[:1], best_cfg, final_rng)  # dummy val
    E, W1, b1, W2, b2 = final_params

    # Write weights.bin
    # Magic: 0x454D424C ('LBME' little-endian)
    magic = 0x454D424C
    D, H = best_cfg["D"], best_cfg["H"]
    out = struct.pack("<5i", magic, d, D, H, C)
    for j in range(d):
        out += E[j].astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x454D424C) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20
        + (long)d * 256 * D * 4
        + (long)d * D * H * 4
        + (long)H * 4
        + (long)H * C * 4
        + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *ptr = (const float *)(hd + 5);
    // embeddings: d * 256 * D
    const float *E = ptr;
    ptr += (long)d * 256 * D;
    // W1: d*D * H
    const float *W1 = ptr;
    ptr += (long)d * D * H;
    // b1: H
    const float *b1 = ptr;
    ptr += H;
    // W2: H * C
    const float *W2 = ptr;
    ptr += (long)H * C;
    // b2: C
    const float *b2 = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb = malloc(sizeof(float) * d * D);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = E + (long)j * 256 * D + xi[j] * D;
            memcpy(emb + j * D, e, D * sizeof(float));
        }
        // hidden layer: emb @ W1 + b1, ReLU
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * D; v++) sum += emb[v] * W1[(long)v * H + u];
            hidden[u] = sum > 0 ? sum : 0;
        }
        // output layer: hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += hidden[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(emb); free(hidden); free(logits);
    if (x) free(x);
    free(wb);
    return 0;
}
```