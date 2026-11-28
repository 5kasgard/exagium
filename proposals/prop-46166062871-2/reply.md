```hypothesis
Family: label-embmlp (embedding MLP with tuned hyperparameters). 
Reason: Character prediction benefits from dense embeddings that capture lexical similarity (e.g., 'a'/'b' close) and a hidden layer for nonlinear n-gram interactions. Label-linear uses one-hot (4096-dim) with no hidden layer (33.75%); label-mlp16 likely uses a fixed small hidden layer on one-hot. This model learns per-position byte embeddings (size E), concatenates them (16*E), then applies a ReLU hidden layer (size H) and output. Hyperparameters (E, H, LR, epochs) are chosen on the last 20% tuning split, then retrained on all data. 
Failure signal: Dev accuracy ≤ 0.34 or inference cost > 2× label-linear.
```

```python
"""label-embmlp: Embedding + one-hidden-layer MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy. Tunes on last 20% of train.in."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid (will be evaluated on validation split)
GRID = [
    {"E": 16, "H": 64, "LR": 1e-3, "EPOCHS": 100},
    {"E": 32, "H": 128, "LR": 3e-3, "EPOCHS": 100},
    {"E": 64, "H": 256, "LR": 1e-3, "EPOCHS": 100},
    {"E": 32, "H": 64, "LR": 3e-3, "EPOCHS": 100},
    {"E": 16, "H": 128, "LR": 1e-3, "EPOCHS": 100},
]
SEED = 1
L2 = 1e-5
BATCH = 128
PATIENCE = 5  # early stopping patience


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return np.mean(np.argmax(logits, axis=1) == y)


def forward(X, params, E, H, C):
    """X: (n, 16) uint8. Returns logits (n, C)."""
    n = X.shape[0]
    embs, W1, b1, W2, b2 = params
    # embeddings: (16, 256, E) -> gather -> (n, 16, E) -> (n, 16*E)
    flat = np.empty((n, 16 * E), dtype=np.float32)
    for j in range(16):
        flat[:, j * E:(j + 1) * E] = embs[j][X[:, j]]
    h = flat @ W1 + b1
    h = np.maximum(h, 0, dtype=np.float32)  # ReLU
    return h @ W2 + b2


def train_model(X_tr, y_tr, X_val, y_val, E, H, LR, EPOCHS, rng):
    n, d = X_tr.shape
    C = 65
    # init embeddings per position
    embs = [rng.uniform(-0.1, 0.1, (256, E)).astype(np.float32) for _ in range(16)]
    W1 = rng.uniform(-0.1, 0.1, (16 * E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.uniform(-0.1, 0.1, (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [embs, W1, b1, W2, b2]

    # Adam state
    state = []
    for p in [W1, b1, W2, b2]:
        state.append([np.zeros_like(p), np.zeros_like(p)])
    for e in embs:
        state.append([np.zeros_like(e), np.zeros_like(e)])

    def sg(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits, dtype=np.float32)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return probs / len(yy)

    best_val = -1.0
    best_params = None
    wait = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            # forward
            flat = np.empty((len(xb), 16 * E), dtype=np.float32)
            for j in range(16):
                flat[:, j * E:(j + 1) * E] = embs[j][xb[:, j]]
            h = flat @ W1 + b1
            h_relu = np.maximum(h, 0, dtype=np.float32)
            logits = h_relu @ W2 + b2
            g = sg(logits, yb)  # (B, C)
            # backward
            gW2 = h_relu.T @ g
            gb2 = g.sum(0)
            gh = g @ W2.T
            gh[h <= 0] = 0.0
            gW1 = flat.T @ gh
            gb1 = gh.sum(0)
            # embedding grads
            gembs = [np.zeros((256, E), np.float32) for _ in range(16)]
            for j in range(16):
                gj = gh[:, j * E:(j + 1) * E]
                np.add.at(gembs[j], xb[:, j], gj)
            # Adam update
            t = epoch * (n // BATCH) + s // BATCH + 1
            grads_list = [gW1, gb1, gW2, gb2] + gembs
            for i, (p, gr) in enumerate(zip([W1, b1, W2, b2] + embs, grads_list)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8) - LR * L2 * p  # L2 via weight decay
        # validation
        val_logits = forward(X_val, params, E, H, C)
        val_acc = accuracy(val_logits, y_val)
        if val_acc > best_val:
            best_val = val_acc
            best_params = [e.copy() for e in embs] + [W1.copy(), b1.copy(), W2.copy(), b2.copy()]
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return best_params, best_val


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 16

    n = len(X)
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_overall = None
    best_acc = -1.0
    best_cfg = None

    for cfg in GRID:
        params, val_acc = train_model(X_tr, y_tr, X_val, y_val,
                                      cfg["E"], cfg["H"], cfg["LR"], cfg["EPOCHS"], rng)
        if val_acc > best_acc:
            best_acc = val_acc
            best_overall = params
            best_cfg = cfg
        print(f"Config {cfg}: val_acc={val_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    print(f"Best config: {best_cfg}, retraining on full data...", file=sys.stderr)
    final_params, _ = train_model(X, y, X_val, y_val,
                                  best_cfg["E"], best_cfg["H"], best_cfg["LR"], best_cfg["EPOCHS"], rng)

    embs, W1, b1, W2, b2 = final_params
    E, H = best_cfg["E"], best_cfg["H"]

    # Pack weights: magic, d, E, H, C, then embeddings[16][256][E], W1[(16*E),H], b1[H], W2[H,C], b2[C]
    magic = 0x454D424C  # 'LBME'
    out = struct.pack("<5i", magic, d, E, H, C)
    for j in range(16):
        out += embs[j].astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }

    size_t expected = 20
        + (size_t)16 * 256 * E * 4
        + (size_t)16 * E * H * 4
        + (size_t)H * 4
        + (size_t)H * C * 4
        + (size_t)C * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld expect %zu\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *emb = p; p += 16 * 256 * E;
    const float *W1 = p; p += 16 * E * H;
    const float *b1 = p; p += H;
    const float *W2 = p; p += H * C;
    const float *b2 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * 16 * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        // concatenate embeddings
        for (int j = 0; j < 16; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }

        // hidden = ReLU(concat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int f = 0; f < 16 * E; f++) sum += concat[f] * W1[(size_t)f * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // logits = hidden * W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(size_t)h * C + c];
            logits[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```