```hypothesis
Family: label-emb-mlp (embedding + small MLP).
Why: The 16-byte categorical input has 4096 one-hot dimensions in label-linear; embeddings compress this to 16×D (D≈8) and a hidden layer (H≈32) captures nonlinear interactions, boosting accuracy with fewer parameters and faster inference than label-gbdt.
Failure: If dev accuracy ≤ 0.3375 (label-linear) or inference cost per item exceeds label-linear.
```

```python
"""label-emb-mlp: embedding + one ReLU layer for 16-byte categorical input.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid (tuned on last 20% of train.in)
SEED = 1
CONFIGS = [
    {"D": 8, "H": 0, "lr": 1e-3, "batch": 64, "epochs": 40},
    {"D": 8, "H": 32, "lr": 1e-3, "batch": 64, "epochs": 40},
    {"D": 16, "H": 0, "lr": 1e-3, "batch": 64, "epochs": 40},
    {"D": 16, "H": 32, "lr": 1e-3, "batch": 64, "epochs": 40},
]
L2 = 1e-5
VAL_FRAC = 0.2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    """Return gradient of cross-entropy w.r.t logits (already averaged over batch)."""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def train_eval(config, X_tr, y_tr, X_val, y_val):
    rng = np.random.default_rng(SEED)
    n, d = X_tr.shape
    C = 65
    D, H = config["D"], config["H"]
    # Initialize embeddings: (d, 256, D)
    emb = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / (256 + D))).astype(np.float32)
    if H:
        W1 = (rng.uniform(-1, 1, (d * D, H)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W1 = (rng.uniform(-1, 1, (d * D, C)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
        b1 = np.zeros(C, np.float32)
        params = [emb, W1, b1]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    batch, lr = config["batch"], config["lr"]
    epochs = config["epochs"]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X_tr[idx], y[idx]
            # Forward
            # Embedding lookup: (batch, d, D)
            e = emb[xb]  # xb is (batch, d) of bytes
            flat = e.reshape(batch, d * D)
            if H:
                h = np.maximum(flat @ W1 + b1, 0, dtype=np.float32)
                logits = h @ W2 + b2
            else:
                logits = flat @ W1 + b1
            # Gradient
            g = softmax_cross_entropy_grad(logits, yb)
            if H:
                gW2 = h.T @ g
                gb2 = g.sum(0)
                gh = g @ W2.T
                gh[h == 0] = 0
                gW1 = flat.T @ gh
                gb1 = gh.sum(0)
                # Embedding gradient
                gflat = gh @ W1.T  # (batch, d*D)
                gemb = np.zeros_like(emb)
                # scatter-add: gflat[:, j*D:(j+1)*D] -> gemb[xb[:, j]]
                for j in range(d):
                    np.add.at(gemb[:, :, j], xb[:, j], gflat[:, j*D:(j+1)*D])
                grads = [gemb, gW1, gb1, gW2, gb2]
            else:
                gW1 = flat.T @ g
                gb1 = g.sum(0)
                gflat = g @ W1.T
                gemb = np.zeros_like(emb)
                for j in range(d):
                    np.add.at(gemb[:, :, j], xb[:, j], gflat[:, j*D:(j+1)*D])
                grads = [gemb, gW1, gb1]
            # L2 on weights (not embeddings)
            for i, p in enumerate(params):
                if i != 0:  # skip embeddings for L2
                    grads[i] += L2 * p
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    # Evaluate on validation
    nv = X_val.shape[0]
    e_val = emb[X_val]
    flat_val = e_val.reshape(nv, d * D)
    if H:
        h_val = np.maximum(flat_val @ W1 + b1, 0)
        logits_val = h_val @ W2 + b2
    else:
        logits_val = flat_val @ W1 + b1
    pred = logits_val.argmax(axis=1)
    acc = (pred == y_val).mean()
    return acc, params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    # Split: last 20% for validation
    n = len(X)
    split = int(n * (1 - VAL_FRAC))
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    # Grid search
    best_acc = -1
    best_params = None
    best_cfg = None
    for cfg in CONFIGS:
        acc, params = train_eval(cfg, X_tr, y_tr, X_val, y_val)
        print(f"Config {cfg}: val_acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_cfg = cfg
    print(f"Best config: {best_cfg}, val_acc={best_acc:.4f}", file=sys.stderr)
    # Retrain on full data with best config
    rng = np.random.default_rng(SEED)
    D, H = best_cfg["D"], best_cfg["H"]
    # Re-initialize with same seed for reproducibility
    emb = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / (256 + D))).astype(np.float32)
    if H:
        W1 = (rng.uniform(-1, 1, (d * D, H)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W1 = (rng.uniform(-1, 1, (d * D, C)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
        b1 = np.zeros(C, np.float32)
        params = [emb, W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    batch, lr = best_cfg["batch"], best_cfg["lr"]
    epochs = best_cfg["epochs"]
    n_full = len(X)
    for _ in range(epochs):
        order = rng.permutation(n_full)
        for s in range(0, n_full, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            e = emb[xb]
            flat = e.reshape(batch, d * D)
            if H:
                h = np.maximum(flat @ W1 + b1, 0, dtype=np.float32)
                logits = h @ W2 + b2
            else:
                logits = flat @ W1 + b1
            g = softmax_cross_entropy_grad(logits, yb)
            if H:
                gW2 = h.T @ g
                gb2 = g.sum(0)
                gh = g @ W2.T
                gh[h == 0] = 0
                gW1 = flat.T @ gh
                gb1 = gh.sum(0)
                gflat = gh @ W1.T
                gemb = np.zeros_like(emb)
                for j in range(d):
                    np.add.at(gemb[:, :, j], xb[:, j], gflat[:, j*D:(j+1)*D])
                grads = [gemb, gW1, gb1, gW2, gb2]
            else:
                gW1 = flat.T @ g
                gb1 = g.sum(0)
                gflat = g @ W1.T
                gemb = np.zeros_like(emb)
                for j in range(d):
                    np.add.at(gemb[:, :, j], xb[:, j], gflat[:, j*D:(j+1)*D])
                grads = [gemb, gW1, gb1]
            for i, p in enumerate(params):
                if i != 0:
                    grads[i] += L2 * p
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    # Pack weights: magic, d, D, H, C
    magic = 0x454D4244  # "EMBD"
    out = struct.pack("<5i", magic, d, D, H, C)
    # Embeddings: d * 256 * D floats (row-major: position, byte, dim)
    out += params[0].transpose(0, 1, 2).astype("<f4").tobytes()  # (d, 256, D) -> flat
    if H:
        # W1: (d*D, H), b1: H, W2: (H, C), b2: C
        out += params[1].astype("<f4").tobytes()
        out += params[2].astype("<f4").tobytes()
        out += params[3].astype("<f4").tobytes()
        out += params[4].astype("<f4").tobytes()
    else:
        # W1: (d*D, C), b1: C
        out += params[1].astype("<f4").tobytes()
        out += params[2].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index (0..64). */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || D > 64 || H < 0 || H > 256) {
        fprintf(stderr, "bad header values\n"); return 3;
    }
    size_t emb_size = (size_t)d * 256 * D;
    size_t expected = 20 + 4 * emb_size;
    if (H) {
        expected += 4 * ((size_t)d * D * H + H + (size_t)H * C + C);
    } else {
        expected += 4 * ((size_t)d * D * C + C);
    }
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + emb_size;
    const float *b1 = W1 + (H ? (size_t)d * D * H : (size_t)d * D * C);
    const float *W2 = b1 + (H ? H : C);
    const float *b2 = W2 + (size_t)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *output = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        if (H) {
            for (int h = 0; h < H; h++) hidden[h] = b1[h];
            for (int j = 0; j < d; j++) {
                const float *e = emb + ((size_t)j * 256 + xi[j]) * D;
                for (int kk = 0; kk < D; kk++) {
                    float ev = e[kk];
                    for (int h = 0; h < H; h++) {
                        hidden[h] += ev * W1[((size_t)j * D + kk) * H + h];
                    }
                }
            }
            for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;
            for (int c = 0; c < C; c++) output[c] = b2[c];
            for (int h = 0; h < H; h++) {
                float hv = hidden[h];
                for (int c = 0; c < C; c++) output[c] += hv * W2[(size_t)h * C + c];
            }
        } else {
            for (int c = 0; c < C; c++) output[c] = b1[c];
            for (int j = 0; j < d; j++) {
                const float *e = emb + ((size_t)j * 256 + xi[j]) * D;
                for (int kk = 0; kk < D; kk++) {
                    float ev = e[kk];
                    for (int c = 0; c < C; c++) {
                        output[c] += ev * W1[((size_t)j * D + kk) * C + c];
                    }
                }
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (output[c] > output[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```