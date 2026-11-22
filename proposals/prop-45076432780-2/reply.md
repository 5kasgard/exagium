```hypothesis
Family: label-embedding-mlp. Learns a small embedding per (position, byte) then a ReLU hidden layer.
Why: Character prediction benefits from pairwise interactions between context positions; linear model cannot capture them.
Failure: If dev accuracy ≤ label-linear, the extra capacity overfits the 24k training items or interactions are not predictive.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid to search on the tuning split (last 20% of train)
GRID = [
    {"E": 8, "H": 32, "lr": 3e-3, "epochs": 30},
    {"E": 16, "H": 64, "lr": 3e-3, "epochs": 30},
    {"E": 32, "H": 128, "lr": 1e-3, "epochs": 30},
]
BATCH = 64
L2 = 1e-4
FULL_EPOCHS = 50

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_eval(X_tr, y_tr, X_va, y_va, cfg, rng):
    n, d = X_tr.shape
    C = 65
    E, H = cfg["E"], cfg["H"]
    # Embeddings: (d, 256, E)
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
    # Hidden: (d*E, H)
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output: (H, C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    lr = cfg["lr"]
    for _ in range(cfg["epochs"]):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            # Forward
            # Embedding lookup: (B, d, E)
            emb_out = emb[np.arange(d)[None, :], xb]  # (B, d, E)
            flat = emb_out.reshape(len(xb), -1)       # (B, d*E)
            hidden = np.maximum(flat @ W1 + b1, 0)    # (B, H)
            logits = hidden @ W2 + b2                 # (B, C)
            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = (probs / len(yb)).astype(np.float32)
            # Backprop
            grad_W2 = hidden.T @ grad_logits + L2 * W2
            grad_b2 = grad_logits.sum(0)
            grad_hidden = grad_logits @ W2.T
            grad_hidden[hidden == 0] = 0.0
            grad_W1 = flat.T @ grad_hidden + L2 * W1
            grad_b1 = grad_hidden.sum(0)
            grad_flat = grad_hidden @ W1.T            # (B, d*E)
            grad_emb = np.zeros_like(emb)
            # Scatter add into grad_emb
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_flat[:, j*E:(j+1)*E])
            grad_emb += L2 * emb
            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    # Validation accuracy
    emb_out = emb[np.arange(d)[None, :], X_va]
    flat = emb_out.reshape(len(X_va), -1)
    hidden = np.maximum(flat @ W1 + b1, 0)
    logits = hidden @ W2 + b2
    preds = logits.argmax(axis=1)
    acc = (preds == y_va).mean()
    return acc, params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    # Split last 20% for tuning
    n = len(X)
    split = n - n // 5
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    best_acc, best_params, best_cfg = -1, None, None
    for cfg in GRID:
        acc, params = train_eval(X_tr, y_tr, X_va, y_va, cfg, rng)
        if acc > best_acc:
            best_acc, best_params, best_cfg = acc, params, cfg
    # Retrain on full data with best config
    rng = np.random.default_rng(SEED)
    _, best_params = train_eval(X, y, X[:1], y[:1], best_cfg, rng)  # dummy eval, returns trained params
    emb, W1, b1, W2, b2 = best_params
    # Pack weights: magic, d, E, H, C
    E, H = best_cfg["E"], best_cfg["H"]
    magic = 0x454D424C  # "LBME" in little-endian
    header = struct.pack("<5i", magic, d, E, H, C)
    # Embeddings: (d, 256, E) -> flatten to (d*256*E)
    emb_flat = emb.transpose(0, 1, 2).reshape(-1).astype("<f4")
    W1_flat = W1.astype("<f4").ravel()
    b1_flat = b1.astype("<f4")
    W2_flat = W2.astype("<f4").ravel()
    b2_flat = b2.astype("<f4")
    out = header + emb_flat.tobytes() + W1_flat.tobytes() + b1_flat.tobytes() + W2_flat.tobytes() + b2_flat.tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }
    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Embedding lookup
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, src, E * sizeof(float));
        }
        // Hidden layer: ReLU(concat @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int idx = 0; idx < d * E; idx++) sum += concat[idx] * W1[(long)idx * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```