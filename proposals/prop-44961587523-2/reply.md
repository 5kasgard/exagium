```hypothesis
Family: label-embmlp (embedding + one-hidden-layer MLP). Each of the 16 input bytes is mapped to a learned E-dimensional embedding (shared across positions), concatenated to a 16*E vector, passed through a ReLU hidden layer of size H, then a linear output layer to 65 classes. Embeddings capture character similarities (e.g., 'a' and 'b' close) and reduce parameters vs one-hot linear. A small grid search over E,H on the tuning split (last 20% of train) selects hyperparameters, then full training. Expects to beat label-linear (0.3375) by modeling pairwise interactions; if dev accuracy ≤ 0.3375, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
GRID = [
    {"E": 8, "H": 32, "lr": 1e-3, "epochs": 30, "batch": 128},
    {"E": 16, "H": 64, "lr": 1e-3, "epochs": 30, "batch": 128},
    {"E": 32, "H": 128, "lr": 1e-3, "epochs": 30, "batch": 128},
    {"E": 16, "H": 64, "lr": 3e-3, "epochs": 30, "batch": 128},
    {"E": 32, "H": 64, "lr": 1e-3, "epochs": 30, "batch": 128},
]
L2 = 1e-5
EARLY_STOP_PATIENCE = 5

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(d, E, H, C, rng):
    # embeddings: 256 x E
    emb = (rng.normal(0, 0.01, (256, E))).astype(np.float32)
    # W1: (d*E) x H
    W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: H x C
    W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return emb, W1, b1, W2, b2

def forward(X, emb, W1, b1, W2, b2, d, E, H, C):
    n = X.shape[0]
    # Embedding lookup: (n, d, E) -> (n, d*E)
    emb_seq = emb[X]  # (n, d, E)
    flat = emb_seq.reshape(n, d * E)
    # Hidden layer
    hidden = flat @ W1 + b1
    np.maximum(hidden, 0, out=hidden)  # ReLU in-place
    # Output
    logits = hidden @ W2 + b2
    return logits, hidden, flat

def loss_and_grad(X, y, emb, W1, b1, W2, b2, d, E, H, C, L2):
    n = X.shape[0]
    logits, hidden, flat = forward(X, emb, W1, b1, W2, b2, d, E, H, C)
    # Softmax cross-entropy
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    # Gradient on logits
    grad_logits = probs
    grad_logits[np.arange(n), y] -= 1.0
    grad_logits /= n
    # Output layer gradients
    grad_W2 = hidden.T @ grad_logits + L2 * W2
    grad_b2 = grad_logits.sum(axis=0)
    # Hidden layer gradients
    grad_hidden = grad_logits @ W2.T
    grad_hidden[hidden == 0] = 0  # ReLU derivative
    # Input to hidden gradients
    grad_W1 = flat.T @ grad_hidden + L2 * W1
    grad_b1 = grad_hidden.sum(axis=0)
    # Embedding gradients
    grad_flat = grad_hidden @ W1.T  # (n, d*E)
    grad_emb_seq = grad_flat.reshape(n, d, E)  # (n, d, E)
    grad_emb = np.zeros((256, E), np.float32)
    np.add.at(grad_emb, X, grad_emb_seq)
    grad_emb += L2 * emb
    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
    return loss, grad_emb, grad_W1, grad_b1, grad_W2, grad_b2

def evaluate(X, y, emb, W1, b1, W2, b2, d, E, H, C):
    logits, _, _ = forward(X, emb, W1, b1, W2, b2, d, E, H, C)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def train_config(X_tr, y_tr, X_val, y_val, d, E, H, C, lr, epochs, batch, L2, seed):
    rng = np.random.default_rng(seed)
    emb, W1, b1, W2, b2 = init_params(d, E, H, C, rng)
    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W1, v_W1 = np.zeros_like(W1), np.zeros_like(W1)
    m_b1, v_b1 = np.zeros_like(b1), np.zeros_like(b1)
    m_W2, v_W2 = np.zeros_like(W2), np.zeros_like(W2)
    m_b2, v_b2 = np.zeros_like(b2), np.zeros_like(b2)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    n = X_tr.shape[0]
    best_val_acc = -1
    best_params = None
    patience = 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            Xb, yb = X_tr[idx], y_tr[idx]
            loss, g_emb, g_W1, g_b1, g_W2, g_b2 = loss_and_grad(Xb, yb, emb, W1, b1, W2, b2, d, E, H, C, L2)
            t += 1
            # Adam update for each param
            for param, grad, m, v in [
                (emb, g_emb, m_emb, v_emb),
                (W1, g_W1, m_W1, v_W1),
                (b1, g_b1, m_b1, v_b1),
                (W2, g_W2, m_W2, v_W2),
                (b2, g_b2, m_b2, v_b2),
            ]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)
        val_acc = evaluate(X_val, y_val, emb, W1, b1, W2, b2, d, E, H, C)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = (emb.copy(), W1.copy(), b1.copy(), W2.copy(), b2.copy())
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break
    return best_val_acc, best_params

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    best_config = None
    best_val_acc = -1
    best_params = None
    for i, cfg in enumerate(GRID):
        # Deterministic seed per config
        seed = SEED + i * 1000
        val_acc, params = train_config(X_tr, y_tr, X_val, y_val, d,
                                       cfg["E"], cfg["H"], C,
                                       cfg["lr"], cfg["epochs"], cfg["batch"], L2, seed)
        print(f"Config {i}: E={cfg['E']} H={cfg['H']} lr={cfg['lr']} val_acc={val_acc:.4f}", file=sys.stderr)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = cfg
            best_params = params

    print(f"Best config: E={best_config['E']} H={best_config['H']} lr={best_config['lr']} val_acc={best_val_acc:.4f}", file=sys.stderr)
    # Retrain on full data with best config
    cfg = best_config
    seed = SEED + 9999
    val_acc, params = train_config(X, y, X_val, y_val, d,  # reuse val for early stopping
                                   cfg["E"], cfg["H"], C,
                                   cfg["lr"], cfg["epochs"]*2, cfg["batch"], L2, seed)
    emb, W1, b1, W2, b2 = params
    E, H = cfg["E"], cfg["H"]
    # Pack weights: magic, d, E, H, C; then emb, W1, b1, W2, b2
    magic = 0x454D4244  # "EMBD"
    header = struct.pack("<5i", magic, d, E, H, C)
    out = header
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
    uint8_t *wb = slurp(argv[1], &nw);
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x454D4244) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20L + 4L * (256L * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(wb + 20);
    const float *W1 = emb + 256L * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    uint8_t *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Embedding lookup and concatenation
        float *fp = flat;
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * E;
            memcpy(fp, e, E * sizeof(float));
            fp += E;
        }
        // Hidden layer: flat (d*E) x W1 (d*E x H) -> hidden (H)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int k = 0; k < d * E; k++) sum += flat[k] * W1[(long)k * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;  // ReLU
        }
        // Output layer: hidden (H) x W2 (H x C) -> logits (C)
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        out_byte = best;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```