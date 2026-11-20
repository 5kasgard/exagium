```hypothesis
The label-embmlp family uses per-position byte embeddings (size E) summed into a vector, followed by a ReLU hidden layer (size H) and a linear output layer. This decouples embedding capacity from hidden size, enabling richer context representations than label-linear (where embedding size equals hidden size) while staying fast at inference (two small matmuls). On tinyshakespeare next-char prediction, character interactions are non-linear; a hidden layer should capture them. If dev accuracy does not exceed label-linear's 0.3375, the added complexity is not justified.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameter grid to search on the tuning split (last 20% of train.in)
GRID = [
    {"E": 16, "H": 32, "lr": 1e-3, "l2": 1e-4, "batch": 64, "epochs": 10},
    {"E": 32, "H": 64, "lr": 1e-3, "l2": 1e-4, "batch": 64, "epochs": 10},
    {"E": 32, "H": 128, "lr": 1e-3, "l2": 1e-4, "batch": 64, "epochs": 10},
    {"E": 64, "H": 128, "lr": 1e-3, "l2": 1e-4, "batch": 64, "epochs": 10},
]
FINAL_EPOCHS = 30
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(d, E, H, C, rng):
    # embeddings: d x 256 x E
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 * E))).astype(np.float32)
    # W1: E x H, b1: H
    W1 = (rng.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / E)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: H x C, b2: C
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return emb, W1, b1, W2, b2

def forward(X, emb, W1, b1, W2, b2):
    n, d = X.shape
    # sum embeddings
    emb_sum = np.zeros((n, emb.shape[2]), dtype=np.float32)
    for j in range(d):
        emb_sum += emb[j, X[:, j]]
    # hidden
    hidden = emb_sum @ W1 + b1
    hidden = np.maximum(hidden, 0)
    # logits
    logits = hidden @ W2 + b2
    return emb_sum, hidden, logits

def softmax_grad(logits, y):
    n = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return probs / n

def backward(X, y, emb_sum, hidden, grad_logits, emb, W1, W2):
    n, d = X.shape
    E, H = W1.shape
    C = W2.shape[1]
    # grad W2, b2
    grad_W2 = hidden.T @ grad_logits
    grad_b2 = grad_logits.sum(axis=0)
    # grad hidden
    grad_hidden = grad_logits @ W2.T
    grad_hidden *= (hidden > 0).astype(np.float32)
    # grad W1, b1
    grad_W1 = emb_sum.T @ grad_hidden
    grad_b1 = grad_hidden.sum(axis=0)
    # grad embeddings
    grad_emb_sum = grad_hidden @ W1.T  # n x E
    grad_emb = np.zeros_like(emb)
    for j in range(d):
        # accumulate gradients for each byte value
        idx = X[:, j]
        for v in range(256):
            mask = (idx == v)
            if mask.any():
                grad_emb[j, v] = grad_emb_sum[mask].sum(axis=0)
    return grad_emb, grad_W1, grad_b1, grad_W2, grad_b2

def train_model(X, y, X_val, y_val, cfg, d, C, rng, final=False):
    E, H = cfg["E"], cfg["H"]
    lr, l2, batch = cfg["lr"], cfg["l2"], cfg["batch"]
    epochs = FINAL_EPOCHS if final else cfg["epochs"]
    n = X.shape[0]
    emb, W1, b1, W2, b2 = init_params(d, E, H, C, rng)
    # Adam state
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    best_val_acc = -1
    best_params = None
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            xb, yb = X[idx], y[idx]
            emb_sum, hidden, logits = forward(xb, emb, W1, b1, W2, b2)
            grad_logits = softmax_grad(logits, yb)
            grad_emb, grad_W1, grad_b1, grad_W2, grad_b2 = backward(xb, yb, emb_sum, hidden, grad_logits, emb, W1, W2)
            # L2 regularization
            grad_emb += l2 * emb
            grad_W1 += l2 * W1
            grad_W2 += l2 * W2
            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        if X_val is not None:
            _, _, logits_val = forward(X_val, emb, W1, b1, W2, b2)
            pred = logits_val.argmax(axis=1)
            acc = (pred == y_val).mean()
            if acc > best_val_acc:
                best_val_acc = acc
                best_params = [p.copy() for p in params]
    if best_params is not None:
        return best_params
    return params

def evaluate(params, X, y, d, C):
    emb, W1, b1, W2, b2 = params
    _, _, logits = forward(X, emb, W1, b1, W2, b2)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def save_weights(params, path, d, C, cfg):
    emb, W1, b1, W2, b2 = params
    E, H = cfg["E"], cfg["H"]
    magic = 0x454D4231  # 'EMB1'
    header = struct.pack("<5i", magic, d, E, H, C)
    with open(path, "wb") as f:
        f.write(header)
        f.write(emb.astype("<f4").tobytes())
        f.write(W1.astype("<f4").tobytes())
        f.write(b1.astype("<f4").tobytes())
        f.write(W2.astype("<f4").tobytes())
        f.write(b2.astype("<f4").tobytes())

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n_all = X_all.shape[0]
    split = int(0.8 * n_all)
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_cfg = None
    best_params = None

    for cfg in GRID:
        params = train_model(X_train, y_train, X_tune, y_tune, cfg, d, C, rng)
        acc = evaluate(params, X_tune, y_tune, d, C)
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_params = params

    # Retrain on full data
    final_rng = np.random.default_rng(SEED)
    final_params = train_model(X_all, y_all, None, None, best_cfg, d, C, final_rng, final=True)
    save_weights(final_params, sys.argv[4], d, C, best_cfg)

if __name__ == "__main__":
    main()
```

```c
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
    if (nw < 20 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // zero emb_sum
        for (int e = 0; e < E; e++) emb_sum[e] = 0.0f;
        // sum embeddings
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) emb_sum[e] += eptr[e];
        }

        // hidden = ReLU(emb_sum @ W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int e = 0; e < E; e++) sum += emb_sum[e] * W1[(long)e * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // logits = hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(emb_sum); free(hidden); free(logits);
    return 0;
}
```