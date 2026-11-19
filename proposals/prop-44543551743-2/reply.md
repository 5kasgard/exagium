```hypothesis
Family: label-embed-mlp. Uses per-position byte embeddings (dim E=8) concatenated, then one ReLU hidden layer (H=32), then linear output. Embeddings drastically reduce parameters vs one-hot linear (41k vs 266k) while preserving positional information, enabling better generalization on the 16-byte context. If dev accuracy does not exceed label-linear's 0.3375, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

# Fixed hyperparameter grid for tuning
GRID = [
    {"E": 4, "H": 16, "lr": 1e-3},
    {"E": 4, "H": 32, "lr": 1e-3},
    {"E": 8, "H": 16, "lr": 1e-3},
    {"E": 8, "H": 32, "lr": 1e-3},
    {"E": 8, "H": 32, "lr": 3e-3},
]
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 20
SAMPLES = 600000
SEED = 1

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def train_model(X, y, X_val, y_val, E, H, lr, seed, epochs):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    C = 65
    # Initialize embeddings: d x 256 x E
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # Initialize MLP
    in_dim = d * E
    if H:
        W1 = (rng.normal(0, math.sqrt(2.0 / in_dim), (in_dim, H))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W = (rng.normal(0, math.sqrt(2.0 / in_dim), (in_dim, C))).astype(np.float32)
        b = np.zeros(C, np.float32)
        params = [emb, W, b]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            # Forward
            batch_size = xb.shape[0]
            concat = np.zeros((batch_size, in_dim), np.float32)
            for j in range(d):
                concat[:, j*E:(j+1)*E] = emb[j, xb[:, j]]
            if H:
                hidden = np.maximum(concat @ W1 + b1, 0)
                logits = hidden @ W2 + b2
            else:
                logits = concat @ W + b
            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(batch_size), yb] -= 1.0
            grad_logits = probs / batch_size
            # Backward
            if H:
                grad_W2 = hidden.T @ grad_logits + L2 * W2
                grad_b2 = grad_logits.sum(0)
                grad_hidden = grad_logits @ W2.T
                grad_hidden[hidden <= 0] = 0
                grad_W1 = concat.T @ grad_hidden + L2 * W1
                grad_b1 = grad_hidden.sum(0)
                grad_concat = grad_hidden @ W1.T
                # Embedding gradients
                grad_emb = np.zeros_like(emb)
                for j in range(d):
                    g = grad_concat[:, j*E:(j+1)*E]
                    np.add.at(grad_emb[j], xb[:, j], g)
                grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            else:
                grad_W = concat.T @ grad_logits + L2 * W
                grad_b = grad_logits.sum(0)
                grad_concat = grad_logits @ W.T
                grad_emb = np.zeros_like(emb)
                for j in range(d):
                    g = grad_concat[:, j*E:(j+1)*E]
                    np.add.at(grad_emb[j], xb[:, j], g)
                grads = [grad_emb, grad_W, grad_b]
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
    if X_val is not None:
        batch_size = X_val.shape[0]
        concat = np.zeros((batch_size, in_dim), np.float32)
        for j in range(d):
            concat[:, j*E:(j+1)*E] = emb[j, X_val[:, j]]
        if H:
            hidden = np.maximum(concat @ W1 + b1, 0)
            logits = hidden @ W2 + b2
        else:
            logits = concat @ W + b
        preds = logits.argmax(axis=1)
        acc = (preds == y_val).mean()
        return acc, params
    return None, params

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Tuning split: last 20%
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    best_acc = -1
    best_params = None
    best_cfg = None
    for cfg in GRID:
        acc, params = train_model(X_train, y_train, X_val, y_val, cfg["E"], cfg["H"], cfg["lr"], SEED, MAX_EPOCHS)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_cfg = cfg
    # Retrain on full data with best config
    print(f"Best config: {best_cfg}, val acc: {best_acc:.4f}", file=sys.stderr)
    _, final_params = train_model(X, y, None, None, best_cfg["E"], best_cfg["H"], best_cfg["lr"], SEED, MAX_EPOCHS)
    # Write weights
    E, H = best_cfg["E"], best_cfg["H"]
    emb = final_params[0]
    if H:
        W1, b1, W2, b2 = final_params[1], final_params[2], final_params[3], final_params[4]
    else:
        W, b = final_params[1], final_params[2]
    # Magic: 0x454D4244 (EMBD little endian)
    out = struct.pack("<5i", 0x454D4244, d, E, H, C)
    # Embeddings: d x 256 x E floats
    out += emb.astype("<f4").tobytes()
    if H:
        out += W1.astype("<f4").tobytes()
        out += b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    else:
        out += W.astype("<f4").tobytes()
        out += b.astype("<f4").tobytes()
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long in_dim = (long)d * E;
    long emb_size = (long)d * 256 * E;
    long expected_floats = emb_size + (H ? (in_dim * H + H + (long)H * C + C) : (in_dim * C + C));
    long expected_bytes = 20 + expected_floats * 4;
    if (nw != expected_bytes) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + emb_size;
    const float *b1 = W1 + (H ? in_dim * H : 0);
    const float *W2 = b1 + (H ? H : 0);
    const float *b2 = W2 + (H ? (long)H * C : 0);
    const float *W = W1;
    const float *b = b1;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * in_dim);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }
        if (H) {
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int idx = 0; idx < in_dim; idx++) {
                    sum += concat[idx] * W1[(long)idx * H + h];
                }
                hidden[h] = sum > 0.0f ? sum : 0.0f;
            }
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) {
                    sum += hidden[h] * W2[(long)h * C + c];
                }
                logits[c] = sum;
            }
        } else {
            for (int c = 0; c < C; c++) {
                float sum = b[c];
                for (int idx = 0; idx < in_dim; idx++) {
                    sum += concat[idx] * W[(long)idx * C + c];
                }
                logits[c] = sum;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```