```hypothesis
We propose "label-embedding-linear": a factorized linear model where each of the 16 byte positions has a learned embedding of dimension E (tuned on the validation split). The embeddings for the observed bytes are summed into a single E-dimensional vector, which is then classified by a linear layer. This reduces parameters from d*256*C to d*256*E + E*C, shares statistical strength across byte values, and is faster at inference (E*d adds + E*C muls vs d*C adds). On this character-level language task, many bytes (e.g., letters) have similar predictive power, so embeddings should improve accuracy over the one-hot linear baseline. If dev accuracy does not exceed label-linear's 33.75%, the factorization is too restrictive for this task.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid for validation
E_CHOICES = [4, 8, 16]
LR_CHOICES = [0.1, 0.05, 0.01]
EPOCHS_GRID = 20
EPOCHS_FINAL = 30
BATCH = 64
L2 = 1e-5
MOMENTUM = 0.9
VAL_FRACTION = 0.2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    n = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return probs / n

def evaluate(X, y, emb, W, b):
    n = X.shape[0]
    d, E = emb.shape[0], emb.shape[2]
    h = np.zeros((n, E), dtype=np.float32)
    for j in range(d):
        h += emb[j, X[:, j]]
    logits = h @ W.T + b
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def train_model(X_tr, y_tr, X_val, y_val, E, lr, epochs, rng):
    n, d = X_tr.shape
    C = int(y_tr.max()) + 1
    # Initialize parameters
    emb = rng.normal(0, 0.01, (d, 256, E)).astype(np.float32)
    W = rng.normal(0, 0.01, (C, E)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    # Momentum buffers
    v_emb = np.zeros_like(emb)
    v_W = np.zeros_like(W)
    v_b = np.zeros_like(b)
    best_val = -1.0
    best_params = None
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            # Forward
            h = np.zeros((len(yb), E), dtype=np.float32)
            for j in range(d):
                h += emb[j, Xb[:, j]]
            logits = h @ W.T + b
            # Loss gradient
            dlogits = softmax_cross_entropy_grad(logits, yb)
            # Gradients
            db = dlogits.sum(axis=0)
            dW = dlogits.T @ h + L2 * W
            dh = dlogits @ W
            demb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(demb[j], Xb[:, j], dh)
            demb += L2 * emb
            # Momentum update
            v_emb = MOMENTUM * v_emb + demb
            v_W = MOMENTUM * v_W + dW
            v_b = MOMENTUM * v_b + db
            emb -= lr * v_emb
            W -= lr * v_W
            b -= lr * v_b
        val_acc = evaluate(X_val, y_val, emb, W, b)
        if val_acc > best_val:
            best_val = val_acc
            best_params = (emb.copy(), W.copy(), b.copy())
    return best_val, best_params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]
    val_size = int(n * VAL_FRACTION)
    idx = np.arange(n)
    # Use last 20% as validation (file order)
    tr_idx, val_idx = idx[:-val_size], idx[-val_size:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    rng = np.random.default_rng(SEED)
    best_overall = -1.0
    best_cfg = None
    best_params = None

    for E in E_CHOICES:
        for lr in LR_CHOICES:
            rng_config = np.random.default_rng(SEED + E*100 + int(lr*1000))
            val_acc, params = train_model(X_tr, y_tr, X_val, y_val, E, lr, EPOCHS_GRID, rng_config)
            print(f"E={E} lr={lr} val_acc={val_acc:.4f}", file=sys.stderr)
            if val_acc > best_overall:
                best_overall = val_acc
                best_cfg = (E, lr)
                best_params = params

    print(f"Best: E={best_cfg[0]} lr={best_cfg[1]} val_acc={best_overall:.4f}", file=sys.stderr)
    # Retrain on all data
    E_final, lr_final = best_cfg
    rng_final = np.random.default_rng(SEED + 999)
    _, final_params = train_model(X, y, X[:1], y[:1], E_final, lr_final, EPOCHS_FINAL, rng_final)
    emb, W, b = final_params

    # Write weights.bin
    magic = 0x454D424C  # "L B M E" in little-endian
    out = struct.pack("<4i", magic, d, E_final, C)
    out += emb.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16 || E < 1 || E > 64) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)C * E + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)C * E;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = (float *)malloc(sizeof(float) * E);
    float *scores = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, sizeof(float) * E);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int e_idx = 0; e_idx < E; e_idx++) h[e_idx] += e[e_idx];
        }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            const float *w = W + (long)c * E;
            for (int e_idx = 0; e_idx < E; e_idx++) sum += h[e_idx] * w[e_idx];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```