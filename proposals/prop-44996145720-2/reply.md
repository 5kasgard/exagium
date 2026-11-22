```hypothesis
Family: label-mlp2 (two-hidden-layer MLP on concatenated per-position byte embeddings).
Why: The linear family sums per-position embeddings, losing positional interactions. A small MLP on concatenated embeddings (E=8, H1=64, H2=32) captures n-gram features with ~45k parameters and ~12k FLOPs/item, likely exceeding linear's 33.75% accuracy at similar cost.
Failure: If tuning-split validation accuracy ≤ 0.34 or inference instructions/item > 2× linear.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed architecture
E = 8        # embedding size per position
H1 = 64      # first hidden layer
H2 = 32      # second hidden layer
SEED = 1
LR_CANDIDATES = [1e-3, 3e-3, 1e-2]
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
SAMPLES = 600000
VAL_FRACTION = 0.2  # last 20% for tuning

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(d, C, rng):
    # embeddings: d * 256 * E
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256))).astype(np.float32)
    # dense1: (d*E) -> H1
    W1 = (rng.uniform(-1, 1, (d * E, H1)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    # dense2: H1 -> H2
    W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    # output: H2 -> C
    W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
    b3 = np.zeros(C, np.float32)
    return [emb, W1, b1, W2, b2, W3, b3]

def forward(params, X, training=True):
    emb, W1, b1, W2, b2, W3, b3 = params
    n = X.shape[0]
    # embeddings: (n, d, E)
    emb_seq = emb[X, np.arange(emb.shape[0])]  # X is (n, d), emb is (d, 256, E) -> (n, d, E)
    flat = emb_seq.reshape(n, -1)  # (n, d*E)
    h1 = flat @ W1 + b1
    a1 = np.maximum(h1, 0)
    h2 = a1 @ W2 + b2
    a2 = np.maximum(h2, 0)
    logits = a2 @ W3 + b3
    if training:
        return logits, (flat, a1, a2)
    return logits, None

def backward(params, cache, y, logits, n):
    emb, W1, b1, W2, b2, W3, b3 = params
    flat, a1, a2 = cache
    # softmax gradient
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    grad_logits = probs / n  # (n, C)
    # output layer
    grad_W3 = a2.T @ grad_logits + L2 * W3
    grad_b3 = grad_logits.sum(0)
    grad_a2 = grad_logits @ W3.T
    # relu2
    grad_h2 = grad_a2 * (a2 > 0)
    grad_W2 = a1.T @ grad_h2 + L2 * W2
    grad_b2 = grad_h2.sum(0)
    grad_a1 = grad_h2 @ W2.T
    # relu1
    grad_h1 = grad_a1 * (a1 > 0)
    grad_W1 = flat.T @ grad_h1 + L2 * W1
    grad_b1 = grad_h1.sum(0)
    grad_flat = grad_h1 @ W1.T  # (n, d*E)
    # embeddings
    grad_emb = np.zeros_like(emb)
    grad_emb_seq = grad_flat.reshape(n, emb.shape[0], E)
    # scatter add: for each position j, for each example i, add grad_emb_seq[i, j] to grad_emb[j, X[i, j]]
    # We'll do a loop over positions (d=16 small)
    for j in range(emb.shape[0]):
        np.add.at(grad_emb[j], X[:, j], grad_emb_seq[:, j])
    grad_emb += L2 * emb
    return [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3]

def evaluate(params, X, y):
    logits, _ = forward(params, X, training=False)
    pred = np.argmax(logits, axis=1)
    return (pred == y).mean()

def train_loop(params, X, y, X_val, y_val, lr, epochs, rng):
    n = X.shape[0]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    best_val = -1
    best_params = None
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            logits, cache = forward(params, xb)
            grads = backward(params, cache, yb, logits, len(yb))
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        val_acc = evaluate(params, X_val, y_val)
        if val_acc > best_val:
            best_val = val_acc
            best_params = [p.copy() for p in params]
    return best_params, best_val

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X_all.shape[0]
    val_size = int(n * VAL_FRACTION)
    train_size = n - val_size
    X_train, X_val = X_all[:train_size], X_all[train_size:]
    y_train, y_val = y_all[:train_size], y_all[train_size:]

    rng = np.random.default_rng(SEED)
    best_lr = None
    best_val_acc = -1
    best_final_params = None

    for lr in LR_CANDIDATES:
        params = init_params(d, C, rng)
        trained_params, val_acc = train_loop(params, X_train, y_train, X_val, y_val, lr, MAX_EPOCHS, rng)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_lr = lr
            best_final_params = trained_params

    # Retrain on full data with best lr
    rng = np.random.default_rng(SEED)
    params = init_params(d, C, rng)
    full_params, _ = train_loop(params, X_all, y_all, X_val, y_val, best_lr, MAX_EPOCHS, rng)

    # Pack weights
    emb, W1, b1, W2, b2, W3, b3 = full_params
    # Magic: 0x4D4C5032 ("MLP2")
    out = struct.pack("<5i", 0x4D4C5032, d, E, H1, H2, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    out += W3.astype("<f4").tobytes()
    out += b3.astype("<f4").tobytes()
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
    if (nw < 20 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H1 = hd[3], H2 = hd[4], C = hd[5];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights config\n"); return 3; }
    long expected = 20
        + 4L * d * 256 * E
        + 4L * d * E * H1 + 4L * H1
        + 4L * H1 * H2 + 4L * H2
        + 4L * H2 * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *emb = p; p += d * 256 * E;
    const float *W1 = p; p += d * E * H1;
    const float *b1 = p; p += H1;
    const float *W2 = p; p += H1 * H2;
    const float *b2 = p; p += H2;
    const float *W3 = p; p += H2 * C;
    const float *b3 = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *h1_buf = malloc(sizeof(float) * H1);
    float *h2_buf = malloc(sizeof(float) * H2);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // embed
        for (int j = 0; j < d; j++) {
            const float *e = emb + (j * 256 + xi[j]) * E;
            memcpy(flat + j * E, e, E * sizeof(float));
        }
        // dense1 + ReLU
        for (int u = 0; u < H1; u++) {
            float sum = b1[u];
            for (int v = 0; v < d * E; v++) sum += flat[v] * W1[v * H1 + u];
            h1_buf[u] = sum > 0 ? sum : 0;
        }
        // dense2 + ReLU
        for (int u = 0; u < H2; u++) {
            float sum = b2[u];
            for (int v = 0; v < H1; v++) sum += h1_buf[v] * W2[v * H2 + u];
            h2_buf[u] = sum > 0 ? sum : 0;
        }
        // output
        for (int c = 0; c < C; c++) {
            float sum = b3[c];
            for (int v = 0; v < H2; v++) sum += h2_buf[v] * W3[v * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```