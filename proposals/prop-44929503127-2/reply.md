```hypothesis
We propose a "label-embed" family: each of the 16 context bytes is mapped to a learned dense embedding (size E), summed, and fed to a linear classifier. This compresses the 4096 one-hot dimensions to E (e.g., 64), reducing compute and memory while capturing byte similarities. On character prediction, local byte patterns matter; embeddings generalize across similar contexts. If dev accuracy does not exceed label-linear's 0.3375 at comparable or lower cost, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
# Hyperparameter grid: (embedding_size, learning_rate, momentum, l2, epochs_search, epochs_final)
GRID = [
    (32, 0.1, 0.9, 1e-4, 20, 50),
    (64, 0.1, 0.9, 1e-4, 20, 50),
    (128, 0.1, 0.9, 1e-4, 20, 50),
    (32, 0.05, 0.9, 1e-4, 20, 50),
    (64, 0.05, 0.9, 1e-4, 20, 50),
    (128, 0.05, 0.9, 1e-4, 20, 50),
]
BATCH = 64
MAGIC = 0x44424D45  # 'EMBD' in little-endian

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    # logits: (B, C), y: (B,) int
    B = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B  # (B, C)

def evaluate(emb, W, b, X, y):
    # X: (n, d) uint8, y: (n,) int64
    n, d = X.shape
    E = emb.shape[1]
    C = W.shape[1]
    acc = np.zeros((n, E), dtype=np.float32)
    for j in range(d):
        acc += emb[X[:, j]]
    logits = acc @ W + b
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def train_model(emb, W, b, X, y, lr, momentum, l2, epochs, rng):
    n, d = X.shape
    E = emb.shape[1]
    C = W.shape[1]
    # momentum buffers
    v_emb = np.zeros_like(emb)
    v_W = np.zeros_like(W)
    v_b = np.zeros_like(b)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            # forward
            h = np.zeros((len(yb), E), dtype=np.float32)
            for j in range(d):
                h += emb[xb[:, j]]
            logits = h @ W + b
            # gradient
            g = softmax_cross_entropy_grad(logits, yb)  # (B, C)
            # dW, db
            dW = h.T @ g + l2 * W
            db = g.sum(axis=0)
            # dh
            dh = g @ W.T  # (B, E)
            # demb
            demb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(demb, xb[:, j], dh)
            demb += l2 * emb
            # momentum update
            v_emb = momentum * v_emb + lr * demb
            v_W = momentum * v_W + lr * dW
            v_b = momentum * v_b + lr * db
            emb -= v_emb
            W -= v_W
            b -= v_b

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # tuning split: last 20%
    split = int(n * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_emb = best_W = best_b = None

    for E, lr, momentum, l2, epochs_search, epochs_final in GRID:
        # initialize
        emb = (rng.normal(0, 1/math.sqrt(256), (256, E))).astype(np.float32)
        W = (rng.normal(0, 1/math.sqrt(E), (E, C))).astype(np.float32)
        b = np.zeros(C, dtype=np.float32)
        # train on tuning split
        train_model(emb, W, b, X_tr, y_tr, lr, momentum, l2, epochs_search, rng)
        acc = evaluate(emb, W, b, X_tune, y_tune)
        if acc > best_acc:
            best_acc = acc
            best_params = (E, lr, momentum, l2, epochs_final)
            best_emb, best_W, best_b = emb.copy(), W.copy(), b.copy()

    # retrain on full data with best params
    E, lr, momentum, l2, epochs_final = best_params
    emb = (rng.normal(0, 1/math.sqrt(256), (256, E))).astype(np.float32)
    W = (rng.normal(0, 1/math.sqrt(E), (E, C))).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    train_model(emb, W, b, X, y, lr, momentum, l2, epochs_final, rng)

    # write weights
    out = struct.pack("<4I", MAGIC, d, E, C)
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x44424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights params\n"); return 3; }
    long expected = 16 + 4L * (256L * E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W = emb + 256L * E;
    const float *b = W + (long)E * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * E);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, sizeof(float) * E);
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * E;
            for (int u = 0; u < E; u++) h[u] += e[u];
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int u = 0; u < E; u++) {
            float hu = h[u];
            if (hu == 0.0f) continue;
            const float *wu = W + (long)u * C;
            for (int c = 0; c < C; c++) scores[c] += hu * wu[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```