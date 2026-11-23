```hypothesis
Embedding-linear family: learn a small embedding per byte position (16 positions, 256 values each) followed by a linear classifier. This reduces parameters vs. one-hot linear (label-linear) and should generalize better on character prediction, beating 0.3375 accuracy while keeping inference fast (small matrix multiply). Failure: accuracy ≤ label-linear or inference cost not lower.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameter search space (tuned on last 20% of train.in)
EMBED_DIMS = [4, 8, 16, 32]
LRS = [1e-3, 3e-3]
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def split_data(X, y, tune_frac=0.2):
    n = X.shape[0]
    tune_size = int(n * tune_frac)
    # last 20% in file order
    X_train, X_tune = X[:-tune_size], X[-tune_size:]
    y_train, y_tune = y[:-tune_size], y[-tune_size:]
    return X_train, y_train, X_tune, y_tune

def init_params(d, D, C, rng):
    # embeddings: list of d arrays (256, D)
    embeddings = [(rng.uniform(-1, 1, (256, D)) * math.sqrt(6.0 / 256)).astype(np.float32) for _ in range(d)]
    W_out = (rng.uniform(-1, 1, (d * D, C)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
    b_out = np.zeros(C, np.float32)
    return embeddings, W_out, b_out

def forward(embeddings, W_out, b_out, X, d, D):
    # X: (batch, d) uint8
    batch = X.shape[0]
    emb = np.zeros((batch, d * D), np.float32)
    for j in range(d):
        emb[:, j*D:(j+1)*D] = embeddings[j][X[:, j]]
    logits = emb @ W_out + b_out
    return logits, emb

def loss_grad(logits, y, batch):
    # logits: (batch, C), y: (batch,)
    P = np.exp(logits - logits.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    P[np.arange(batch), y] -= 1.0
    return (P / batch).astype(np.float32)  # dL/dlogits

def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()

def train_loop(embeddings, W_out, b_out, X, y, d, D, C, lr, l2, batch, samples, max_epochs, rng):
    n = X.shape[0]
    epochs = min(max_epochs, math.ceil(samples / n))
    params = embeddings + [W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            xb, yb = X[idx], y[idx]
            logits, emb = forward(embeddings, W_out, b_out, xb, d, D)
            g_logits = loss_grad(logits, yb, len(yb))
            # gradients
            g_W_out = emb.T @ g_logits + l2 * W_out
            g_b_out = g_logits.sum(0)
            g_emb = g_logits @ W_out.T  # (batch, d*D)
            g_embeddings = []
            for j in range(d):
                gj = g_emb[:, j*D:(j+1)*D]  # (batch, D)
                # scatter-add into (256, D)
                gE = np.zeros((256, D), np.float32)
                np.add.at(gE, xb[:, j], gj)
                g_embeddings.append(gE + l2 * embeddings[j])
            grads = g_embeddings + [g_W_out, g_b_out]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return embeddings, W_out, b_out

def evaluate(embeddings, W_out, b_out, X, y, d, D):
    logits, _ = forward(embeddings, W_out, b_out, X, d, D)
    return accuracy(logits, y)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    rng = np.random.default_rng(SEED)

    X_train, y_train, X_tune, y_tune = split_data(X, y)

    best_acc = -1
    best_cfg = None
    best_params = None

    for D in EMBED_DIMS:
        for lr in LRS:
            embeddings, W_out, b_out = init_params(d, D, C, rng)
            embeddings, W_out, b_out = train_loop(embeddings, W_out, b_out, X_train, y_train, d, D, C,
                                                 lr, L2, BATCH, SAMPLES, MAX_EPOCHS, rng)
            acc = evaluate(embeddings, W_out, b_out, X_tune, y_tune, d, D)
            if acc > best_acc:
                best_acc = acc
                best_cfg = (D, lr)
                best_params = (embeddings.copy(), W_out.copy(), b_out.copy())

    # Retrain on full data with best config
    D, lr = best_cfg
    embeddings, W_out, b_out = init_params(d, D, C, rng)
    embeddings, W_out, b_out = train_loop(embeddings, W_out, b_out, X, y, d, D, C,
                                         lr, L2, BATCH, SAMPLES, MAX_EPOCHS, rng)

    # Write weights.bin
    # Header: magic, d, D, C
    magic = 0x454D4244  # 'DBME'
    hdr = struct.pack("<4i", magic, d, D, C)
    # Embeddings: d * 256 * D floats
    emb_bytes = b"".join(E.astype("<f4").tobytes() for E in embeddings)
    # Output weights and bias
    out_bytes = W_out.astype("<f4").tobytes() + b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(hdr + emb_bytes + out_bytes)

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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D <= 0 || D > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W_out = emb + (long)d * 256 * D;
    const float *b_out = W_out + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    float *emb_concat = malloc(sizeof(float) * d * D);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build concatenated embedding vector
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            memcpy(emb_concat + j * D, e, D * sizeof(float));
        }

        // logits = emb_concat @ W_out + b_out
        for (int c = 0; c < C; c++) logits[c] = b_out[c];
        for (int k = 0; k < d * D; k++) {
            float ek = emb_concat[k];
            const float *w = W_out + (long)k * C;
            for (int c = 0; c < C; c++) logits[c] += ek * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```