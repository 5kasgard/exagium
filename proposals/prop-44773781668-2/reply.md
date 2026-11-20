```hypothesis
Propose "label-emb": per-position byte embeddings (dimension D=4) followed by a linear layer.
This factorizes the 16×256×65 weight matrix of label-linear into 16×256×D + 16×D×65 parameters,
capturing cross-position interactions while reducing model size and inference cost.
If dev accuracy does not exceed label-linear's 0.3375, the family does not justify its complexity.
```

```python
"""label-emb: per-position embeddings + linear layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Tunes embedding dimension and learning rate on last 20% of train.in, then trains on all data.
Exports: magic 0x454D424C, d, D, C; float32 emb[d][256][D], W[16*D][C], b[C]."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameter grid for tuning split
SEED = 1
EMB_DIMS = [2, 4, 8]
LRS = [0.01, 0.003]
L2 = 1e-4
BATCH = 64
MAX_EPOCHS_TUNE = 5
MAX_EPOCHS_FULL = min(100, math.ceil(600000 / 24000))  # ~25 epochs on full data

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(d, D, C, rng):
    # embeddings: list of d arrays (256, D)
    emb = [rng.normal(0, math.sqrt(2.0 / (256 + D)), (256, D)).astype(np.float32) for _ in range(d)]
    W = rng.normal(0, math.sqrt(2.0 / (d * D + C)), (d * D, C)).astype(np.float32)
    b = np.zeros(C, np.float32)
    return emb, W, b

def forward(emb, W, b, X):
    # X: (N, d) uint8
    N, d = X.shape
    D = emb[0].shape[1]
    flat = np.empty((N, d * D), np.float32)
    for j in range(d):
        flat[:, j*D:(j+1)*D] = emb[j][X[:, j]]
    logits = flat @ W + b
    return logits, flat

def loss_grad(logits, y):
    # logits: (N, C), y: (N,)
    N = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(N), y] -= 1.0
    return probs / N  # dL/dlogits

def train_epoch(emb, W, b, X, y, lr, l2, batch, rng, adam_state):
    N = X.shape[0]
    order = rng.permutation(N)
    t = adam_state['t']
    for s in range(0, N, batch):
        idx = order[s:s+batch]
        xb, yb = X[idx], y[idx]
        logits, flat = forward(emb, W, b, xb)
        g = loss_grad(logits, yb)  # (B, C)
        # gradients for W and b
        gW = flat.T @ g + l2 * W
        gb = g.sum(axis=0)
        # gradients for embeddings
        d, D = len(emb), emb[0].shape[1]
        g_emb = [np.zeros_like(e) for e in emb]
        for j in range(d):
            gj = g @ W[j*D:(j+1)*D].T  # (B, D)
            np.add.at(g_emb[j], xb[:, j], gj)
            g_emb[j] += l2 * emb[j]
        # Adam update
        t += 1
        params = [W, b] + g_emb  # order for state
        grads = [gW, gb] + g_emb
        for i, (p, gr) in enumerate(zip(params, grads)):
            m, v = adam_state[i]
            m *= 0.9; m += 0.1 * gr
            v *= 0.999; v += 0.001 * gr * gr
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        adam_state['t'] = t
    return emb, W, b

def evaluate(emb, W, b, X, y):
    logits, _ = forward(emb, W, b, X)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_emb, best_W, best_b = None, None, None
    best_D, best_lr = 4, 0.01

    # Grid search on tuning split
    for D in EMB_DIMS:
        for lr in LRS:
            emb, W, b = init_params(d, D, C, rng)
            adam_state = {'t': 0}
            for _ in range(1, MAX_EPOCHS_TUNE + 1):
                adam_state = [{'m': np.zeros_like(p), 'v': np.zeros_like(p)} for p in [W, b] + emb]
                adam_state.append({'t': 0})  # hack to keep t separate
                # Reinitialize state properly
                adam_state = {'t': 0}
                for p in [W, b] + emb:
                    adam_state[id(p)] = (np.zeros_like(p), np.zeros_like(p))
                emb, W, b = train_epoch(emb, W, b, X_tr, y_tr, lr, L2, BATCH, rng, adam_state)
            acc = evaluate(emb, W, b, X_val, y_val)
            if acc > best_acc:
                best_acc = acc
                best_D, best_lr = D, lr
                best_emb, best_W, best_b = emb, W, b

    # Retrain on full data with best hyperparameters
    rng = np.random.default_rng(SEED)
    emb, W, b = init_params(d, best_D, C, rng)
    adam_state = {'t': 0}
    for p in [W, b] + emb:
        adam_state[id(p)] = (np.zeros_like(p), np.zeros_like(p))
    for _ in range(MAX_EPOCHS_FULL):
        emb, W, b = train_epoch(emb, W, b, X, y, best_lr, L2, BATCH, rng, adam_state)

    # Write weights
    magic = 0x454D424C  # 'EMBL'
    out = struct.pack("<4i", magic, d, best_D, C)
    for j in range(d):
        out += best_emb[j].astype("<f4").tobytes()
    out += best_W.astype("<f4").tobytes()
    out += best_b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x454D424C, int32 d, D, C; float32 emb[d][256][D], W[d*D][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * D);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Gather embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            memcpy(flat + j * D, e, sizeof(float) * D);
        }
        // Compute logits = flat @ W + b
        for (int c = 0; c < C; c++) logits[c] = b[c];
        for (int c = 0; c < C; c++) {
            const float *wc = W + c;  // column-major? W is row-major (d*D rows, C cols)
            float sum = 0.0f;
            for (int k = 0; k < d * D; k++) sum += flat[k] * wc[k * C];
            logits[c] += sum;
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```