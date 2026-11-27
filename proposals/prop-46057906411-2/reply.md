```hypothesis
Family: embedding-mlp (positional byte embeddings + one ReLU layer).  
Why: The linear model uses one-hot per position (16×65 dims) and cannot capture interactions between positions. Learned embeddings (16×256×E) compress the input, allow parameter sharing across positions, and a hidden ReLU layer models non-linear interactions. This should beat label-linear (0.3375) with modest CPU cost.  
Failure sign: If dev accuracy ≤ 0.3375 or inference cost exceeds label-linear by >2×, the family does not improve the trade-off.
```

```python
import json, math, struct, sys
import numpy as np

# Fixed seed for reproducibility
SEED = 12345
# Hyperparameter grid (tuned on last 20% of train.in)
GRID = [
    (8, 32, 3e-3, 1e-4),
    (8, 64, 3e-3, 1e-4),
    (16, 32, 1e-3, 1e-4),
    (16, 64, 1e-3, 1e-4),
]
EPOCHS_TUNE = 30
EPOCHS_FULL = 80
BATCH = 64
SAMPLES = 600000

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def init_params(E, H, d, V, C, rng):
    # embeddings: d x V x E
    emb = rng.normal(0, 0.01, (d, V, E)).astype(np.float32)
    # hidden: (d*E) x H
    W1 = rng.normal(0, math.sqrt(2.0/(d*E)), (d*E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # output: H x C
    W2 = rng.normal(0, math.sqrt(2.0/H), (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return [emb, W1, b1, W2, b2]

def forward(params, X_batch, E, H, d, V):
    emb, W1, b1, W2, b2 = params
    bs = X_batch.shape[0]
    # Embedding lookup: (bs, d, E)
    embedded = np.zeros((bs, d, E), dtype=np.float32)
    for j in range(d):
        embedded[:, j, :] = emb[j, X_batch[:, j], :]
    flat = embedded.reshape(bs, d*E)
    hidden = flat @ W1 + b1
    np.maximum(hidden, 0, out=hidden)  # ReLU in-place
    logits = hidden @ W2 + b2
    return logits, hidden, flat

def train(params, X, y, E, H, d, V, C, lr, l2, rng, epochs):
    n = X.shape[0]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            logits, hidden, flat = forward(params, xb, E, H, d, V)
            # Softmax cross-entropy gradient
            max_logits = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(len(yb)), yb] -= 1.0
            grad_logits /= len(yb)
            # Backprop
            grad_W2 = hidden.T @ grad_logits + l2 * params[3]
            grad_b2 = np.sum(grad_logits, axis=0)
            grad_hidden = grad_logits @ params[3].T
            grad_hidden[hidden <= 0] = 0
            grad_W1 = flat.T @ grad_hidden + l2 * params[1]
            grad_b1 = np.sum(grad_hidden, axis=0)
            grad_flat = grad_hidden @ params[1].T
            grad_flat_rs = grad_flat.reshape(len(yb), d, E)
            grad_emb = np.zeros_like(params[0])
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_flat_rs[:, j, :])
            grad_emb += l2 * params[0]
            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return params

def evaluate(params, X, y, E, H, d, V, C):
    logits, _, _ = forward(params, X, E, H, d, V)
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y)

def write_weights(path, params, E, H, d, V, C):
    emb, W1, b1, W2, b2 = params
    magic = 0x4D4C5045  # "EPLM"
    with open(path, "wb") as f:
        f.write(struct.pack("<5i", magic, d, V, E, H, C))
        f.write(emb.astype("<f4").tobytes())
        f.write(W1.astype("<f4").tobytes())
        f.write(b1.astype("<f4").tobytes())
        f.write(W2.astype("<f4").tobytes())
        f.write(b2.astype("<f4").tobytes())

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    V = 256
    C = card["classes"]
    X_rec = records(sys.argv[2])
    y_rec = records(sys.argv[3])
    n = len(X_rec)
    X = np.frombuffer(b"".join(X_rec), dtype=np.uint8).reshape(n, d)
    y = np.array([r[0] for r in y_rec], dtype=np.int64)
    # Split: last 20% for tuning
    split = int(n * 0.8)
    X_tr, X_tune = X[:split], X[split:]
    y_tr, y_tune = y[:split], y[split:]
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    best_params = None
    for E, H, lr, l2 in GRID:
        params = init_params(E, H, d, V, C, rng)
        params = train(params, X_tr, y_tr, E, H, d, V, C, lr, l2, rng, EPOCHS_TUNE)
        acc = evaluate(params, X_tune, y_tune, E, H, d, V, C)
        if acc > best_acc:
            best_acc = acc
            best_cfg = (E, H, lr, l2)
            best_params = [p.copy() for p in params]
    # Retrain best on full data
    E, H, lr, l2 = best_cfg
    params = init_params(E, H, d, V, C, rng)
    params = train(params, X, y, E, H, d, V, C, lr, l2, rng, EPOCHS_FULL)
    write_weights(sys.argv[4], params, E, H, d, V, C)

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
    uint32_t k = buf[*pos] | (buf[*pos+1]<<8) | (buf[*pos+2]<<16) | ((uint32_t)buf[*pos+3]<<24);
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, (k>>8)&255, (k>>16)&255, (k>>24)&255};
    fwrite(h, 1, 4, f);
    fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *w = slurp(argv[1], &nw);
    if (nw < 24) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)w;
    if (hd[0] != 0x4D4C5045) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], V = hd[2], E = hd[3], H = hd[4], C = hd[5];
    if (d != 16 || V != 256 || C != 65) { fprintf(stderr, "unexpected dimensions\n"); return 3; }
    long expected = 24
        + (long)d * V * E * 4
        + (long)d * E * H * 4
        + (long)H * 4
        + (long)H * C * 4
        + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    float *emb = (float *)(w + 24);
    float *W1 = emb + (long)d * V * E;
    float *b1 = W1 + (long)d * E * H;
    float *W2 = b1 + H;
    float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);
    float *flat = malloc(sizeof(float) * d * E);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            int idx = xi[j];
            const float *e = emb + ((long)j * V + idx) * E;
            memcpy(flat + j * E, e, E * sizeof(float));
        }
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int f = 0; f < d * E; f++)
                sum += flat[f] * W1[(long)f * H + h];
            hidden[h] = sum > 0 ? sum : 0;
        }
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++)
                sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++)
            if (logits[c] > logits[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```