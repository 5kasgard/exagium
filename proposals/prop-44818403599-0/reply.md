```hypothesis
Factorization Machines (FM) capture pairwise feature interactions with low-rank embeddings, ideal for tabular data where interactions (e.g., education × occupation) matter. Treating all 14 columns as 256-category features (ordinal bins already quantized, categorical indices) avoids manual encoding. A per-class FM with embedding size k=4 adds ~35k parameters vs linear's ~7k, but inference remains ~500 FLOPs/item (91 dot products of size 4). We tune k∈{2,4,8} on the last 20% of train.in; if dev accuracy ≤ label-linear (0.8524) or cost exceeds 2× label-linear, the family is not worthwhile.
```

```python
import json, math, struct, sys

import numpy as np

# hyper-parameters (tuned on last 20% of train.in)
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
K_CANDIDATES = [2, 4, 8]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def fm_forward(X, w0, W, V, k):
    """X: (n, d) uint8 indices. Returns logits (n, C)."""
    n, d = X.shape
    C = w0.shape[0]
    logits = np.broadcast_to(w0.astype(np.float32), (n, C))
    # linear part
    for c in range(C):
        for j in range(d):
            logits[:, c] += W[c, j, X[:, j]]
    # pairwise interactions
    for c in range(C):
        for i in range(d):
            vi = V[c, i, X[:, i], :]          # (n, k)
            for j in range(i + 1, d):
                vj = V[c, j, X[:, j], :]      # (n, k)
                logits[:, c] += np.sum(vi * vj, axis=1)
    return logits

def fm_backward(X, y, logits, w0, W, V, k):
    """Compute gradients for one minibatch. Returns (g_w0, g_W, g_V)."""
    n, d = X.shape
    C = w0.shape[0]
    # softmax gradient
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n
    g_w0 = probs.sum(axis=0).astype(np.float32)
    g_W = np.zeros_like(W)
    g_V = np.zeros_like(V)
    for c in range(C):
        pc = probs[:, c]                      # (n,)
        # linear gradients
        for j in range(d):
            idx = X[:, j]
            np.add.at(g_W[c, j], idx, pc)
        # interaction gradients
        for i in range(d):
            vi = V[c, i, X[:, i], :]          # (n, k)
            for j in range(i + 1, d):
                vj = V[c, j, X[:, j], :]      # (n, k)
                # d/dvi = pc * vj, d/dvj = pc * vi
                gvi = pc[:, None] * vj
                gvj = pc[:, None] * vi
                np.add.at(g_V[c, i], X[:, i], gvi)
                np.add.at(g_V[c, j], X[:, j], gvj)
    return g_w0, g_W, g_V

def train_fm(X, y, C, k, rng, n_epochs):
    d = X.shape[1]
    # init
    scale = math.sqrt(6.0 / (256 * k))
    w0 = np.zeros(C, np.float32)
    W = (rng.uniform(-1, 1, (C, d, 256)) * math.sqrt(6.0 / 256)).astype(np.float32)
    V = (rng.uniform(-1, 1, (C, d, 256, k)) * scale).astype(np.float32)
    params = [w0, W, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi, yi = X[idx], y[idx]
            logits = fm_forward(xi, w0, W, V, k)
            g_w0, g_W, g_V = fm_backward(xi, yi, logits, w0, W, V, k)
            grads = [g_w0 + L2 * w0, g_W + L2 * W, g_V + L2 * V]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return w0, W, V

def eval_accuracy(X, y, w0, W, V, k):
    logits = fm_forward(X, w0, W, V, k)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X_all.shape[0]
    # tuning split: last 20% in file order
    split = int(0.8 * n)
    X_train, y_train = X_all[:split], y_all[:split]
    X_tune, y_tune = X_all[split:], y_all[split:]
    rng = np.random.default_rng(SEED)
    best_k, best_acc = K_CANDIDATES[0], -1.0
    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / split))
    for k in K_CANDIDATES:
        w0, W, V = train_fm(X_train, y_train, C, k, rng, n_epochs)
        acc = eval_accuracy(X_tune, y_tune, w0, W, V, k)
        if acc > best_acc:
            best_acc, best_k = acc, k
    # final training on all data
    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    w0, W, V = train_fm(X_all, y_all, C, best_k, rng, n_epochs)
    # save weights: magic, d, k, C, then w0[C], W[C,d,256], V[C,d,256,k]
    magic = 0x464D3231  # 'FM21'
    out = struct.pack("<4i", magic, d, best_k, C)
    out += w0.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D3231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 14) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L
        + 4L * C
        + 4L * C * d * 256
        + 4L * C * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *w0 = (const float *)(hd + 4);
    const float *W = w0 + C;
    const float *V = W + (long)C * d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if ((int)rec_len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        for (int c = 0; c < C; c++) scores[c] = w0[c];
        // linear part
        for (int c = 0; c < C; c++) {
            const float *Wc = W + (long)c * d * 256;
            for (int j = 0; j < d; j++) scores[c] += Wc[j * 256 + xi[j]];
        }
        // pairwise interactions
        for (int c = 0; c < C; c++) {
            const float *Vc = V + (long)c * d * 256 * k;
            for (int ii = 0; ii < d; ii++) {
                const float *vi = Vc + (long)ii * 256 * k + xi[ii] * k;
                for (int jj = ii + 1; jj < d; jj++) {
                    const float *vj = Vc + (long)jj * 256 * k + xi[jj] * k;
                    float sum = 0.0f;
                    for (int f = 0; f < k; f++) sum += vi[f] * vj[f];
                    scores[c] += sum;
                }
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```