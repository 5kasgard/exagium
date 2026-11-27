```hypothesis
Family: label-embsum (per-pixel embedding sum + linear). Each of the 64 pixels (values 0..16) gets a learned E-dim embedding; we sum them and apply a linear layer to 10 classes. This captures per-pixel non-linearities without cross-pixel interactions in the first layer, so it should beat linear (96.1%) while staying near its cost (E=8 gives ~592 MAs vs linear's 640). If dev accuracy ≤ 96.1% or Ir/item ≥ 6000, the idea fails.
```
```python
import json, math, struct, sys

import numpy as np

# embedding dimension candidates (tuned on last 20% of train.in)
E_CANDIDATES = [4, 8, 16]
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_eval(X_tr, y_tr, X_va, y_va, E, rng):
    n, d = X_tr.shape
    C = 10
    # embeddings: d x 17 x E
    emb = (rng.uniform(-1, 1, (d, 17, E)) * math.sqrt(6.0 / 17)).astype(np.float32)
    W = (rng.uniform(-1, 1, (E, C)) * math.sqrt(6.0 / E)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(X):
        # X: (n, d) uint8
        n = X.shape[0]
        h = np.zeros((n, E), np.float32)
        for j in range(d):
            h += emb[j, X[:, j]]
        return h @ W + b

    def loss_grad(X, y):
        n = X.shape[0]
        h = np.zeros((n, E), np.float32)
        for j in range(d):
            h += emb[j, X[:, j]]
        logits = h @ W + b
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(n), y] -= 1.0
        probs /= n
        # grads
        gW = h.T @ probs + L2 * W
        gb = probs.sum(0)
        gemb = np.zeros_like(emb)
        gh = probs @ W.T
        for j in range(d):
            for v in range(17):
                mask = (X[:, j] == v)
                if mask.any():
                    gemb[j, v] = gh[mask].sum(0) + L2 * emb[j, v]
        return [gemb, gW, gb], (probs.argmax(1) == y).mean()

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            grads, _ = loss_grad(X_tr[idx], y_tr[idx])
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    # validation accuracy
    _, acc = loss_grad(X_va, y_va)
    return params, acc


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    assert d == 64 and C == 10

    n = X.shape[0]
    tune_size = n // 5
    X_tr, y_tr = X[:-tune_size], y[:-tune_size]
    X_va, y_va = X[-tune_size:], y[-tune_size:]

    rng = np.random.default_rng(SEED)
    best_E, best_acc, best_params = None, -1, None
    for E in E_CANDIDATES:
        params, acc = train_eval(X_tr, y_tr, X_va, y_va, E, rng)
        if acc > best_acc:
            best_acc, best_E, best_params = acc, E, params

    # retrain on all data with best_E
    rng = np.random.default_rng(SEED)
    best_params, _ = train_eval(X, y, X[:1], y[:1], best_E, rng)  # dummy val
    emb, W, b = best_params

    # write weights: magic, d, E, C, then emb[d][17][E], W[E][C], b[C]
    magic = 0x454D4253  # "EMBS"
    out = struct.pack("<4i", magic, d, best_E, C)
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
    if (nw < 16 || hd[0] != 0x454D4253) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 64 || C != 10 || E < 1 || E > 16) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 17 * E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 17 * E;
    const float *b = W + (long)E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * E);
    float *sc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int e = 0; e < E; e++) h[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            if (v > 16) v = 16;
            const float *e = emb + ((long)j * 17 + v) * E;
            for (int e = 0; e < E; e++) h[e] += e[e];
        }
        for (int c = 0; c < C; c++) sc[c] = b[c];
        for (int e = 0; e < E; e++) {
            float he = h[e];
            const float *we = W + (long)e * C;
            for (int c = 0; c < C; c++) sc[c] += he * we[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```