```hypothesis
Family: label-embed (position-specific byte embeddings with sum pooling + linear classifier).
Why: Character prediction benefits from modeling byte similarities (vowels, punctuation clusters) which one-hot linear regression cannot capture. Sum-pooling position-specific embeddings (D tuned on validation) yields a strict generalization of label-linear with structured parameters (133K at D=32 vs 266K) and similar inference cost (~2.5K ops vs 1K ops).
Failure signal: Dev accuracy ≤ 33.75% (label-linear) or inference cost > 2× label-linear.
```
```python
"""label-embed: Position-specific byte embeddings with sum pooling + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Tunes embedding dimension D on last 20% of train (validation), then trains on all data.
Exports: int32 magic 0x45424D45, d, D, C; float32 E[d][256][D], W[D][C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
D_CANDIDATES = [16, 32, 48, 64]
TUNE_EPOCHS = 30

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_eval(X, y, val_X, val_y, d, C, D, rng, epochs):
    E = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [E, W, b]
    state = [(np.zeros_like(p, np.float64), np.zeros_like(p, np.float64)) for p in params]
    t = 0
    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xi, yy = X[idx], y[idx]
            bs = len(yy)
            x_emb = np.zeros((bs, D), np.float32)
            for j in range(d):
                x_emb += E[j, xi[:, j]]
            logits = x_emb @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yy] -= 1.0
            dlogits = (probs / bs).astype(np.float32)
            dW = x_emb.T @ dlogits + L2 * W
            db = dlogits.sum(axis=0)
            dE = np.zeros_like(E)
            d_x_emb = dlogits @ W.T
            for j in range(d):
                np.add.at(dE[j], xi[:, j], d_x_emb)
            dE += L2 * E
            grads = [dE, dW, db]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    val_emb = np.zeros((val_X.shape[0], D), np.float32)
    for j in range(d):
        val_emb += E[j, val_X[:, j]]
    val_logits = val_emb @ W + b
    val_pred = val_logits.argmax(axis=1)
    val_acc = (val_pred == val_y).mean()
    return val_acc, params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    n_val = n // 5
    idx = np.arange(n)
    train_idx, val_idx = idx[:-n_val], idx[-n_val:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    rng = np.random.default_rng(SEED)
    best_D, best_acc, best_params = None, -1, None
    for D in D_CANDIDATES:
        acc, params = train_eval(X_train, y_train, X_val, y_val, d, C, D, rng, TUNE_EPOCHS)
        if acc > best_acc:
            best_acc, best_D, best_params = acc, D, params
    rng = np.random.default_rng(SEED)
    _, best_params = train_eval(X, y, X_val, y_val, d, C, best_D, rng, min(MAX_EPOCHS, math.ceil(SAMPLES / n)))
    E, W, b = best_params
    out = struct.pack("<4i", 0x45424D45, d, best_D, C)
    out += E.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embed: Position-specific byte embeddings with sum pooling + linear classifier.
 * ./infer weights.bin in.rec N
 * Weights: int32 magic 0x45424D45, d, D, C; float32 E[d][256][D], W[D][C], b[C].
 * Input: d-byte records. Output: 1-byte class index per record. */
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
    if (nw < 16 || hd[0] != 0x45424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16 || D < 1 || D > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)d * 256 * D;
    const float *b = W + (long)D * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int dd = 0; dd < D; dd++) emb[dd] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = E + ((long)j * 256 + xi[j]) * D;
            for (int dd = 0; dd < D; dd++) emb[dd] += e[dd];
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int dd = 0; dd < D; dd++) {
            float ev = emb[dd];
            const float *w = W + (long)dd * C;
            for (int c = 0; c < C; c++) scores[c] += ev * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```