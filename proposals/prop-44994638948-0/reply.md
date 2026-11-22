```hypothesis
Family: label-emb (factorized linear model with per-feature embeddings).
Why: Adult has 14 features (6 ordinal, 8 categorical) each quantized to 256 bins. A full one-hot linear model (label-linear) uses ~10k parameters; a low-rank embedding (E=8) uses ~23k but shares strength across categories and captures pairwise interactions implicitly. On this scale it should match or beat label-linear accuracy (0.8524) while keeping inference near 1k Ir/item.
Failure: dev accuracy ≤ 0.8524 or instructions/item > 2000.
```

```python
"""label-emb: factorized linear model with per-feature embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Weights format (little-endian):
  int32 magic=0x454D424C, d, E, C
  float32 emb[d][256][E]
  float32 W[d*E][C]
  float32 b[C]
"""
import json, math, struct, sys

import numpy as np

# tuning candidates for embedding size E
E_CANDIDATES = (4, 8, 16, 32)
SEED = 1
LR, L2, BATCH, MAX_EPOCHS = 3e-3, 1e-4, 64, 100
TUNE_EPOCHS = 20          # epochs for each candidate on train-tune split
FINAL_EPOCHS = 100        # epochs on full train after best E chosen

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def build_dataset(card, in_path, key_path):
    X = np.frombuffer(b"".join(records(in_path)), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(key_path)], dtype=np.int64)
    return X, y

def softmax_cross_entropy_grad(logits, y):
    # logits: (n, C), y: (n,)
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)   # (n, C)

def train_one(X, y, d, C, E, epochs, rng):
    # parameters
    emb = rng.uniform(-0.1, 0.1, (d, 256, E)).astype(np.float32)
    W = rng.uniform(-0.1, 0.1, (d*E, C)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [emb, W, b]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]          # (b, d)
            yb = y[idx]          # (b,)
            # forward: accumulate embeddings
            # we can vectorize: (b, d, E) -> (b, d*E)
            batch_emb = emb[np.arange(d)[None, :, None], xb[:, :, None], np.arange(E)[None, None, :]]  # (b, d, E)
            h = batch_emb.reshape(len(yb), d*E)                         # (b, d*E)
            logits = h @ W + b                                          # (b, C)
            # gradient
            g = softmax_cross_entropy_grad(logits, yb)                  # (b, C)
            # grad_W = h.T @ g + L2 * W
            grad_W = h.T @ g + L2 * W
            grad_b = g.sum(axis=0)
            # grad_emb: for each feature j, each sample i, each e: g[i] @ W[j*E+e, :]^T
            # W_grad_emb shape (d, E, C) -> we need (d, 256, E)
            # g @ W.T -> (b, d*E) then reshape to (b, d, E)
            grad_h = g @ W.T                                              # (b, d*E)
            grad_h = grad_h.reshape(len(yb), d, E)                       # (b, d, E)
            grad_emb = np.zeros_like(emb)
            # scatter-add: for each feature j, index xb[:,j] in 0..255
            np.add.at(grad_emb, (np.arange(d)[None, :], xb[:, :, None], np.arange(E)[None, None, :]), grad_h)
            grad_emb += L2 * emb
            grads = [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9**t)
                v_hat = v[i] / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return emb, W, b

def eval_acc(X, y, emb, W, b):
    d, E = emb.shape[0], emb.shape[2]
    batch_emb = emb[np.arange(d)[None, :, None], X[:, :, None], np.arange(E)[None, None, :]]
    h = batch_emb.reshape(len(y), d*E)
    logits = h @ W + b
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X_all, y_all = build_dataset(card, sys.argv[2], sys.argv[3])
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(y_all)
    split = int(0.8 * n)
    X_tr, y_tr = X_all[:split], y_all[:split]
    X_tune, y_tune = X_all[split:], y_all[split:]
    rng = np.random.default_rng(SEED)
    best_E, best_acc = None, -1.0
    for E in E_CANDIDATES:
        emb, W, b = train_one(X_tr, y_tr, d, C, E, TUNE_EPOCHS, rng)
        acc = eval_acc(X_tune, y_tune, emb, W, b)
        if acc > best_acc:
            best_acc, best_E = acc, E
    # final training on all data with best_E
    emb, W, b = train_one(X_all, y_all, d, C, best_E, FINAL_EPOCHS, rng)
    # write weights
    magic = 0x454D424C  # 'LBME' little-endian
    out = struct.pack("<4i", magic, d, best_E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class index.
 * Weights format (little-endian):
 *   int32 magic=0x454D424C, d, E, C
 *   float32 emb[d][256][E]
 *   float32 W[d*E][C]
 *   float32 b[C]
 */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d < 1 || d > 256 || E < 1 || E > 256 || C < 1 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) acc[c] = b[c];
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int ee = 0; ee < E; ee++) {
                float ev = e[ee];
                const float *w = W + ((long)j * E + ee) * C;
                for (int c = 0; c < C; c++) acc[c] += ev * w[c];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```