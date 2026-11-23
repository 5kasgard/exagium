```hypothesis
Family: label-embed (linear model with learned per-feature-value embeddings).
Why: Adult has 14 byte-wide features (6 ordinal quantized to 255 bins, 8 categorical). 
Label-linear uses one-hot for categorical (high-dim sparse) and z-score for ordinal. 
A small dense embedding (E=8) per feature value captures similarities between bins/categories 
and learns interactions via the final linear layer, boosting accuracy. 
Inference cost stays low: 14*E adds + E*C mul-adds per item (~112 adds + 16 mul-adds for E=8, C=2).
Failure: dev accuracy ≤ 0.8524 (label-linear) or Ir/item > 1000.
```

```python
"""label-embed: linear model on summed per-feature-value embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all train.
Exported weights: magic 0x44424D45 ('EMBD' LE), d, E, C; then float32 emb[d][256][E], W[E][C], b[C]."""
import json, math, struct, sys

import numpy as np

# hyperparameter grid searched on tuning split
EMB_SIZES = [4, 8, 16]
LRS = [1e-3, 3e-3, 1e-2]
L2 = 1e-4
BATCH = 64
EPOCHS = 50
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_eval(emb_size, lr, X_tr, y_tr, X_va, y_va, C, d, rng):
    # init embeddings [d, 256, E] and linear [E, C], b[C]
    scale = math.sqrt(6.0 / (256 * d))
    emb = (rng.uniform(-1, 1, (d, 256, emb_size)) * scale).astype(np.float32)
    W = (rng.uniform(-1, 1, (emb_size, C)) * math.sqrt(6.0 / emb_size)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n_tr = X_tr.shape[0]
    for _ in range(EPOCHS):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            # forward
            sums = np.zeros((len(yb), emb_size), np.float32)
            for j in range(d):
                sums += emb[j, xb[:, j]]
            logits = sums @ W + b
            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            probs /= len(yb)
            # gradients
            grad_W = sums.T @ probs + L2 * W
            grad_b = probs.sum(axis=0)
            grad_sums = probs @ W.T
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_sums)
            grad_emb += L2 * emb
            grads = [grad_emb, grad_W, grad_b]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    # validation accuracy
    sums = np.zeros((X_va.shape[0], emb_size), np.float32)
    for j in range(d):
        sums += emb[j, X_va[:, j]]
    logits = sums @ W + b
    preds = logits.argmax(axis=1)
    return (preds == y_va).mean(), (emb, W, b)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]
    # split: last 20% for tuning
    split = int(0.8 * n)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]
    rng = np.random.default_rng(SEED)
    best_acc, best_params, best_cfg = -1, None, None
    for E in EMB_SIZES:
        for lr in LRS:
            rng_split = np.random.default_rng(SEED)  # same init each config
            acc, params = train_eval(E, lr, X_tr, y_tr, X_va, y_va, C, d, rng_split)
            if acc > best_acc:
                best_acc, best_params, best_cfg = acc, params, (E, lr)
    # retrain on full data with best config
    E, lr = best_cfg
    rng_full = np.random.default_rng(SEED)
    _, final_params = train_eval(E, lr, X, y, X[:1], y[:1], C, d, rng_full)  # dummy val
    emb, W, b = final_params
    # write weights
    magic = 0x44424D45  # 'EMBD' little-endian
    out = struct.pack("<4i", magic, d, E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index.
 * Weights format: magic 0x44424D45, int32 d, E, C; float32 emb[d][256][E], W[E][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x44424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)E * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum = malloc(sizeof(float) * E);
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int e = 0; e < E; e++) sum[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) sum[e] += eptr[e];
        }
        for (int c = 0; c < C; c++) {
            float val = b[c];
            for (int e = 0; e < E; e++) val += sum[e] * W[(long)e * C + c];
            logits[c] = val;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```