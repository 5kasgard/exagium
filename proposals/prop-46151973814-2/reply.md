```hypothesis
Family: label-emb (factorized embedding + linear). Each of the 16 positions gets a learned 256×D embedding; the 16 vectors are summed and fed to a linear layer (D×65). This captures per-position character similarities with far fewer parameters than one-hot linear (16×256×D vs 16×256×65), so it should generalize better on small data while keeping inference extremely fast (only 16×D adds + D×65 mul-adds). If dev accuracy ≤ label-linear (0.3375) or instructions/item ≥ 2× label-linear, the approach fails.
```

```python
"""label-emb: factorized embedding + linear classifier for fixed-size byte sequences.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Hyper-parameters chosen on last 20% of train.in (tuning split), then retrained on all train."""
import json, math, struct, sys

import numpy as np

# Fixed hyper-parameter grid searched on the tuning split
D_CHOICES = [16, 32, 48]
LR_CHOICES = [0.05, 0.1, 0.2]
L2_CHOICES = [1e-4, 1e-3]
BATCH = 128
EPOCHS_TUNE = 8
EPOCHS_FULL = 20
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    """Return gradient of CE w.r.t logits, shape (N, C)."""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return (probs / len(y)).astype(np.float32)

def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()

def train_eval(Emb, W, b, X, y, lr, l2, epochs, rng, batch=BATCH, verbose=False):
    """In-place Adam on Emb (d,256,D), W (D,C), b (C). Returns final params."""
    d, _, D = Emb.shape
    C = W.shape[1]
    # Adam state
    mEmb = np.zeros_like(Emb); vEmb = np.zeros_like(Emb)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    n = X.shape[0]
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]
            # forward: sum embeddings
            h = np.zeros((B, D), np.float32)
            for j in range(d):
                h += Emb[j, xb[:, j]]       # (B, D)
            logits = h @ W + b              # (B, C)
            # loss gradient
            g = softmax_cross_entropy_grad(logits, yb)   # (B, C)
            # gradients
            gW = h.T @ g + l2 * W           # (D, C)
            gb = g.sum(axis=0)              # (C,)
            gEmb = np.zeros_like(Emb)
            for j in range(d):
                # scatter-add g into gEmb[j, xb[:,j]]
                np.add.at(gEmb[j], xb[:, j], g)
            gEmb += l2 * Emb
            # Adam update
            t += 1
            for param, grad, m, v in [(Emb, gEmb, mEmb, vEmb),
                                      (W, gW, mW, vW),
                                      (b, gb, mb, vb)]:
                m *= beta1; m += (1-beta1) * grad
                v *= beta2; v += (1-beta2) * grad * grad
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return Emb, W, b

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # load all data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # split: last 20% for tuning
    split = int(0.8 * n)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_params = None
    best_hp = None

    for D in D_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                # init
                Emb = (rng.normal(0, 1/math.sqrt(D), size=(d, 256, D)).astype(np.float32))
                W = (rng.normal(0, 1/math.sqrt(D), size=(D, C)).astype(np.float32))
                b = np.zeros(C, np.float32)
                # train on train split
                Emb, W, b = train_eval(Emb, W, b, X_tr, y_tr, lr, l2, EPOCHS_TUNE, rng)
                # evaluate on validation
                h_va = np.zeros((X_va.shape[0], D), np.float32)
                for j in range(d):
                    h_va += Emb[j, X_va[:, j]]
                logits_va = h_va @ W + b
                acc = accuracy(logits_va, y_va)
                if acc > best_acc:
                    best_acc = acc
                    best_hp = (D, lr, l2)
                    best_params = (Emb.copy(), W.copy(), b.copy())

    # retrain on full data with best hyper-parameters
    D, lr, l2 = best_hp
    Emb, W, b = best_params
    rng_full = np.random.default_rng(SEED)  # reset for reproducibility
    Emb, W, b = train_eval(Emb, W, b, X, y, lr, l2, EPOCHS_FULL, rng_full)

    # pack weights: magic, d, D, C; then Emb[d][256][D], W[D][C], b[C] (all float32 LE)
    magic = 0x454D4245  # 'EMBE'
    out = struct.pack("<4i", magic, d, D, C)
    out += Emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte (class index).
 * Weights format: int32 magic(0x454D4245), d, D, C; float32 Emb[d][256][D], W[D][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *Emb = (const float *)(hd + 4);
    const float *W = Emb + (long)d * 256 * D;
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * D);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, D * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = Emb + ((long)j * 256 + xi[j]) * D;
            for (int u = 0; u < D; u++) h[u] += e[u];
        }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int u = 0; u < D; u++) sum += h[u] * W[(long)u * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```