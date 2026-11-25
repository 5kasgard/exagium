```hypothesis
Family: label-embsum (per-position byte embeddings summed then linear). The task is a character-level language model with 16-byte context; summing learned embeddings captures positional n-gram statistics with far fewer parameters than one-hot linear (16*256*D + D*65 vs 16*256*65). With D=16–32 this should match or beat label-linear accuracy (33.75%) at lower CPU cost (≈16*D + D*65 FMA per item). Failure: dev accuracy ≤ 0.3375 or Ir/item ≥ 7359.9.
```

```python
"""label-embsum: per-position byte embeddings summed then linear classifier.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyper-parameters (tuned on last 20% of train.in)
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 50
D_CANDIDATES = [8, 16, 32, 64]  # embedding dimension

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_eval(Xtr, ytr, Xva, yva, D, C, rng):
    d = Xtr.shape[1]          # 16
    V = 256                   # byte values
    # Parameters
    E = (rng.normal(0, 0.01, (d, V, D))).astype(np.float32)
    W = (rng.normal(0, 0.01, (D, C))).astype(np.float32)
    b = np.zeros(C, np.float32)
    # Adam state
    mE, vE = np.zeros_like(E), np.zeros_like(E)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    n = Xtr.shape[0]
    best_acc, best_params = -1, None
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = Xtr[idx], ytr[idx]
            # Forward
            h = np.zeros((len(yb), D), np.float32)
            for pos in range(d):
                h += E[pos, xb[:, pos]]
            scores = h @ W + b
            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            probs /= len(yb)
            # Gradients
            gW = h.T @ probs + L2 * W
            gb = probs.sum(axis=0)
            gh = probs @ W.T
            gE = np.zeros_like(E)
            for pos in range(d):
                for i in range(V):
                    mask = (xb[:, pos] == i)
                    if mask.any():
                        gE[pos, i] = gh[mask].sum(axis=0)
                gE[pos] += L2 * E[pos]
            # Adam update
            t += 1
            lr_t = LR * math.sqrt(1 - 0.999**t) / (1 - 0.9**t)
            for param, grad, m, v in [(E, gE, mE, vE), (W, gW, mW, vW), (b, gb, mb, vb)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                param -= lr_t * m / (np.sqrt(v) + 1e-8)
        # Validation accuracy
        hva = np.zeros((Xva.shape[0], D), np.float32)
        for pos in range(d):
            hva += E[pos, Xva[:, pos]]
        pred = (hva @ W + b).argmax(axis=1)
        acc = (pred == yva).mean()
        if acc > best_acc:
            best_acc = acc
            best_params = (E.copy(), W.copy(), b.copy())
    return best_acc, best_params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 16
    n = X.shape[0]
    split = int(0.8 * n)
    Xtr, ytr = X[:split], y[:split]
    Xva, yva = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    best_D, best_acc, best_params = None, -1, None
    for D in D_CANDIDATES:
        acc, params = train_eval(Xtr, ytr, Xva, yva, D, C, rng)
        if acc > best_acc:
            best_acc, best_D, best_params = acc, D, params
    # Retrain on full data with best_D
    _, full_params = train_eval(X, y, X[:1], y[:1], best_D, C, rng)  # dummy validation
    E, W, b = full_params
    # Pack weights.bin
    # Magic: 0x454D4253 ("EMBS" little-endian), d, D, C
    magic = 0x454D4253
    header = struct.pack("<4i", magic, d, best_D, C)
    # Embeddings: d * V * D float32 (row-major: pos, byte, dim)
    emb_bytes = E.astype("<f4").tobytes()
    # W: D * C float32 (row-major)
    w_bytes = W.astype("<f4").tobytes()
    # b: C float32
    b_bytes = b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + w_bytes + b_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-embsum inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h, 1, 4, f);
    fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x454D4253) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D <= 0 || D > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)d * 256 * D;
    const float *b = W + (long)D * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = (float *)malloc(sizeof(float) * D);
    float *scores = (float *)malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, sizeof(float) * D);
        for (int pos_i = 0; pos_i < d; pos_i++) {
            const float *emb = E + ((long)pos_i * 256 + xi[pos_i]) * D;
            for (int j = 0; j < D; j++) h[j] += emb[j];
        }
        memcpy(scores, b, sizeof(float) * C);
        for (int j = 0; j < D; j++) {
            float hj = h[j];
            const float *wj = W + (long)j * C;
            for (int c = 0; c < C; c++) scores[c] += hj * wj[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```