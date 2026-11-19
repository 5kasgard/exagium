```hypothesis
Family: label-mlp2 (two-hidden-layer MLP with ReLU, 32 and 16 units).  
Why: The 8x8 optdigits pixels have local spatial structure; a tiny MLP with two hidden layers can capture feature interactions that a single linear layer or 16-unit MLP miss, while staying far cheaper than GBDT/knn.  
Failure signal: If dev accuracy ≤ 0.962 (label-linear) or CPU cost ≥ 2× label-mlp16, the extra depth is not justified.
```

```python
"""label-mlp2: two-hidden-layer MLP (32, 16 ReLU) for fixed-byte label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyper-parameters chosen on last 20% of train.in (file order), then retrained on all.
Weights format (little-endian):
  magic 0x4D4C5032 ('MLP2'), d, H1, H2, C
  float32: W1[d*H1], b1[H1], W2[H1*H2], b2[H2], W3[H2*C], b3[C]"""
import json, math, struct, sys

import numpy as np

SEED = 1
LR_GRID = [1e-3, 3e-3, 1e-2]
L2_GRID = [1e-4, 1e-3]
EPOCHS_MAX = 100
BATCH = 64
PATIENCE = 5
H1_FIXED, H2_FIXED = 32, 16

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def encode(X, types):
    """Return list of per-byte encoding matrices (256 x width) and total width."""
    enc, widths = [], []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256) - mu) / sd)[:, None]
            widths.append(1)
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), np.float32)
            E[vals, np.arange(len(vals))] = 1.0
            widths.append(len(vals))
        enc.append(E.astype(np.float32))
    return enc, sum(widths)

def forward(z, W1, b1, W2, b2, W3, b3):
    a1 = z @ W1 + b1
    h1 = np.maximum(a1, 0)
    a2 = h1 @ W2 + b2
    h2 = np.maximum(a2, 0)
    logits = h2 @ W3 + b3
    return logits, h1, h2

def loss_grad(logits, y, W3, h2, h1, z, L2, W1, W2):
    n = logits.shape[0]
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n
    g3 = h2.T @ probs + L2 * W3
    b3g = probs.sum(0)
    gh2 = probs @ W3.T * (h2 > 0)
    g2 = h1.T @ gh2 + L2 * W2
    b2g = gh2.sum(0)
    gh1 = gh2 @ W2.T * (h1 > 0)
    g1 = z.T @ gh1 + L2 * W1
    b1g = gh1.sum(0)
    return [g1, b1g, g2, b2g, g3, b3g]

def accuracy(logits, y):
    return (logits.argmax(1) == y).mean()

def train_split(X, y, tune_frac=0.2):
    n = X.shape[0]
    split = int(n * (1 - tune_frac))
    return X[:split], y[:split], X[split:], y[split:]

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]

    # Encode once
    enc, k = encode(X, card["input"]["types"])
    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

    # Split for tuning
    Ztr, ytr, Zva, yva = train_split(Z, y)
    ntr = Ztr.shape[0]

    rng = np.random.default_rng(SEED)
    best_acc, best_cfg = -1, None

    for lr in LR_GRID:
        for L2 in L2_GRID:
            # Init
            W1 = (rng.uniform(-1, 1, (k, H1_FIXED)) * math.sqrt(6.0 / k)).astype(np.float32)
            b1 = np.zeros(H1_FIXED, np.float32)
            W2 = (rng.uniform(-1, 1, (H1_FIXED, H2_FIXED)) * math.sqrt(6.0 / H1_FIXED)).astype(np.float32)
            b2 = np.zeros(H2_FIXED, np.float32)
            W3 = (rng.uniform(-1, 1, (H2_FIXED, C)) * math.sqrt(6.0 / H2_FIXED)).astype(np.float32)
            b3 = np.zeros(C, np.float32)
            params = [W1, b1, W2, b2, W3, b3]
            state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
            t = 0
            best_va, wait = -1, 0

            for epoch in range(EPOCHS_MAX):
                order = rng.permutation(ntr)
                for s in range(0, ntr, BATCH):
                    idx = order[s:s+BATCH]
                    z, yy = Ztr[idx], ytr[idx]
                    logits, h1, h2 = forward(z, W1, b1, W2, b2, W3, b3)
                    grads = loss_grad(logits, yy, W3, h2, h1, z, L2, W1, W2)
                    t += 1
                    for i, (p, gr) in enumerate(zip(params, grads)):
                        m, v = state[i]
                        m *= 0.9; m += 0.1 * gr
                        v *= 0.999; v += 0.001 * gr * gr
                        m_hat = m / (1 - 0.9 ** t)
                        v_hat = v / (1 - 0.999 ** t)
                        p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                # Validation
                va_logits, _, _ = forward(Zva, W1, b1, W2, b2, W3, b3)
                va_acc = accuracy(va_logits, yva)
                if va_acc > best_va:
                    best_va, wait = va_acc, 0
                else:
                    wait += 1
                    if wait >= PATIENCE:
                        break
            if best_va > best_acc:
                best_acc, best_cfg = best_va, (lr, L2)

    # Retrain on full data with best_cfg
    lr, L2 = best_cfg
    W1 = (rng.uniform(-1, 1, (k, H1_FIXED)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H1_FIXED, np.float32)
    W2 = (rng.uniform(-1, 1, (H1_FIXED, H2_FIXED)) * math.sqrt(6.0 / H1_FIXED)).astype(np.float32)
    b2 = np.zeros(H2_FIXED, np.float32)
    W3 = (rng.uniform(-1, 1, (H2_FIXED, C)) * math.sqrt(6.0 / H2_FIXED)).astype(np.float32)
    b3 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = Z.shape[0]
    samples = min(600000, n * EPOCHS_MAX)
    for _ in range(math.ceil(samples / n)):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            z, yy = Z[idx], y[idx]
            logits, h1, h2 = forward(z, W1, b1, W2, b2, W3, b3)
            grads = loss_grad(logits, yy, W3, h2, h1, z, L2, W1, W2)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Flatten encodings into lookup tables per byte
    off = 0
    T = []
    for E in enc:
        w = E.shape[1]
        T.append((E @ W1[off:off+w]).astype(np.float32))
        off += w
    # Pack weights
    out = struct.pack("<5i", 0x4D4C5032, d, H1_FIXED, H2_FIXED, C)
    out += np.concatenate(T).astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    out += W3.astype("<f4").tobytes()
    out += b3.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-mlp2 infer: ./infer weights.bin in.rec N
 * Two hidden layers (H1, H2 ReLU). Input: d bytes. Output: 1 byte class.
 * Weights layout (little-endian float32):
 *   magic 0x4D4C5032, d, H1, H2, C
 *   T[d*256*H1], b1[H1], W2[H1*H2], b2[H2], W3[H2*C], b3[C] */
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
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x4D4C5032) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H1 + H1 + (long)H1 * H2 + H2 + (long)H2 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *T   = (const float *)(hd + 5);
    const float *b1  = T + (long)d * 256 * H1;
    const float *W2  = b1 + H1;
    const float *b2  = W2 + (long)H1 * H2;
    const float *W3  = b2 + H2;
    const float *b3  = W3 + (long)H2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h1 = malloc(sizeof(float) * H1);
    float *h2 = malloc(sizeof(float) * H2);
    float *sc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int u = 0; u < H1; u++) h1[u] = b1[u];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H1;
            for (int u = 0; u < H1; u++) h1[u] += t[u];
        }
        for (int u = 0; u < H1; u++) if (h1[u] < 0.0f) h1[u] = 0.0f;

        for (int v = 0; v < H2; v++) h2[v] = b2[v];
        for (int u = 0; u < H1; u++) {
            float hu = h1[u];
            const float *w = W2 + (long)u * H2;
            for (int v = 0; v < H2; v++) h2[v] += hu * w[v];
        }
        for (int v = 0; v < H2; v++) if (h2[v] < 0.0f) h2[v] = 0.0f;

        for (int c = 0; c < C; c++) sc[c] = b3[c];
        for (int v = 0; v < H2; v++) {
            float hv = h2[v];
            const float *w = W3 + (long)v * C;
            for (int c = 0; c < C; c++) sc[c] += hv * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```