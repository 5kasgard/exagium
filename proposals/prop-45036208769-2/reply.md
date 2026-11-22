```hypothesis
Family: label-factorized-mlp. Uses a global token embedding (256×E) plus a positional embedding (16×E) that are added per position, summed over the 16-byte context, then passed through a single ReLU hidden layer (size H) to 65 outputs. This factorization drastically cuts parameters vs. per-position tables (label-linear), allowing larger H for the same memory and better regularization. The summed embedding captures positional n-gram statistics efficiently. If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds label-linear by >20%, the hypothesis fails.
```

```python
"""label-factorized-mlp: factorized embeddings + 1-hidden-layer MLP for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid searched on the tuning split (last 20% of train.in)
GRID = {
    "E": [32, 64],          # embedding dimension
    "H": [64, 128],         # hidden layer size
    "lr": [1e-3, 3e-3],     # Adam learning rate
    "L2": [1e-4, 3e-4],     # weight decay
    "batch": [128],         # minibatch size
    "epochs": [30],         # max epochs (early stop on tuning loss)
}
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def accuracy(logits, y):
    return np.mean(np.argmax(logits, axis=1) == y)

def train_eval(X_tr, y_tr, X_va, y_va, C, E, H, lr, L2, batch, epochs, rng):
    n, d = X_tr.shape
    # Initialize weights
    tok_emb = rng.normal(0, 0.02, (256, E)).astype(np.float32)   # token embedding
    pos_emb = rng.normal(0, 0.02, (d, E)).astype(np.float32)     # positional embedding
    W1 = rng.normal(0, math.sqrt(2.0/E), (E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, math.sqrt(2.0/H), (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [tok_emb, pos_emb, W1, b1, W2, b2]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    best_va_acc, best_params = -1.0, None
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            xb, yb = X_tr[idx], y_tr[idx]
            bs = len(yb)
            # Forward
            # Sum of (tok_emb[x] + pos_emb[pos]) over 16 positions
            h = np.zeros((bs, E), np.float32)
            for j in range(d):
                h += tok_emb[xb[:, j]] + pos_emb[j]
            a = h @ W1 + b1
            h_relu = np.maximum(a, 0)
            logits = h_relu @ W2 + b2
            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs
            # Backward
            gW2 = h_relu.T @ probs + L2 * W2
            gb2 = probs.sum(axis=0)
            gh_relu = probs @ W2.T
            gh = gh_relu * (a > 0)
            gW1 = h.T @ gh + L2 * W1
            gb1 = gh.sum(axis=0)
            # Gradient w.r.t. summed embedding h
            gh_sum = gh @ W1.T
            # Distribute to token and positional embeddings
            g_tok = np.zeros_like(tok_emb)
            g_pos = np.zeros_like(pos_emb)
            for j in range(d):
                np.add.at(g_tok, xb[:, j], gh_sum)
                g_pos[j] += gh_sum.sum(axis=0)
            grads = [g_tok, g_pos, gW1, gb1, gW2, gb2]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9**t)
                v_hat = v[i] / (1 - 0.999**t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Validation accuracy
        h = np.zeros((len(X_va), E), np.float32)
        for j in range(d):
            h += tok_emb[X_va[:, j]] + pos_emb[j]
        a = h @ W1 + b1
        h_relu = np.maximum(a, 0)
        logits = h_relu @ W2 + b2
        va_acc = accuracy(logits, y_va)
        if va_acc > best_va_acc:
            best_va_acc = va_acc
            best_params = [p.copy() for p in params]
    return best_va_acc, best_params

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = len(X)
    # Tuning split: last 20%
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    best_score, best_cfg, best_weights = -1.0, None, None
    for E in GRID["E"]:
        for H in GRID["H"]:
            for lr in GRID["lr"]:
                for L2 in GRID["L2"]:
                    for batch in GRID["batch"]:
                        for epochs in GRID["epochs"]:
                            va_acc, params = train_eval(X_tr, y_tr, X_va, y_va, C, E, H, lr, L2, batch, epochs, rng)
                            if va_acc > best_score:
                                best_score, best_cfg, best_weights = va_acc, (E, H, lr, L2, batch, epochs), params
    print(f"Best tuning acc: {best_score:.4f} cfg: {best_cfg}", file=sys.stderr)
    # Retrain on full training set with best config
    E, H, lr, L2, batch, epochs = best_cfg
    rng_full = np.random.default_rng(SEED)
    # Re-initialize with same seed for reproducibility
    tok_emb = rng_full.normal(0, 0.02, (256, E)).astype(np.float32)
    pos_emb = rng_full.normal(0, 0.02, (d, E)).astype(np.float32)
    W1 = rng_full.normal(0, math.sqrt(2.0/E), (E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng_full.normal(0, math.sqrt(2.0/H), (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [tok_emb, pos_emb, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    for ep in range(epochs):
        order = rng_full.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            xb, yb = X[idx], y[idx]
            bs = len(yb)
            h = np.zeros((bs, E), np.float32)
            for j in range(d):
                h += tok_emb[xb[:, j]] + pos_emb[j]
            a = h @ W1 + b1
            h_relu = np.maximum(a, 0)
            logits = h_relu @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs
            gW2 = h_relu.T @ probs + L2 * W2
            gb2 = probs.sum(axis=0)
            gh_relu = probs @ W2.T
            gh = gh_relu * (a > 0)
            gW1 = h.T @ gh + L2 * W1
            gb1 = gh.sum(axis=0)
            gh_sum = gh @ W1.T
            g_tok = np.zeros_like(tok_emb)
            g_pos = np.zeros_like(pos_emb)
            for j in range(d):
                np.add.at(g_tok, xb[:, j], gh_sum)
                g_pos[j] += gh_sum.sum(axis=0)
            grads = [g_tok, g_pos, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9**t)
                v_hat = v[i] / (1 - 0.999**t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    tok_emb, pos_emb, W1, b1, W2, b2 = params
    # Write weights: magic, d, E, H, C; then tok_emb[256*E], pos_emb[d*E], W1[E*H], b1[H], W2[H*C], b2[C] (all float32 LE)
    magic = 0x46434D4C  # "LFML" little-endian
    expected = 16 + 4*(256*E + d*E + E*H + H + H*C + C)
    out = struct.pack("<5i", magic, d, E, H, C)
    out += tok_emb.astype("<f4").tobytes()
    out += pos_emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    assert len(out) == expected
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-factorized-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format: magic(0x46434D4C), d, E, H, C (int32 each);
 *   tok_emb[256*E], pos_emb[d*E], W1[E*H], b1[H], W2[H*C], b2[C] (float32 LE). */
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
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x46434D4C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20L + 4L*(256L*E + (long)d*E + (long)E*H + H + (long)H*C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *tok_emb = (const float *)(hd + 5);
    const float *pos_emb = tok_emb + 256L * E;
    const float *W1 = pos_emb + (long)d * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * E);
    float *a = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // h = sum_j (tok_emb[xi[j]] + pos_emb[j])
        memset(h, 0, sizeof(float) * E);
        for (int j = 0; j < d; j++) {
            const float *te = tok_emb + (long)xi[j] * E;
            const float *pe = pos_emb + (long)j * E;
            for (int e = 0; e < E; e++) h[e] += te[e] + pe[e];
        }
        // a = h @ W1 + b1
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int e = 0; e < E; e++) sum += h[e] * W1[(long)e * H + u];
            a[u] = sum > 0.0f ? sum : 0.0f; // ReLU
        }
        // logits = a @ W2 + b2
        memcpy(logits, b2, sizeof(float) * C);
        for (int u = 0; u < H; u++) {
            if (a[u] == 0.0f) continue;
            const float *w2u = W2 + (long)u * C;
            for (int c = 0; c < C; c++) logits[c] += a[u] * w2u[c];
        }
        // argmax
        best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```