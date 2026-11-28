```hypothesis
We propose the "label-hashedngram" family. It hashes unigrams and bigrams from the 16‑byte context into a fixed‑size feature vector, then applies a linear classifier. This captures local character interactions (digrams) crucial for language modeling while keeping inference fast via sparse linear combination. If it fails to beat label‑linear, it suggests pairwise interactions are not sufficient or hashing collisions degrade performance.
```

```python
import json, math, struct, sys
import numpy as np

# ---------- Hash function (splitmix64) ----------
MASK64 = (1 << 64) - 1
def splitmix64(x):
    x = (x + 0x9e3779b97f4a7c15) & MASK64
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9 & MASK64
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb & MASK64
    x = x ^ (x >> 31)
    return x & MASK64

def hash_ngram(pos, vals, D):
    h = splitmix64(pos)
    for v in vals:
        h = splitmix64(h ^ v)
    return h % D

# ---------- Record I/O ----------
def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

# ---------- Feature extraction ----------
def compute_indices(X, d, D):
    """X: (n, d) uint8, returns list of n lists of feature indices (31 per example)"""
    n = X.shape[0]
    idxs = []
    for i in range(n):
        xi = X[i]
        inds = []
        # unigrams
        for p in range(d):
            inds.append(hash_ngram(p, [xi[p]], D))
        # bigrams
        for p in range(d - 1):
            inds.append(hash_ngram(p, [xi[p], xi[p + 1]], D))
        idxs.append(inds)
    return idxs

# ---------- Training ----------
def train_model(X_tr, y_tr, X_val, y_val, D, C, lr, l2, batch, max_epochs, samples, seed):
    rng = np.random.default_rng(seed)
    n_tr = X_tr.shape[0]
    # Precompute feature indices
    tr_idx = compute_indices(X_tr, X_tr.shape[1], D)
    val_idx = compute_indices(X_val, X_val.shape[1], D)

    # Parameters
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def softmax_grad(logits, y):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(y)), y] -= 1.0
        return (probs / len(y)).astype(np.float32)

    for epoch in range(max_epochs):
        if t * batch >= samples:
            break
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, batch):
            idx = order[s:s + batch]
            if len(idx) == 0:
                continue
            # Build logits
            logits = np.tile(b, (len(idx), 1))
            for b_i, example_idx in enumerate(idx):
                for f_idx in tr_idx[example_idx]:
                    logits[b_i] += W[f_idx]
            g = softmax_grad(logits, y_tr[idx])
            # Gradients
            grad_W = np.zeros_like(W)
            grad_b = g.sum(axis=0)
            for b_i, example_idx in enumerate(idx):
                for f_idx in tr_idx[example_idx]:
                    grad_W[f_idx] += g[b_i]
            grad_W += l2 * W
            grads = [grad_W, grad_b]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Validation accuracy
        val_logits = np.tile(b, (len(X_val), 1))
        for i, inds in enumerate(val_idx):
            for f_idx in inds:
                val_logits[i] += W[f_idx]
        val_pred = val_logits.argmax(axis=1)
        val_acc = (val_pred == y_val).mean()
    return val_acc, params

# ---------- Main ----------
def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load all training data
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_all = len(X_all)
    # Split: last 20% for tuning
    split = int(0.8 * n_all)
    X_tr, y_tr = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]

    # Grid search on tuning split
    best_acc = -1
    best_params = None
    best_D = None
    best_lr = None
    for D in [2048, 4096, 8192]:
        for lr in [1e-3, 3e-3]:
            acc, params = train_model(X_tr, y_tr, X_val, y_val, D, C, lr, 1e-4, 64, 20, 200000, 1)
            if acc > best_acc:
                best_acc = acc
                best_params = params
                best_D = D
                best_lr = lr
    print(f"Best tuning: D={best_D}, lr={best_lr}, acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on all data with best config
    final_acc, final_params = train_model(X_all, y_all, X_val, y_val, best_D, C, best_lr, 1e-4, 64, 100, 600000, 1)
    W, b = final_params

    # Write weights.bin
    # Format: magic(4), d(4), D(4), C(4), bias[C] (float32), W[D][C] (float32 row-major)
    magic = 0x48534148  # 'HASH' little-endian
    out = struct.pack("<4i", magic, d, best_D, C)
    out += b.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
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

// splitmix64
static uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
    x = x ^ (x >> 31);
    return x;
}

static uint32_t hash_ngram(int pos, int len, const unsigned char *bytes, int D) {
    uint64_t h = splitmix64((uint64_t)pos);
    for (int i = 0; i < len; i++) {
        h = splitmix64(h ^ (uint64_t)bytes[i]);
    }
    return (uint32_t)(h % (uint64_t)D);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x48534148) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + 4L * C + 4L * D * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *bias = (const float *)(hd + 4);
    const float *W = bias + C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Start with bias
        for (int c = 0; c < C; c++) acc[c] = bias[c];
        // Unigrams
        for (int p = 0; p < d; p++) {
            uint32_t idx = hash_ngram(p, 1, xi + p, D);
            const float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) acc[c] += w[c];
        }
        // Bigrams
        for (int p = 0; p < d - 1; p++) {
            uint32_t idx = hash_ngram(p, 2, xi + p, D);
            const float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) acc[c] += w[c];
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (acc[c] > acc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```