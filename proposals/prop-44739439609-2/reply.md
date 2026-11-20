```hypothesis
Family: Hashed N-gram Linear (hash-ngram-linear). Uses a fixed-size hash table (M=16384) to represent positional unigrams (16) and bigrams (15) from the 16-byte context. A linear model on these hashed features captures local character interactions (e.g., "th", "he") while keeping inference fast: 31 feature lookups per item, each adding a 65-dim vector. Expected to beat label-linear (0.3375) because bigrams model strong local dependencies in English text. If dev accuracy ≤ 0.34, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameter grid for tuning on the last 20% of train.in
GRID = [
    {"M": 8192, "lr": 1e-3},
    {"M": 16384, "lr": 1e-3},
    {"M": 32768, "lr": 1e-3},
    {"M": 8192, "lr": 3e-3},
    {"M": 16384, "lr": 3e-3},
    {"M": 32768, "lr": 3e-3},
]
BATCH = 64
TUNE_EPOCHS = 5
FINAL_EPOCHS = 20
L2 = 1e-4
SEED = 12345

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def fnv1a_32(*vals):
    h = 0x811c9dc5
    for v in vals:
        h ^= v
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

def compute_indices(X, M):
    """X: (n, 16) uint8. Returns (n, 31) int32 indices."""
    n = X.shape[0]
    indices = np.empty((n, 31), dtype=np.int32)
    for i in range(n):
        xi = X[i]
        pos = 0
        # unigrams
        for p in range(16):
            indices[i, pos] = fnv1a_32(p, xi[p]) % M
            pos += 1
        # bigrams
        for p in range(15):
            indices[i, pos] = fnv1a_32(p, xi[p], xi[p+1]) % M
            pos += 1
    return indices

def train_model(X, y, indices, M, C, lr, epochs, rng, batch_size=BATCH, l2=L2):
    n = X.shape[0]
    W = (rng.uniform(-1, 1, (M, C)) * math.sqrt(6.0 / M)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            batch_idx = indices[idx]  # (batch, 31)
            batch_y = y[idx]
            batch_size_curr = len(batch_y)
            # forward
            scores = W[batch_idx].sum(axis=1) + b  # (batch, C)
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(batch_size_curr), batch_y] -= 1.0
            d_scores = probs / batch_size_curr  # (batch, C)
            # gradient for b
            grad_b = d_scores.sum(axis=0)  # (C,)
            # gradient for W
            flat_idx = batch_idx.ravel()  # (batch*31,)
            flat_grad = np.repeat(d_scores, 31, axis=0)  # (batch*31, C)
            grad_W = np.zeros_like(W)
            np.add.at(grad_W, flat_idx, flat_grad)
            grad_W += l2 * W
            # Adam update
            t += 1
            mW = beta1 * mW + (1 - beta1) * grad_W
            vW = beta2 * vW + (1 - beta2) * (grad_W * grad_W)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            mb = beta1 * mb + (1 - beta1) * grad_b
            vb = beta2 * vb + (1 - beta2) * (grad_b * grad_b)
            mb_hat = mb / (1 - beta1 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
    return W, b

def evaluate(W, b, indices, y):
    scores = W[indices].sum(axis=1) + b
    preds = scores.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, 16)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = n - n // 5
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    for cfg in GRID:
        M = cfg["M"]
        lr = cfg["lr"]
        indices_train = compute_indices(X_train, M)
        indices_val = compute_indices(X_val, M)
        W, b = train_model(X_train, y_train, indices_train, M, C, lr, TUNE_EPOCHS, rng)
        acc = evaluate(W, b, indices_val, y_val)
        print(f"M={M}, lr={lr}: val acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
    print(f"Best config: {best_cfg}, acc={best_acc:.4f}", file=sys.stderr)
    M = best_cfg["M"]
    lr = best_cfg["lr"]
    indices_full = compute_indices(X, M)
    W_final, b_final = train_model(X, y, indices_full, M, C, lr, FINAL_EPOCHS, rng)
    magic = 0x484E474C  # 'HNGL'
    header = struct.pack("<iii", magic, M, C)
    weights_bytes = W_final.astype("<f4").tobytes() + b_final.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + weights_bytes)

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

static uint32_t fnv1a_32(uint32_t h, uint32_t v) {
    h ^= v;
    h *= 0x01000193;
    return h;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 12) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x484E474C) { fprintf(stderr, "bad weights: magic mismatch\n"); return 3; }
    int M = hd[1], C = hd[2];
    if (C < 1 || C > 256 || M < 1) { fprintf(stderr, "bad weights: invalid M or C\n"); return 3; }
    long expected = 12 + 4L * M * C + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    float *W = (float *)(wb + 12);
    float *b = W + (long)M * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        // unigrams
        for (int p = 0; p < 16; p++) {
            uint32_t h = 0x811c9dc5;
            h = fnv1a_32(h, (uint32_t)p);
            h = fnv1a_32(h, (uint32_t)xi[p]);
            uint32_t idx = h % (uint32_t)M;
            const float *wrow = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += wrow[c];
        }
        // bigrams
        for (int p = 0; p < 15; p++) {
            uint32_t h = 0x811c9dc5;
            h = fnv1a_32(h, (uint32_t)p);
            h = fnv1a_32(h, (uint32_t)xi[p]);
            h = fnv1a_32(h, (uint32_t)xi[p+1]);
            uint32_t idx = h % (uint32_t)M;
            const float *wrow = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += wrow[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```