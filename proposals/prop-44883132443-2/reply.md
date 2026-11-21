```hypothesis
Family: label-ngram-hash – linear model on hashed n-gram features (unigrams, bigrams, trigrams) from the 16-byte context.
Why: captures local sequential patterns (character n-grams) that a pure unigram linear model misses, while staying lightweight via feature hashing (fixed-width weight matrix).
Failure: if dev accuracy ≤ label-linear (0.3375) or inference cost (ns/item) exceeds label-linear by >2×.
```

```python
import json, math, struct, sys
import numpy as np

# hyperparameter grid for tuning split (last 20% of train.in)
D_GRID = [8192, 16384, 32768, 65536]
LR_GRID = [0.01, 0.03, 0.1]
L2 = 1e-4
BATCH = 64
SEED = 1
MAX_EPOCHS_GRID = 5
MAX_EPOCHS_FULL = 20
SAMPLES_FULL = 600000
NGRAM_MAX = 3
CONTEXT_LEN = 16
MAGIC = 0x4E475248  # 'NGRH'

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def hash_ngram(ngram, D):
    h = 0
    for b in ngram:
        h = (h * 31 + b) & 0xFFFFFFFF
    return h % D

def extract_indices(seq, D):
    """seq: 16 bytes. Return array of 45 hashed indices (unigrams, bigrams, trigrams)."""
    idx = []
    L = len(seq)
    for n in range(1, NGRAM_MAX + 1):
        for start in range(L - n + 1):
            idx.append(hash_ngram(seq[start:start + n], D))
    return np.array(idx, dtype=np.int32)

def train_val_split(X, y, val_frac=0.2):
    n = len(X)
    split = int(n * (1 - val_frac))
    return X[:split], y[:split], X[split:], y[split:]

def evaluate(W, b, X_idx, y):
    n = len(y)
    correct = 0
    for i in range(0, n, BATCH):
        batch_idx = X_idx[i:i+BATCH]
        logits = np.sum(W[batch_idx], axis=1) + b
        pred = np.argmax(logits, axis=1)
        correct += np.sum(pred == y[i:i+BATCH])
    return correct / n

def train_model(X_idx, y, D, C, lr, epochs, rng):
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    n = len(y)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            batch_idx = X_idx[idx]
            yy = y[idx]
            logits = np.sum(W[batch_idx], axis=1) + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            dlogits = probs / len(yy)
            flat_idx = batch_idx.ravel()
            flat_grad = np.repeat(dlogits, NGRAM_MAX * CONTEXT_LEN - (NGRAM_MAX-1)*NGRAM_MAX//2, axis=0)  # 45
            np.add.at(W, flat_idx, flat_grad)
            np.add.at(b, np.arange(C), dlogits.sum(axis=0))
            W += L2 * W
            b += L2 * b
            t += 1
            for param, grad, m, v in [(W, flat_grad, mW, vW), (b, dlogits.sum(axis=0), mb, vb)]:
                if param is W:
                    m *= 0.9; m += 0.1 * grad; v *= 0.999; v += 0.001 * grad * grad
                else:
                    m *= 0.9; m += 0.1 * grad; v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W, b

def main():
    card = json.load(open(sys.argv[1]))
    X_raw = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # split for tuning
    X_train, y_train, X_val, y_val = train_val_split(X_raw, y)

    best_acc, best_D, best_lr = -1, None, None
    for D in D_GRID:
        print(f"Trying D={D}", file=sys.stderr)
        X_train_idx = np.stack([extract_indices(seq, D) for seq in X_train])
        X_val_idx = np.stack([extract_indices(seq, D) for seq in X_val])
        for lr in LR_GRID:
            W, b = train_model(X_train_idx, y_train, D, C, lr, MAX_EPOCHS_GRID, rng)
            acc = evaluate(W, b, X_val_idx, y_val)
            print(f"  lr={lr} val_acc={acc:.4f}", file=sys.stderr)
            if acc > best_acc:
                best_acc, best_D, best_lr = acc, D, lr

    print(f"Best: D={best_D}, lr={best_lr}, val_acc={best_acc:.4f}", file=sys.stderr)

    # retrain on full data
    X_full_idx = np.stack([extract_indices(seq, best_D) for seq in X_raw])
    W, b = train_model(X_full_idx, y, best_D, C, best_lr, MAX_EPOCHS_FULL, rng)

    # save weights
    out = struct.pack("<4i", MAGIC, best_D, C, NGRAM_MAX)
    out += W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

static uint32_t hash_ngram(const unsigned char *seq, int start, int len, uint32_t D) {
    uint32_t h = 0;
    for (int i = 0; i < len; i++) h = h * 31 + seq[start + i];
    return h % D;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x4E475248) { fprintf(stderr, "bad weights\n"); return 3; }
    int D = hd[1], C = hd[2], NGRAM_MAX = hd[3];
    if (C < 1 || C > 256 || NGRAM_MAX != 3) { fprintf(stderr, "bad weights header\n"); return 3; }
    const int CONTEXT_LEN = 16;
    const int NGRAM_COUNT = CONTEXT_LEN + (CONTEXT_LEN-1) + (CONTEXT_LEN-2); // 16+15+14=45
    long expected = 16 + 4L * ((long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *W = (const float *)(hd + 4);
    const float *b = W + (long)D * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    unsigned char *indices = malloc(NGRAM_COUNT);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != CONTEXT_LEN) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, CONTEXT_LEN); return 3; }
        int idx_pos = 0;
        for (int n = 1; n <= NGRAM_MAX; n++) {
            for (int start = 0; start <= CONTEXT_LEN - n; start++) {
                indices[idx_pos++] = hash_ngram(xi, start, n, D);
            }
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int j = 0; j < NGRAM_COUNT; j++) {
            const float *wrow = W + (long)indices[j] * C;
            for (int c = 0; c < C; c++) scores[c] += wrow[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```