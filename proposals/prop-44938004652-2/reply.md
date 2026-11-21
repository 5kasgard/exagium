```hypothesis
Family: label-adjhash (adjacent-pair feature hashing with linear model).
Why: Character-level language modeling benefits from local n-gram interactions. This model hashes 16 single-byte positions and 15 adjacent pairs into a fixed-size table, capturing pairwise dependencies while keeping inference fast (31 table lookups per record). It should beat label-linear (0.3375) by modeling bigrams, and stay cheaper than label-gbdt.
Failure signal: Dev accuracy ≤ 0.3375 or instructions/item ≥ 2× label-linear.
```

```python
"""label-adjhash: linear model on hashed unigram+adjacent-bigram features.
train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only.
Hyperparameters (hash_size, L2) chosen on last 20% of train.in, then retrained on all."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2_GRID = 3e-3, [1e-4, 1e-3, 1e-2]
HASH_SIZES = [16384, 32768, 65536]
BATCH, SAMPLES, MAX_EPOCHS = 64, 600000, 100
MAGIC = 0x41444A48  # 'ADJH'

def splitmix64(state: np.uint64) -> np.uint64:
    state = (state + np.uint64(0x9e3779b97f4a7c15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = state
    z = (z ^ (z >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
    z = (z ^ (z >> 27)) * np.uint64(0x94d049bb133111eb)
    return (z ^ (z >> 31)) & np.uint64(0xFFFFFFFFFFFFFFFF)

def hash_feature(vals, hash_size, rng_state):
    """vals: list of ints. Returns (index, new_rng_state)."""
    state = rng_state
    for v in vals:
        state = (state + np.uint64(v)) & np.uint64(0xFFFFFFFFFFFFFFFF)
        state = splitmix64(state)
    return int(state % hash_size), state

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def make_features(X, hash_size):
    """X: (n, 16) uint8. Returns list of length n, each a list of 31 feature indices."""
    n = X.shape[0]
    feats = []
    for i in range(n):
        row = X[i]
        idxs = []
        state = np.uint64(0)
        # 16 unigrams: (1, pos, byte)
        for pos in range(16):
            idx, state = hash_feature([1, pos, row[pos]], hash_size, state)
            idxs.append(idx)
        # 15 adjacent bigrams: (2, pos, byte, pos+1, byte)
        for pos in range(15):
            idx, state = hash_feature([2, pos, row[pos], pos+1, row[pos+1]], hash_size, state)
            idxs.append(idx)
        feats.append(idxs)
    return feats

def train_linear(Z_feats, y, C, hash_size, lr, l2, epochs, batch, rng):
    """Z_feats: list of list of feature indices. Returns W (hash_size, C), b (C)."""
    W = np.zeros((hash_size, C), np.float32)
    b = np.zeros(C, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    n = len(Z_feats)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            # accumulate gradients
            gW = np.zeros_like(W)
            gb = np.zeros_like(b)
            for i in idx:
                feats = Z_feats[i]
                yy = y[i]
                # scores
                scores = b.copy()
                for f in feats:
                    scores += W[f]
                # softmax grad
                scores -= scores.max()
                exp = np.exp(scores)
                probs = exp / exp.sum()
                probs[yy] -= 1.0
                # gradients
                gb += probs
                for f in feats:
                    gW[f] += probs
            gW /= len(idx)
            gb /= len(idx)
            gW += l2 * W
            # Adam
            t += 1
            for param, grad, m, v in [(W, gW, mW, vW), (b, gb, mb, vb)]:
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9**t)
                v_hat = v / (1 - 0.999**t)
                param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W, b

def eval_acc(Z_feats, y, W, b):
    correct = 0
    for feats, yy in zip(Z_feats, y):
        scores = b.copy()
        for f in feats:
            scores += W[f]
        if scores.argmax() == yy:
            correct += 1
    return correct / len(y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc, best_hs, best_l2 = -1, HASH_SIZES[0], L2_GRID[0]
    # quick grid search on validation (5 epochs each)
    for hs in HASH_SIZES:
        Z_tr = make_features(X_tr, hs)
        Z_va = make_features(X_va, hs)
        for l2 in L2_GRID:
            W, b = train_linear(Z_tr, y_tr, C, hs, LR, l2, 5, BATCH, rng)
            acc = eval_acc(Z_va, y_va, W, b)
            if acc > best_acc:
                best_acc, best_hs, best_l2 = acc, hs, l2
    # retrain on full data with best hyperparameters
    Z_full = make_features(X, best_hs)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    W, b = train_linear(Z_full, y, C, best_hs, LR, best_l2, epochs, BATCH, rng)

    # write weights
    out = struct.pack("<4i", MAGIC, X.shape[1], best_hs, C)
    out += W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-adjhash inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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

/* splitmix64 hash */
static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static int hash_feature(const int *vals, int nvals, uint32_t hash_size, uint64_t *state) {
    for (int i = 0; i < nvals; i++) {
        *state += (uint64_t)vals[i];
        *state = splitmix64(state);
    }
    return (int)(*state % hash_size);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x41444A48) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], hash_size = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * ((long)hash_size * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W = (const float *)(hd + 4);
    const float *b = W + (long)hash_size * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);
    int vals[5];  // max feature tuple length

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(scores, b, sizeof(float) * C);
        uint64_t state = 0;

        /* 16 unigrams */
        for (int posi = 0; posi < 16; posi++) {
            vals[0] = 1; vals[1] = posi; vals[2] = xi[posi];
            int idx = hash_feature(vals, 3, hash_size, &state);
            const float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }
        /* 15 adjacent bigrams */
        for (int posi = 0; posi < 15; posi++) {
            vals[0] = 2; vals[1] = posi; vals[2] = xi[posi]; vals[3] = posi+1; vals[4] = xi[posi+1];
            int idx = hash_feature(vals, 5, hash_size, &state);
            const float *w = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```