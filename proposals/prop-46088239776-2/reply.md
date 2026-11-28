```hypothesis
We propose a "label-hashing" family: a linear model on hashed features that includes both unigram (position, byte) and bigram (position, byte_i, byte_{i+1}) features. This captures local sequential interactions that the pure linear (unigram-only) model misses, while feature hashing keeps the model size fixed and inference fast. If dev accuracy does not exceed label-linear (0.3375), it suggests bigram interactions are either not predictive enough or hashing collisions degrade performance.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then trained on all train)
HASH_SIZE = 65536          # 2^16
SEED = 0x12345678
LR = 0.01
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
VAL_FRACTION = 0.2         # last 20% for tuning

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def hash_feature(tup, seed, hash_size):
    h = seed
    for x in tup:
        h = (h * 31 + x) & 0xFFFFFFFF
    return h % hash_size

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]
    assert d == 16

    # Precompute feature indices for all examples: 16 unigrams + 15 bigrams = 31 features
    feat_indices = np.empty((n, 31), dtype=np.int32)
    for i in range(n):
        xi = X[i]
        idx = 0
        for pos in range(16):
            feat_indices[i, idx] = hash_feature((0, pos, xi[pos]), SEED, HASH_SIZE)
            idx += 1
        for pos in range(15):
            feat_indices[i, idx] = hash_feature((1, pos, xi[pos], xi[pos+1]), SEED, HASH_SIZE)
            idx += 1

    # Split into train/val (last 20% for validation)
    val_start = int(n * (1 - VAL_FRACTION))
    train_idx = np.arange(val_start)
    val_idx = np.arange(val_start, n)
    X_train, y_train = feat_indices[train_idx], y[train_idx]
    X_val, y_val = feat_indices[val_idx], y[val_idx]
    n_train = len(X_train)

    # Initialize weights
    rng = np.random.default_rng(1)
    W = rng.normal(0, 0.01, size=(HASH_SIZE, C)).astype(np.float32)

    # Adam state
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best_val_acc = -1.0
    best_W = None
    best_epoch = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # Shuffle training data
        order = rng.permutation(n_train)
        for s in range(0, n_train, BATCH):
            idx = order[s:s+BATCH]
            feats = X_train[idx]          # (batch, 31)
            yy = y_train[idx]             # (batch,)
            batch_size = len(yy)

            # Forward: logits = sum of weight rows for each feature
            # W[feats] -> (batch, 31, C)
            logits = W[feats].sum(axis=1)  # (batch, C)

            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(batch_size), yy] -= 1.0
            grad_logits = probs / batch_size  # (batch, C)

            # Gradient wrt W: scatter-add
            grad_W = np.zeros_like(W)
            np.add.at(grad_W, feats.reshape(-1), grad_logits.repeat(31, axis=0).reshape(-1, C))
            grad_W += L2 * W

            # Adam update
            t += 1
            m = beta1 * m + (1 - beta1) * grad_W
            v = beta2 * v + (1 - beta2) * (grad_W * grad_W)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            W -= LR * m_hat / (np.sqrt(v_hat) + eps)

        # Validation accuracy
        val_logits = W[X_val].sum(axis=1)
        val_pred = val_logits.argmax(axis=1)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_W = W.copy()
            best_epoch = epoch

    # Retrain on all data for best_epoch epochs
    W = rng.normal(0, 0.01, size=(HASH_SIZE, C)).astype(np.float32)
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    t = 0
    all_feats, all_y = feat_indices, y
    for epoch in range(1, best_epoch + 1):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            feats = all_feats[idx]
            yy = all_y[idx]
            batch_size = len(yy)
            logits = W[feats].sum(axis=1)
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(batch_size), yy] -= 1.0
            grad_logits = probs / batch_size
            grad_W = np.zeros_like(W)
            np.add.at(grad_W, feats.reshape(-1), grad_logits.repeat(31, axis=0).reshape(-1, C))
            grad_W += L2 * W
            t += 1
            m = beta1 * m + (1 - beta1) * grad_W
            v = beta2 * v + (1 - beta2) * (grad_W * grad_W)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            W -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # Save weights: magic, hash_size, C, seed, then W (row-major float32)
    magic = 0x48415348  # "HASH"
    out = struct.pack("<4i", magic, HASH_SIZE, C, SEED)
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

static uint32_t hash_feature(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t seed, uint32_t hash_size) {
    uint32_t h = seed;
    h = h * 31 + a;
    h = h * 31 + b;
    h = h * 31 + c;
    h = h * 31 + d;
    return h % hash_size;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x48415348) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t hash_size = hd[1];
    uint32_t C = hd[2];
    uint32_t seed = hd[3];
    if (C < 1 || C > 256 || nw != 16 + 4L * (long)hash_size * C) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *W = (const float *)(hd + 4);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 16) { fprintf(stderr, "item %ld is %u bytes, not 16\n", i, k); return 3; }

        memset(scores, 0, C * sizeof(float));

        // 16 unigrams: (0, pos, byte)
        for (int pos_u = 0; pos_u < 16; pos_u++) {
            uint32_t idx = hash_feature(0, pos_u, xi[pos_u], 0, seed, hash_size);
            const float *w = W + (long)idx * C;
            for (int c = 0; c < (int)C; c++) scores[c] += w[c];
        }
        // 15 bigrams: (1, pos, byte1, byte2)
        for (int pos_b = 0; pos_b < 15; pos_b++) {
            uint32_t idx = hash_feature(1, pos_b, xi[pos_b], xi[pos_b+1], seed, hash_size);
            const float *w = W + (long)idx * C;
            for (int c = 0; c < (int)C; c++) scores[c] += w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < (int)C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```