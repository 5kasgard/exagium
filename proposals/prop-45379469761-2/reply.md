```hypothesis
Family: label-ngram (linear model with unigram + hashed bigram features).
Why: Tinyshakespeare character prediction benefits from local n-gram statistics (e.g., "th", "he", "in"). Adding bigram features to the linear unigram model captures pairwise interactions while keeping inference fast (O(d*C + (d-1)*C) with small lookup tables). The model remains a single linear layer, so it should beat label-linear (0.3375) on accuracy with only a modest increase in CPU cost.
Failure mode: If dev accuracy does not exceed label-linear, it indicates that bigram hashing collisions lose too much information or that higher-order interactions are needed.
```

```python
"""label-ngram: linear model with unigram and hashed bigram features for fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x4E47524D, d, H=0, C, B, U; float32 T_uni[d][256][C], T_bi[d-1][B][C], bias[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR_GRID = [1e-3, 3e-3, 1e-2]
L2_GRID = [1e-4, 1e-3]
B_GRID = [32, 64, 128]          # bigram buckets (powers of two)
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_vocab(X, d):
    """Return list of dicts: value -> feature index for each position."""
    vocab = []
    for j in range(d):
        vals = np.unique(X[:, j])
        vocab.append({v: idx for idx, v in enumerate(vals)})
    return vocab

def bigram_hash(b1, b2, B):
    """Deterministic hash to [0, B-1], B power of two."""
    return ((b1 << 8) | b2) * 31 & (B - 1)

def make_features(X, vocab, B, d):
    """Construct dense feature matrix Z (n, n_features)."""
    n = X.shape[0]
    # count unigram features
    uni_sizes = [len(v) for v in vocab]
    uni_offsets = np.cumsum([0] + uni_sizes)
    n_uni = uni_offsets[-1]
    n_bi = (d - 1) * B
    n_feat = n_uni + n_bi
    Z = np.zeros((n, n_feat), dtype=np.float32)
    # unigrams
    for j in range(d):
        off = uni_offsets[j]
        for i, v in enumerate(X[:, j]):
            if v in vocab[j]:
                Z[i, off + vocab[j][v]] = 1.0
    # bigrams
    bi_off = n_uni
    for j in range(d - 1):
        for i in range(n):
            bucket = bigram_hash(X[i, j], X[i, j + 1], B)
            Z[i, bi_off + j * B + bucket] = 1.0
    return Z, n_uni, n_bi

def train_linear(Z, y, C, lr, l2, rng):
    """Adam on linear softmax. Returns W (n_feat, C), b (C)."""
    n, n_feat = Z.shape
    W = (rng.uniform(-1, 1, (n_feat, C)) * math.sqrt(6.0 / n_feat)).astype(np.float32)
    b = np.zeros(C, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # forward
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad = probs / len(yy)
            # gradients
            gW = z.T @ grad + l2 * W
            gb = grad.sum(axis=0)
            t += 1
            # Adam updates
            mW = 0.9 * mW + 0.1 * gW
            vW = 0.999 * vW + 0.001 * gW * gW
            mW_hat = mW / (1 - 0.9 ** t)
            vW_hat = vW / (1 - 0.999 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            mb = 0.9 * mb + 0.1 * gb
            vb = 0.999 * vb + 0.001 * gb * gb
            mb_hat = mb / (1 - 0.9 ** t)
            vb_hat = vb / (1 - 0.999 ** t)
            b -= lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)
    return W, b

def evaluate(Z, y, W, b):
    logits = Z @ W + b
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_total = X.shape[0]
    n_tune = n_total // 5
    train_idx = np.arange(n_total - n_tune)
    tune_idx = np.arange(n_total - n_tune, n_total)

    rng = np.random.default_rng(SEED)

    # hyperparameter search on tuning split
    best_acc = -1
    best_params = None
    best_vocab = None
    for B in B_GRID:
        for lr in LR_GRID:
            for l2 in L2_GRID:
                # build vocab on training split
                vocab = build_vocab(X[train_idx], d)
                Z_train, n_uni, n_bi = make_features(X[train_idx], vocab, B, d)
                Z_tune, _, _ = make_features(X[tune_idx], vocab, B, d)
                W, b = train_linear(Z_train, y[train_idx], C, lr, l2, rng)
                acc = evaluate(Z_tune, y[tune_idx], W, b)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (B, lr, l2)
                    best_vocab = vocab
    B, lr, l2 = best_params
    vocab = best_vocab

    # final training on all data
    vocab = build_vocab(X, d)
    Z_all, n_uni, n_bi = make_features(X, vocab, B, d)
    W, b = train_linear(Z_all, y, C, lr, l2, rng)

    # build lookup tables for inference
    # T_uni[d][256][C]
    T_uni = np.zeros((d, 256, C), dtype=np.float32)
    uni_offsets = np.cumsum([0] + [len(v) for v in vocab])
    for j in range(d):
        off = uni_offsets[j]
        for v, idx in vocab[j].items():
            T_uni[j, v] = W[off + idx]
    # T_bi[d-1][B][C]
    T_bi = np.zeros((d - 1, B, C), dtype=np.float32)
    bi_off = n_uni
    for j in range(d - 1):
        for bucket in range(B):
            T_bi[j, bucket] = W[bi_off + j * B + bucket]

    # write weights.bin
    magic = 0x4E47524D  # 'MRGN'
    H = 0
    U = n_uni
    header = struct.pack("<5i", magic, d, H, C, B, U)
    out = header
    out += T_uni.astype("<f4").tobytes()
    out += T_bi.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-ngram inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: magic 0x4E47524D, d, H=0, C, B, U;
 * float32 T_uni[d][256][C], T_bi[d-1][B][C], bias[C]. */
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
    if (nw < 24 || hd[0] != 0x4E47524D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], B = hd[4], U = hd[5];
    if (H != 0 || C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 24
        + 4L * d * 256 * C
        + 4L * (d - 1) * B * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T_uni = (const float *)(hd + 6);
    const float *T_bi = T_uni + (long)d * 256 * C;
    const float *bias = T_bi + (long)(d - 1) * B * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(scores, bias, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *t = T_uni + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) scores[c] += t[c];
        }
        for (int j = 0; j < d - 1; j++) {
            uint32_t bucket = ((xi[j] << 8) | xi[j + 1]) * 31u & (B - 1);
            const float *t = T_bi + ((long)j * B + bucket) * C;
            for (int c = 0; c < C; c++) scores[c] += t[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```