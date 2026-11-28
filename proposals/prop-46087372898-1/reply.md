```hypothesis
Family: label-pairwise — linear model on 64 pixels + 112 adjacent pairwise products (horizontal + vertical neighbors on 8×8 grid).
Why: Optdigits digits are strokes; local pixel correlations (edges, curves) are discriminative but invisible to per-pixel linear models. Pairwise terms capture this at ~2× linear's FLOPs.
Failure signal: dev accuracy ≤ 0.9613 (label-linear) or Ir/item ≥ 15000 (3× linear).
```
```python
"""label-pairwise: linear on pixels + adjacent pairwise products for 8x8 optdigits.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x50414952, d=64, C=10; float32 mean[176], std[176], W[176,10], b[10]."""
import json, math, struct, sys

import numpy as np

SEED, LR, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 64, 600000, 100
L2_GRID = [1e-4, 1e-3, 1e-2, 1e-1]  # tuned on last 20% of train.in

# 8x8 grid neighbor pairs (horizontal + vertical)
PAIRS = []
for r in range(8):
    for c in range(7):
        PAIRS.append((r*8 + c, r*8 + c + 1))  # horizontal
for r in range(7):
    for c in range(8):
        PAIRS.append((r*8 + c, (r+1)*8 + c))  # vertical
assert len(PAIRS) == 112
D_IN, D_PAIR, D_TOT, C = 64, 112, 176, 10


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def expand(X):  # X: (n, 64) uint8 -> (n, 176) float32
    n = X.shape[0]
    Xf = X.astype(np.float32)
    Z = np.empty((n, D_TOT), dtype=np.float32)
    Z[:, :D_IN] = Xf
    for idx, (i, j) in enumerate(PAIRS):
        Z[:, D_IN + idx] = Xf[:, i] * Xf[:, j]
    return Z


def train_logreg(Z, y, l2, rng, epochs, batch):
    n, d = Z.shape
    W = (rng.uniform(-1, 1, (d, C)) * math.sqrt(6.0 / d)).astype(np.float32)
    b = np.zeros(C, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z[idx], y[idx]
            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            grad_W = (z.T @ probs) / len(yy) + l2 * W
            grad_b = probs.sum(axis=0) / len(yy)
            t += 1
            beta1, beta2 = 0.9, 0.999
            mW = beta1 * mW + (1 - beta1) * grad_W
            vW = beta2 * vW + (1 - beta2) * grad_W * grad_W
            mb = beta1 * mb + (1 - beta1) * grad_b
            vb = beta2 * vb + (1 - beta2) * grad_b * grad_b
            lr_t = LR * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
            W -= lr_t * mW / (np.sqrt(vW) + 1e-8)
            b -= lr_t * mb / (np.sqrt(vb) + 1e-8)
    return W, b


def accuracy(Z, y, W, b):
    logits = Z @ W + b
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    assert card["input"]["bytes"] == D_IN and card["classes"] == C
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, D_IN)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = len(y)
    Z = expand(X)

    # Split: last 20% for tuning
    split = int(n * 0.8)
    idx_tr, idx_va = np.arange(split), np.arange(split, n)
    Ztr, ytr = Z[idx_tr], y[idx_tr]
    Zva, yva = Z[idx_va], y[idx_va]

    rng = np.random.default_rng(SEED)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(ytr)))

    # Tune L2 on validation split
    best_l2, best_acc = L2_GRID[0], -1.0
    for l2 in L2_GRID:
        W, b = train_logreg(Ztr, ytr, l2, np.random.default_rng(SEED), epochs, BATCH)
        acc = accuracy(Zva, yva, W, b)
        if acc > best_acc:
            best_acc, best_l2 = acc, l2

    # Retrain on all data with best L2
    mean = Z.mean(axis=0).astype(np.float32)
    std = Z.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    Zn = (Z - mean) / std
    W, b = train_logreg(Zn, y, best_l2, np.random.default_rng(SEED),
                        min(MAX_EPOCHS, math.ceil(SAMPLES / n)), BATCH)

    # Write weights
    magic = 0x50414952  # "PAIR"
    out = struct.pack("<3i", magic, D_IN, C)
    out += mean.astype("<f4").tobytes()
    out += std.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-pairwise inference: ./infer weights.bin in.rec N
 * Input: 64-byte records (8x8 pixels 0..16). Output: 1-byte class 0..9.
 * Weights: magic 0x50414952, d=64, C=10; float32 mean[176], std[176], W[176,10], b[10].
 * Hardcoded 112 neighbor pairs (56 horizontal, 56 vertical) for 8x8 grid. */
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

/* 112 pairs: horizontal (r, c)-(r, c+1) then vertical (r, c)-(r+1, c) for 8x8 grid */
static const uint8_t PAIRS[112][2] = {
    {0,1},{1,2},{2,3},{3,4},{4,5},{5,6},{6,7},
    {8,9},{9,10},{10,11},{11,12},{12,13},{13,14},{14,15},
    {16,17},{17,18},{18,19},{19,20},{20,21},{21,22},{22,23},
    {24,25},{25,26},{26,27},{27,28},{28,29},{29,30},{30,31},
    {32,33},{33,34},{34,35},{35,36},{36,37},{37,38},{38,39},
    {40,41},{41,42},{42,43},{43,44},{44,45},{45,46},{46,47},
    {48,49},{49,50},{50,51},{51,52},{52,53},{53,54},{54,55},
    {56,57},{57,58},{58,59},{59,60},{60,61},{61,62},{62,63},
    {0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15},
    {8,16},{9,17},{10,18},{11,19},{12,20},{13,21},{14,22},{15,23},
    {16,24},{17,25},{18,26},{19,27},{20,28},{21,29},{22,30},{23,31},
    {24,32},{25,33},{26,34},{27,35},{28,36},{29,37},{30,38},{31,39},
    {32,40},{33,41},{34,42},{35,43},{36,44},{37,45},{38,46},{39,47},
    {40,48},{41,49},{42,50},{43,51},{44,52},{45,53},{46,54},{47,55},
    {48,56},{49,57},{50,58},{51,59},{52,60},{53,61},{54,62},{55,63}
};

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 12 || hd[0] != 0x50414952 || hd[1] != 64 || hd[2] != 10) {
        fprintf(stderr, "bad weights header\n"); return 3;
    }
    const int D_IN = 64, D_PAIR = 112, D_TOT = 176, C = 10;
    long expected = 12L + 4L * (2L * D_TOT + (long)D_TOT * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 3);
    const float *std  = mean + D_TOT;
    const float *W    = std + D_TOT;
    const float *b    = W + (long)D_TOT * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float feat[D_TOT], score[C];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)D_IN) { fprintf(stderr, "item %ld: %u bytes != %d\n", i, k, D_IN); return 3; }

        /* raw pixels */
        for (int j = 0; j < D_IN; j++) feat[j] = (float)xi[j];
        /* pairwise products */
        for (int p = 0; p < D_PAIR; p++) {
            int a = PAIRS[p][0], b2 = PAIRS[p][1];
            feat[D_IN + p] = feat[a] * feat[b2];
        }
        /* normalize + linear */
        for (int c = 0; c < C; c++) score[c] = b[c];
        for (int j = 0; j < D_TOT; j++) {
            float z = (feat[j] - mean[j]) / std[j];
            const float *wj = W + (long)j * C;
            for (int c = 0; c < C; c++) score[c] += z * wj[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (score[c] > score[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```