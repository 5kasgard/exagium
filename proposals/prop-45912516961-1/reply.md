```hypothesis
We propose "label-adjquad": a linear model augmented with adjacent pairwise pixel products (horizontal and vertical) on the 8x8 grid. Spatial correlations in handwritten digits mean local products capture edge/corner information that a purely linear model misses. The feature count grows from 64 to 176, keeping inference extremely fast (≈10k FLOPs/item). If dev accuracy does not exceed ~0.965, the extra complexity is not justified over label-linear.
```
```python
"""label-adjquad: linear + adjacent pairwise products for 8x8 optdigits.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR_CANDIDATES = [3e-3, 1e-3]
L2_CANDIDATES = [1e-4, 1e-3, 1e-2]
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

# adjacency for 8x8: horizontal (row, col) -> (row, col+1), vertical -> (row+1, col)
ADJ = []
for r in range(8):
    for c in range(7):
        ADJ.append((r*8 + c, r*8 + c + 1))
for r in range(7):
    for c in range(8):
        ADJ.append((r*8 + c, (r+1)*8 + c))
NUM_ADJ = len(ADJ)  # 112

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def expand(X, mean, std):
    """X: (n,64) uint8 -> (n, 176) float32: normalized linear + pairwise products."""
    n = X.shape[0]
    Z = (X.astype(np.float32) - mean) / std          # (n,64)
    quad = np.empty((n, NUM_ADJ), dtype=np.float32)
    for idx, (i, j) in enumerate(ADJ):
        quad[:, idx] = Z[:, i] * Z[:, j]
    return np.concatenate([Z, quad], axis=1)

def softmax_grad(logits, y):
    """logits: (n,C), y: (n,) -> gradient (n,C)"""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def train_once(X, y, C, lr, l2, epochs, rng):
    n, d = X.shape
    W = (rng.uniform(-1, 1, (d, C)) * math.sqrt(6.0 / d)).astype(np.float32)
    b = np.zeros(C, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            logits = xb @ W + b
            g = softmax_grad(logits, yb).astype(np.float32)
            gradW = xb.T @ g + l2 * W
            gradb = g.sum(axis=0)
            t += 1
            mW = 0.9 * mW + 0.1 * gradW
            vW = 0.999 * vW + 0.001 * gradW * gradW
            mb = 0.9 * mb + 0.1 * gradb
            vb = 0.999 * vb + 0.001 * gradb * gradb
            W -= lr * (mW / (1 - 0.9**t)) / (np.sqrt(vW / (1 - 0.999**t)) + 1e-8)
            b -= lr * (mb / (1 - 0.9**t)) / (np.sqrt(vb / (1 - 0.999**t)) + 1e-8)
    return W, b

def accuracy(X, y, W, b):
    logits = X @ W + b
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]
    assert X.shape[1] == 64

    # split last 20% for tuning
    n_val = n // 5
    X_train, y_train = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]

    # normalization stats on training split
    mean = X_train.mean(axis=0).astype(np.float32)
    std = X_train.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0

    X_train_exp = expand(X_train, mean, std)
    X_val_exp = expand(X_val, mean, std)

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_W, best_b = None, None
    best_lr, best_l2 = LR_CANDIDATES[0], L2_CANDIDATES[0]

    for lr in LR_CANDIDATES:
        for l2 in L2_CANDIDATES:
            epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
            W, b = train_once(X_train_exp, y_train, C, lr, l2, epochs, rng)
            acc = accuracy(X_val_exp, y_val, W, b)
            if acc > best_acc:
                best_acc = acc
                best_W, best_b = W.copy(), b.copy()
                best_lr, best_l2 = lr, l2

    # retrain on full data with best hyperparameters
    mean_full = X.mean(axis=0).astype(np.float32)
    std_full = X.std(axis=0).astype(np.float32)
    std_full[std_full == 0] = 1.0
    X_full_exp = expand(X, mean_full, std_full)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    rng = np.random.default_rng(SEED)
    W, b = train_once(X_full_exp, y, C, best_lr, best_l2, epochs, rng)

    # pack weights: magic, d, C, num_adj, mean[64], std[64], W[176,10] (col-major? we store row-major per class), b[10]
    # We'll store linear weights as [C, 64], quad weights as [C, 112], bias [C]
    linear_W = best_W[:64].T   # (10,64)
    quad_W = best_W[64:].T     # (10,112)
    magic = 0x41444A51  # 'ADJQ'
    header = struct.pack("<4I", magic, 64, C, NUM_ADJ)
    body = (mean_full.tobytes() + std_full.tobytes() +
            linear_W.astype("<f4").tobytes() +
            quad_W.astype("<f4").tobytes() +
            b.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```
```c
/* label-adjquad inference: ./infer weights.bin in.rec N
 * Input: 64 bytes per record. Output: 1 byte class per record. */
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

/* adjacency for 8x8: horizontal then vertical */
static const int adj_i[112] = {
    0,1, 1,2, 2,3, 3,4, 4,5, 5,6, 6,7,
    8,9, 9,10, 10,11, 11,12, 12,13, 13,14, 14,15,
    16,17, 17,18, 18,19, 19,20, 20,21, 21,22, 22,23,
    24,25, 25,26, 26,27, 27,28, 28,29, 29,30, 30,31,
    32,33, 33,34, 34,35, 35,36, 36,37, 37,38, 38,39,
    40,41, 41,42, 42,43, 43,44, 44,45, 45,46, 46,47,
    48,49, 49,50, 50,51, 51,52, 52,53, 53,54, 54,55,
    56,57, 57,58, 58,59, 59,60, 60,61, 61,62, 62,63,
    0,8, 1,9, 2,10, 3,11, 4,12, 5,13, 6,14, 7,15,
    8,16, 9,17, 10,18, 11,19, 12,20, 13,21, 14,22, 15,23,
    16,24, 17,25, 18,26, 19,27, 20,28, 21,29, 22,30, 23,31,
    24,32, 25,33, 26,34, 27,35, 28,36, 29,37, 30,38, 31,39,
    32,40, 33,41, 34,42, 35,43, 36,44, 37,45, 38,46, 39,47,
    40,48, 41,49, 42,50, 43,51, 44,52, 45,53, 46,54, 47,55,
    48,56, 49,57, 50,58, 51,59, 52,60, 53,61, 54,62, 55,63
};
static const int adj_j[112] = {
    1,0, 2,1, 3,2, 4,3, 5,4, 6,5, 7,6,
    9,8, 10,9, 11,10, 12,11, 13,12, 14,13, 15,14,
    17,16, 18,17, 19,18, 20,19, 21,20, 22,21, 23,22,
    25,24, 26,25, 27,26, 28,27, 29,28, 30,29, 31,30,
    33,32, 34,33, 35,34, 36,35, 37,36, 38,37, 39,38,
    41,40, 42,41, 43,42, 44,43, 45,44, 46,45, 47,46,
    49,48, 50,49, 51,50, 52,51, 53,52, 54,53, 55,54,
    57,56, 58,57, 59,58, 60,59, 61,60, 62,61, 63,62,
    8,0, 9,1, 10,2, 11,3, 12,4, 13,5, 14,6, 15,7,
    16,8, 17,9, 18,10, 19,11, 20,12, 21,13, 22,14, 23,15,
    24,16, 25,17, 26,18, 27,19, 28,20, 29,21, 30,22, 31,23,
    32,24, 33,25, 34,26, 35,27, 36,28, 37,29, 38,30, 39,31,
    40,32, 41,33, 42,34, 43,35, 44,36, 45,37, 46,38, 47,39,
    48,40, 49,41, 50,42, 51,43, 52,44, 53,45, 54,46, 55,47,
    56,48, 57,49, 58,50, 59,51, 60,52, 61,53, 62,54, 63,55
};

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t *hdr = (uint32_t *)wb;
    if (hdr[0] != 0x41444A51) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], C = hdr[2], num_adj = hdr[3];
    if (d != 64 || C != 10 || num_adj != 112) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 4L*(64*2 + 10*64 + 10*112 + 10);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *ptr = (const float *)(hdr + 4);
    const float *mean = ptr; ptr += 64;
    const float *std  = ptr; ptr += 64;
    const float *linW = ptr; ptr += 10*64;
    const float *quadW = ptr; ptr += 10*112;
    const float *bias = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float norm[64];
    float quad[112];
    float scores[10];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        for (int j = 0; j < 64; j++) norm[j] = (xi[j] - mean[j]) / std[j];
        for (int p = 0; p < 112; p++) quad[p] = norm[adj_i[p]] * norm[adj_j[p]];
        for (int c = 0; c < 10; c++) {
            float s = bias[c];
            const float *lw = linW + c*64;
            const float *qw = quadW + c*112;
            for (int j = 0; j < 64; j++) s += lw[j] * norm[j];
            for (int p = 0; p < 112; p++) s += qw[p] * quad[p];
            scores[c] = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < 10; c++) if (scores[c] > scores[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```