```hypothesis
We propose a Factorization Machine family "label-fm" that adds pairwise feature interactions via low-rank embeddings to a linear model. The Adult dataset has known feature interactions (e.g., education × occupation). FMs capture these efficiently with O(d·k) inference cost, where k is the embedding dimension (tuned on the 20% tuning split). At k=8 we expect dev accuracy >0.855 with Ir/item < 2000, beating label-linear on accuracy at moderate cost. If accuracy ≤0.8524 or Ir/item > 3000, the model fails its goal.
```

```python
"""Factorization Machine for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each of the 256 values per feature with a linear weight and a k-dim embedding.
Trains with Adam (lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n))).
Chooses embedding dimension k on the last 20% of train.in (tuning split), then retrains on all train.
Weights format: magic 0x464D4C31, d, k, C; float32 W_lin[C*d*256], V[C*d*256*k], b[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
K_CHOICES = [2, 4, 8, 16, 32]  # embedding dimensions to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def fm_forward(X_idx, W_lin, V, b, C, d, k):
    """X_idx: (n, d) uint8. Returns scores (n, C)."""
    n = X_idx.shape[0]
    scores = np.zeros((n, C), dtype=np.float32)
    for c in range(C):
        # linear term
        lin = b[c] + np.sum(W_lin[c][X_idx], axis=1, dtype=np.float32)
        # FM term
        sum_v = np.zeros((n, k), dtype=np.float32)
        sum_sq = np.zeros((n, k), dtype=np.float32)
        for j in range(d):
            v = V[c, j][X_idx[:, j]]  # (n, k)
            sum_v += v
            sum_sq += v * v
        inter = 0.5 * np.sum(sum_v * sum_v - sum_sq, axis=1)
        scores[:, c] = lin + inter
    return scores


def fm_backward(X_idx, y, scores, W_lin, V, b, C, d, k):
    """Compute gradients for W_lin, V, b. Returns (gW_lin, gV, gb)."""
    n = X_idx.shape[0]
    # softmax gradient
    P = np.exp(scores - scores.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    P[np.arange(n), y] -= 1.0
    P /= n  # (n, C)

    gW_lin = np.zeros_like(W_lin)
    gV = np.zeros_like(V)
    gb = np.zeros_like(b)

    for c in range(C):
        gb[c] = P[:, c].sum()
        # linear gradients
        for j in range(d):
            idx = X_idx[:, j]
            np.add.at(gW_lin[c, j], idx, P[:, c])
        # FM gradients
        # compute sum_v per sample for class c
        sum_v = np.zeros((n, k), dtype=np.float32)
        for j in range(d):
            sum_v += V[c, j][X_idx[:, j]]
        # gradient for each feature
        for j in range(d):
            idx = X_idx[:, j]
            emb = V[c, j][idx]  # (n, k)
            # d(inter)/dV = sum_v - emb
            grad_emb = (sum_v - emb) * P[:, c, None]  # (n, k)
            np.add.at(gV[c, j], idx, grad_emb)
    return gW_lin, gV, gb


def train_fm(X, y, C, d, k, epochs, rng):
    n = X.shape[0]
    # parameters
    W_lin = rng.uniform(-0.01, 0.01, (C, d, 256)).astype(np.float32)
    V = rng.uniform(-0.01, 0.01, (C, d, 256, k)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [W_lin, V, b]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            scores = fm_forward(Xb, W_lin, V, b, C, d, k)
            gW_lin, gV, gb = fm_backward(Xb, yb, scores, W_lin, V, b, C, d, k)
            grads = [gW_lin, gV, gb]
            # L2 regularization
            grads[0] += L2 * W_lin
            grads[1] += L2 * V
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return W_lin, V, b


def accuracy(X, y, W_lin, V, b, C, d, k):
    scores = fm_forward(X, W_lin, V, b, C, d, k)
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)

    n = X.shape[0]
    tune_size = int(0.2 * n)
    train_idx = slice(0, n - tune_size)
    tune_idx = slice(n - tune_size, n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))

    best_k, best_acc = None, -1.0
    for k in K_CHOICES:
        W_lin, V, b = train_fm(X_train, y_train, C, d, k, epochs, rng)
        acc = accuracy(X_tune, y_tune, W_lin, V, b, C, d, k)
        if acc > best_acc:
            best_acc, best_k = acc, k

    # retrain on full data with best_k
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    W_lin, V, b = train_fm(X, y, C, d, best_k, epochs_full, rng)

    # write weights.bin
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, best_k, C)
    out += W_lin.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weights format: int32 magic 0x464D4C31, d, k, C; float32 W_lin[C*d*256], V[C*d*256*k], b[C]. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C < 1 || C > 256 || k < 1 || k > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)C * d * 256 + (long)C * d * 256 * k + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *W_lin = (const float *)(hd + 4);
    const float *V = W_lin + (long)C * d * 256;
    const float *b = V + (long)C * d * 256 * k;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * k);
    float *sum_sq = malloc(sizeof(float) * k);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        for (int c = 0; c < C; c++) {
            scores[c] = b[c];
            // linear term
            for (int j = 0; j < d; j++) {
                scores[c] += W_lin[((long)c * d + j) * 256 + xi[j]];
            }
            // FM term
            memset(sum_v, 0, sizeof(float) * k);
            memset(sum_sq, 0, sizeof(float) * k);
            for (int j = 0; j < d; j++) {
                const float *v = V + (((long)c * d + j) * 256 + xi[j]) * k;
                for (int f = 0; f < k; f++) {
                    float val = v[f];
                    sum_v[f] += val;
                    sum_sq[f] += val * val;
                }
            }
            float inter = 0.0f;
            for (int f = 0; f < k; f++) inter += sum_v[f] * sum_v[f] - sum_sq[f];
            scores[c] += 0.5f * inter;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(sum_v); free(sum_sq); free(scores);
    return 0;
}
```