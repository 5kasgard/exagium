```hypothesis
Family: low-rank factorized linear (positional embeddings + shared output).
Why: The linear model learns a separate 256x65 weight matrix per position (16*256*65 params). 
By factorizing each position's matrix as 256xR * Rx65 with shared Rx65, we drastically reduce parameters, 
encourage generalization across bytes and positions, and keep inference fast (O(16*R + R*C) vs O(16*C)).
Failure: If dev accuracy <= label-linear (0.3375) or inference cost exceeds label-linear.
```

```python
"""Low-rank factorized linear model for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Architecture: for each of 16 positions, a 256xR embedding matrix; shared output layer Rx65 + bias.
Training: Adam on full softmax cross-entropy. Tuning: last 20% of train.in for rank R selection.
Weights format: magic 0x4C4F5752, int32 d,R,C; float32 E[d][256][R], W[R][C], b[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 0.01
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
BATCH = 64
TUNE_EPOCHS = 30
FINAL_EPOCHS = 100
CANDIDATE_R = [16, 32, 64, 128]  # ranks to try on tuning split
MAGIC = 0x4C4F5752  # "LOWR"


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_cross_entropy_grad(logits, y):
    """Return gradient of loss w.r.t logits, shape (batch, C)."""
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)


def evaluate(X, y, E, W, b):
    """Return accuracy on given data."""
    n = X.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        Xb = X[s:s+BATCH]
        yb = y[s:s+BATCH]
        # sum embeddings
        emb_sum = np.zeros((len(Xb), E.shape[2]), dtype=np.float32)
        for j in range(E.shape[0]):
            emb_sum += E[j, Xb[:, j]]
        logits = emb_sum @ W + b
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def train_split(X_tr, y_tr, X_val, y_val, R, epochs, rng):
    """Train model with given rank R, return (E, W, b, val_acc)."""
    d = X_tr.shape[1]
    C = int(y_tr.max()) + 1
    # init
    scale = 1.0 / math.sqrt(R)
    E = rng.normal(0, scale, (d, 256, R)).astype(np.float32)
    W = rng.normal(0, scale, (R, C)).astype(np.float32)
    b = np.zeros(C, np.float32)
    # Adam state
    mE, vE = np.zeros_like(E), np.zeros_like(E)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    n = X_tr.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            # forward
            emb_sum = np.zeros((len(Xb), R), np.float32)
            for j in range(d):
                emb_sum += E[j, Xb[:, j]]
            logits = emb_sum @ W + b
            # backward
            g_logits = softmax_cross_entropy_grad(logits, yb).astype(np.float32)
            # grad W, b
            gW = emb_sum.T @ g_logits
            gb = g_logits.sum(axis=0)
            # grad E
            gE = np.zeros_like(E)
            for j in range(d):
                # gE[j, byte, :] += g_logits[i] for each example i with that byte
                # vectorized: for each byte value, sum g_logits where Xb[:,j]==byte
                bytes_j = Xb[:, j]
                for byte_val in range(256):
                    mask = (bytes_j == byte_val)
                    if mask.any():
                        gE[j, byte_val] += g_logits[mask].sum(axis=0)
            # Adam updates
            t += 1
            # E
            mE = BETA1 * mE + (1 - BETA1) * gE
            vE = BETA2 * vE + (1 - BETA2) * gE * gE
            mE_hat = mE / (1 - BETA1 ** t)
            vE_hat = vE / (1 - BETA2 ** t)
            E -= LR * mE_hat / (np.sqrt(vE_hat) + EPS)
            # W
            mW = BETA1 * mW + (1 - BETA1) * gW
            vW = BETA2 * vW + (1 - BETA2) * gW * gW
            mW_hat = mW / (1 - BETA1 ** t)
            vW_hat = vW / (1 - BETA2 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + EPS)
            # b
            mb = BETA1 * mb + (1 - BETA1) * gb
            vb = BETA2 * vb + (1 - BETA2) * gb * gb
            mb_hat = mb / (1 - BETA1 ** t)
            vb_hat = vb / (1 - BETA2 ** t)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + EPS)
    val_acc = evaluate(X_val, y_val, E, W, b)
    return E, W, b, val_acc


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # load all training data
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # split: last 20% for tuning
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    rng = np.random.default_rng(SEED)
    # tune rank
    best_R, best_acc = None, -1.0
    for R in CANDIDATE_R:
        _, _, _, val_acc = train_split(X_tr, y_tr, X_val, y_val, R, TUNE_EPOCHS, rng)
        if val_acc > best_acc:
            best_acc, best_R = val_acc, R
    # retrain on full data with best_R
    E, W, b, _ = train_split(X, y, X[:1], y[:1], best_R, FINAL_EPOCHS, rng)  # dummy val
    # write weights
    out = struct.pack("<4i", MAGIC, d, best_R, C)
    out += E.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Low-rank factorized linear inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x4C4F5752, int32 d,R,C; float32 E[d][256][R], W[R][C], b[C]. */
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
    if (nw < 16 || hd[0] != 0x4C4F5752) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], R = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * R + (long)R * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)d * 256 * R;
    const float *b = W + (long)R * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * R);
    float *sc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int r = 0; r < R; r++) acc[r] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = E + ((long)j * 256 + xi[j]) * R;
            for (int r = 0; r < R; r++) acc[r] += e[r];
        }
        for (int c = 0; c < C; c++) sc[c] = b[c];
        for (int r = 0; r < R; r++) {
            float ar = acc[r];
            const float *wr = W + (long)r * C;
            for (int c = 0; c < C; c++) sc[c] += ar * wr[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```