```hypothesis
We propose an embedding-based MLP (label-mlpemb) that learns a dense embedding for each (position, byte) pair, sums them, and passes through a single ReLU hidden layer. This captures non-linear interactions between context positions while keeping inference cost low (only a few thousand FLOPs). It should beat the linear model's 33.75% accuracy on this character prediction task. If it fails to exceed 34% accuracy or its instruction cost exceeds 15k Ir/item, the approach is not worthwhile.
```
```python
"""Embedding MLP for next-byte prediction: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy. Learns per-position byte embeddings, sums them,
then one ReLU hidden layer, then linear output. Hyperparameters chosen on last 20% of train.in,
then retrained on all data. Exports: magic 0x50424D45, d, E, H, C; embeddings[d][256][E], W1[E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, EPOCHS_SEARCH, EPOCHS_FULL = 3e-3, 1e-4, 64, 20, 60
GRID = [(16, 32), (32, 64), (64, 128)]  # (E, H) pairs to try


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(params, X, y, d, C):
    emb, W1, b1, W2, b2 = params
    n = X.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        idx = slice(s, min(s + BATCH, n))
        xb, yb = X[idx], y[idx]
        # sum embeddings
        emb_sum = np.zeros((len(xb), W1.shape[0]), np.float32)
        for j in range(d):
            emb_sum += emb[j, xb[:, j]]
        h = np.maximum(emb_sum @ W1 + b1, 0)
        logits = h @ W2 + b2
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def train_model(X, y, d, C, E, H, epochs, rng):
    n, k = X.shape[0], d
    # init
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 * E))).astype(np.float32)
    W1 = (rng.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / E)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(xb)
            # forward
            emb_sum = np.zeros((bs, E), np.float32)
            for j in range(d):
                emb_sum += emb[j, xb[:, j]]
            h_pre = emb_sum @ W1 + b1
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2
            # softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs
            # backward
            dW2 = h.T @ probs + L2 * W2
            db2 = probs.sum(0)
            dh = probs @ W2.T
            dh[h_pre <= 0] = 0.0
            dW1 = emb_sum.T @ dh + L2 * W1
            db1 = dh.sum(0)
            demb_sum = dh @ W1.T  # (bs, E)
            demb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(demb[j], xb[:, j], demb_sum)
            grads = [demb, dW1, db1, dW2, db2]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = card["input"]["bytes"], card["classes"]
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc, best_params, best_cfg = -1, None, None
    for E, H in GRID:
        params = train_model(X_tr, y_tr, d, C, E, H, EPOCHS_SEARCH, rng)
        acc = accuracy(params, X_val, y_val, d, C)
        if acc > best_acc:
            best_acc, best_params, best_cfg = acc, params, (E, H)
    # retrain on full data
    E, H = best_cfg
    rng = np.random.default_rng(SEED)  # reset seed for reproducibility
    final_params = train_model(X, y, d, C, E, H, EPOCHS_FULL, rng)
    emb, W1, b1, W2, b2 = final_params

    # write weights
    magic = 0x50424D45  # 'EMBP'
    header = struct.pack("<5i", magic, d, E, H, C)
    out = bytearray(header)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Embedding MLP inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: magic 0x50424D45, int32 d, E, H, C;
 * float32 emb[d][256][E], W1[E][H], b1[H], W2[H][C], b2[C] (all little-endian). */
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
    if (nw < 20 || hd[0] != 0x50424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // sum embeddings
        for (int e = 0; e < E; e++) emb_sum[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) emb_sum[e] += eptr[e];
        }
        // hidden layer with ReLU
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int e = 0; e < E; e++) sum += emb_sum[e] * W1[(long)e * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        // output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }
        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(emb_sum); free(hidden); free(logits);
    return 0;
}
```