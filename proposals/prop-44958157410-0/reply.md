```hypothesis
Family: label-embmlp (embedding MLP). Each of the 14 byte features gets a learned embedding vector (size E), concatenated and fed to a single ReLU hidden layer (size H) then a 2-class output. Embeddings capture non-linear categorical relationships better than one-hot linear; the small MLP adds interaction capacity. Tuning E,H on the last 20% of train should beat label-linear's 0.8524 at similar latency. If dev accuracy ≤ 0.8524 or instructions/item > 1000, the family does not justify its complexity.
```
```python
"""label-embmlp: embedding + one ReLU hidden layer for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Grid-searches E,H on last 20% of train (tuning split), then retrains on all train.
Exports: magic 0x454D424D, d, E, H, C; float32 embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH = 3e-3, 1e-4, 64
GRID_EPOCHS, FINAL_EPOCHS = 10, 60
E_CHOICES = [8, 16]
H_CHOICES = [16, 32]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, Xv, yv, d, E, H, C, epochs, rng):
    # embeddings: list of d arrays (256, E)
    emb = [(rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32) for _ in range(d)]
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = emb + [W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            # forward
            Z = np.zeros((len(yb), d * E), np.float32)
            for j in range(d):
                Z[:, j*E:(j+1)*E] = emb[j][xb[:, j]]
            A = Z @ W1 + b1
            H_act = np.maximum(A, 0)
            logits = H_act @ W2 + b2
            # loss gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            dlogits = (probs / len(yb)).astype(np.float32)
            # backward
            dW2 = H_act.T @ dlogits + L2 * W2
            db2 = dlogits.sum(0)
            dH = dlogits @ W2.T
            dA = dH * (A > 0)
            dW1 = Z.T @ dA + L2 * W1
            db1 = dA.sum(0)
            dZ = dA @ W1.T
            # embedding grads
            demb = [np.zeros((256, E), np.float32) for _ in range(d)]
            for j in range(d):
                np.add.at(demb[j], xb[:, j], dZ[:, j*E:(j+1)*E])
            grads = demb + [dW1, db1, dW2, db2]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    # validation accuracy
    if Xv is not None:
        Zv = np.zeros((Xv.shape[0], d * E), np.float32)
        for j in range(d):
            Zv[:, j*E:(j+1)*E] = emb[j][Xv[:, j]]
        Av = Zv @ W1 + b1
        Hv = np.maximum(Av, 0)
        logitsv = Hv @ W2 + b2
        pred = logitsv.argmax(axis=1)
        acc = (pred == yv).mean()
    else:
        acc = 0.0
    return params, acc

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    val_size = n // 5
    train_size = n - val_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    rng = np.random.default_rng(SEED)
    best_acc, best_params, best_E, best_H = -1, None, 8, 16
    for E in E_CHOICES:
        for H in H_CHOICES:
            params, acc = train_model(X_train, y_train, X_val, y_val, d, E, H, C, GRID_EPOCHS, rng)
            if acc > best_acc:
                best_acc, best_params, best_E, best_H = acc, params, E, H
    # final training on all data
    final_params, _ = train_model(X, y, None, None, d, best_E, best_H, C, FINAL_EPOCHS, rng)
    emb, W1, b1, W2, b2 = final_params[:-4], final_params[-4], final_params[-3], final_params[-2], final_params[-1]
    # write weights
    magic = 0x454D424D
    out = struct.pack("<5I", magic, d, best_E, best_H, C)
    for j in range(d):
        out += emb[j].astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class index (0 or 1). */
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C != 2 || d != 14 || E < 1 || H < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20 + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // concatenate embeddings
        for (int w = 0; w < d * E; w++) concat[w] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int ee = 0; ee < E; ee++) concat[j * E + ee] = e[ee];
        }
        // hidden layer
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int w = 0; w < d * E; w++) sum += concat[w] * W1[(long)w * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        // output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```