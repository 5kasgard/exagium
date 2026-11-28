```hypothesis
We propose a factorized embedding MLP (label-embed-mlp) that learns a low-dimensional embedding per context position, then a hidden layer, then output. This reduces parameters vs. full one-hot linear layer, improving generalization on small data. We expect it to beat label-linear (0.3375) and approach label-gbdt (0.3923). If it doesn't beat label-linear, the factorization is too restrictive or hyperparameters poorly chosen.
```

```python
"""label-embed-mlp: factorized embedding + one hidden ReLU layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 123), single thread, numpy only.
Chooses embedding dim E and hidden dim H on last 20% of train.in, then trains on all.
Exports precomputed tables for fast inference: magic 0x454D424C, d, E, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 123
LR, L2, BATCH = 3e-3, 1e-4, 64
GRID_EPOCHS, FINAL_EPOCHS = 15, 30
E_CHOICES = [4, 8, 16]
H_CHOICES = [32, 64, 128]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(params, X, y):
    emb, W1, b1, W2, b2 = params
    d, E, H, C = emb.shape[0], emb.shape[1], W1.shape[1], W2.shape[1]
    n = X.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        xb = X[s:s + BATCH]
        yb = y[s:s + BATCH]
        idx = xb + (np.arange(d) * 256)  # (B, d)
        emb_batch = emb.reshape(d * 256, E)[idx]  # (B, d, E)
        flat = emb_batch.reshape(len(xb), -1)  # (B, d*E)
        h = np.maximum(flat @ W1 + b1, 0)
        logits = h @ W2 + b2
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def train_model(X, y, E, H, epochs, rng):
    n, d = X.shape
    C = 65
    # params
    emb = rng.normal(0, 0.01, (d, 256, E)).astype(np.float32)
    W1 = rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = rng.normal(0, math.sqrt(2.0 / H), (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            # forward
            flat_idx = xb + (np.arange(d) * 256)  # (B, d)
            emb_batch = emb.reshape(d * 256, E)[flat_idx]  # (B, d, E)
            flat = emb_batch.reshape(len(xb), -1)  # (B, d*E)
            pre = flat @ W1 + b1
            h = np.maximum(pre, 0)
            logits = h @ W2 + b2
            # softmax + cross-entropy grad
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            probs /= len(yb)
            # backward
            gW2 = h.T @ probs + L2 * W2
            gb2 = probs.sum(axis=0)
            gh = probs @ W2.T
            gh[pre <= 0] = 0
            gW1 = flat.T @ gh + L2 * W1
            gb1 = gh.sum(axis=0)
            # emb grad
            gflat = gh @ W1.T  # (B, d*E)
            gemb = gflat.reshape(len(xb), d, E)  # (B, d, E)
            # scatter add into emb
            geom = np.zeros_like(emb)
            for j in range(d):
                np.add.at(geom[j], xb[:, j], gemb[:, j, :])
            geom += L2 * emb
            grads = [geom, gW1, gb1, gW2, gb2]
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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    split = int(0.8 * n)
    rng = np.random.default_rng(SEED)

    # grid search on tuning split
    best_acc, best_params, best_E, best_H = -1, None, None, None
    for E in E_CHOICES:
        for H in H_CHOICES:
            params = train_model(X[:split], y[:split], E, H, GRID_EPOCHS, rng)
            acc = accuracy(params, X[split:], y[split:])
            if acc > best_acc:
                best_acc, best_params, best_E, best_H = acc, params, E, H

    # final training on all data
    final_params = train_model(X, y, best_E, best_H, FINAL_EPOCHS, rng)
    emb, W1, b1, W2, b2 = final_params

    # precompute T[j][byte][h] = emb[j,byte] @ W1[j*E:(j+1)*E, h]
    T = np.zeros((d, 256, best_H), dtype=np.float32)
    for j in range(d):
        T[j] = emb[j] @ W1[j * best_E:(j + 1) * best_E]

    # write weights.bin
    magic = 0x454D424C  # 'EMBL' little-endian
    hdr = struct.pack("<5i", magic, d, best_E, best_H, C)
    out = hdr
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x454D424C, int32 d, E, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(acc, b1, H * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += t[h];
        }
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;
        memcpy(scores, b2, C * sizeof(float));
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            if (ah == 0.0f) continue;
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += ah * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```