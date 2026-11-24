```hypothesis
Family: label-embmlp (embedding + small MLP). Learns a per-position byte embedding (dim D) and a one-hidden-layer MLP (size H) on the concatenated embeddings. This captures positional interactions more efficiently than a linear model on one-hot vectors (label-linear) while keeping inference fast via precomputed tables (same pattern as label-linear). If dev accuracy does not exceed label-linear's 0.3375 at similar or lower Ir/item, the family is not justified.
```
```python
"""label-embmlp: per-position byte embeddings + one ReLU layer (or linear if H=0).
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports precomputed tables for fast inference: magic 0x454D424D, d, D, H, C;
float32 T[d][256][W], b1[W]; if H>0: W2[H][C], b2[C] (W = H if H>0 else C)."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_eval(X, y, val_idx, D, H, C, rng):
    n, d = X.shape
    W = H if H else C
    # embeddings: list of (256, D)
    E = [(rng.uniform(-1, 1, (256, D)) * math.sqrt(6.0 / D)).astype(np.float32) for _ in range(d)]
    # first layer weights: (d*D, W)
    W1 = (rng.uniform(-1, 1, (d * D, W)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
    b1 = np.zeros(W, np.float32)
    if H:
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [E, W1, b1, W2, b2]
    else:
        params = [E, W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params[1:]]  # skip E (updated separately)
    t = 0
    tr_idx = np.setdiff1d(np.arange(n), val_idx)
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / len(tr_idx)))):
        order = rng.permutation(len(tr_idx))
        for s in range(0, len(tr_idx), BATCH):
            idx = tr_idx[order[s:s + BATCH]]
            xb, yb = X[idx], y[idx]
            # forward
            emb = np.concatenate([E[j][xb[:, j]] for j in range(d)], axis=1)  # (b, d*D)
            h = emb @ W1 + b1
            if H:
                h = np.maximum(h, 0)
                logits = h @ W2 + b2
            else:
                logits = h
            # softmax gradient
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            dlogits = (probs / len(yb)).astype(np.float32)
            # backward
            if H:
                dW2 = h.T @ dlogits + L2 * W2
                db2 = dlogits.sum(0)
                dh = dlogits @ W2.T
                dh *= (h > 0)
            else:
                dh = dlogits
            dW1 = emb.T @ dh + L2 * W1
            db1 = dh.sum(0)
            demb = dh @ W1.T  # (b, d*D)
            # embedding gradients
            dE = [np.zeros_like(e) for e in E]
            for j in range(d):
                dj = demb[:, j*D:(j+1)*D]
                for i, v in enumerate(xb[:, j]):
                    dE[j][v] += dj[i]
            # Adam updates
            grads = [dW1, db1]
            if H:
                grads += [dW2, db2]
            t += 1
            for i, (p, gr) in enumerate(zip(params[1:], grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            # embeddings: simple SGD with same LR (could use Adam but keep simple)
            for j in range(d):
                E[j] -= LR * dE[j]
    # validation accuracy
    emb_val = np.concatenate([E[j][X[val_idx, j]] for j in range(d)], axis=1)
    h_val = emb_val @ W1 + b1
    if H:
        h_val = np.maximum(h_val, 0)
        logits_val = h_val @ W2 + b2
    else:
        logits_val = h_val
    pred = logits_val.argmax(axis=1)
    acc = (pred == y[val_idx]).mean()
    return acc, (E, W1, b1, W2, b2) if H else (E, W1, b1)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)
    # tuning split: last 20%
    val_size = n // 5
    val_idx = np.arange(n - val_size, n)
    # hyperparameter search
    best_acc, best_cfg, best_params = -1, None, None
    for D in (8, 16, 32):
        for H in (0, 32, 64, 128):
            acc, params = train_eval(X, y, val_idx, D, H, C, np.random.default_rng(SEED))
            if acc > best_acc:
                best_acc, best_cfg, best_params = acc, (D, H), params
    D, H = best_cfg
    W = H if H else C
    # retrain on all data
    rng = np.random.default_rng(SEED)
    E = [(rng.uniform(-1, 1, (256, D)) * math.sqrt(6.0 / D)).astype(np.float32) for _ in range(d)]
    W1 = (rng.uniform(-1, 1, (d * D, W)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
    b1 = np.zeros(W, np.float32)
    if H:
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [E, W1, b1, W2, b2]
    else:
        params = [E, W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params[1:]]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            emb = np.concatenate([E[j][xb[:, j]] for j in range(d)], axis=1)
            h = emb @ W1 + b1
            if H:
                h = np.maximum(h, 0)
                logits = h @ W2 + b2
            else:
                logits = h
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            dlogits = (probs / len(yb)).astype(np.float32)
            if H:
                dW2 = h.T @ dlogits + L2 * W2
                db2 = dlogits.sum(0)
                dh = dlogits @ W2.T
                dh *= (h > 0)
            else:
                dh = dlogits
            dW1 = emb.T @ dh + L2 * W1
            db1 = dh.sum(0)
            demb = dh @ W1.T
            dE = [np.zeros_like(e) for e in E]
            for j in range(d):
                dj = demb[:, j*D:(j+1)*D]
                for i, v in enumerate(xb[:, j]):
                    dE[j][v] += dj[i]
            grads = [dW1, db1]
            if H:
                grads += [dW2, db2]
            t += 1
            for i, (p, gr) in enumerate(zip(params[1:], grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            for j in range(d):
                E[j] -= LR * dE[j]
    # precompute T[j][v] = E[j][v] @ W1_j  (W1 split into d blocks of D x W)
    T = np.zeros((d, 256, W), dtype=np.float32)
    for j in range(d):
        W1_j = W1[j*D:(j+1)*D]
        T[j] = E[j] @ W1_j
    # write weights.bin
    magic = 0x454D424D  # 'EMBM'
    out = struct.pack("<5i", magic, d, D, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: magic 0x454D424D, int32 d, D, H, C; float32 T[d][256][W], b1[W];
 * if H>0: float32 W2[H][C], b2[C] (W = H if H>0 else C). */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    int W = H ? H : C;
    long expected = 20L + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0));
    if (C < 1 || C > 256 || nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hd + 5);
    const float *b1 = T + (long)d * 256 * W;
    const float *W2 = b1 + W;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W);
    float *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(acc, b1, sizeof(float) * W);
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```