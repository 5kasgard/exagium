```hypothesis
We propose a "hashed bigram linear" family: a linear model on unigram (per-position) and bigram (adjacent pair) features, where bigrams are hashed into a shared fixed-size table. This captures local character interactions crucial for language modeling while keeping inference fast (O(d + d*H_bigram) ops). The hash size H_bigram is tuned on the last 20% of train.in; if the best tuned H_bigram yields no improvement over label-linear, it shows bigram collisions or insufficient signal.
```
```python
"""Hashed bigram linear model for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Adds shared bigram features (hashed) to the per-position unigram features of label-linear.
Tunes hash size H on the last 20% of train.in, then trains on all data.
Exported format: int32 magic 0x48424C31, d, H, C; float32 U[d][256][C], B[H][C], bias[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
H_CANDIDATES = [128, 256, 512, 1024, 2048]  # hash sizes to try

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_eval(card, X_tr, y_tr, X_val, y_val, H):
    """Train on X_tr,y_tr, evaluate on X_val,y_val, return accuracy and params."""
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)
    # init
    U = (rng.uniform(-1, 1, (d, 256, C)) * math.sqrt(6.0 / (256 * d))).astype(np.float32)
    B = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    bias = np.zeros(C, np.float32)
    params = [U, B, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    n_tr = X_tr.shape[0]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            bsz = xb.shape[0]
            # forward
            logits = bias[None, :].repeat(bsz, axis=0)
            for j in range(d):
                logits += U[j, xb[:, j], :]
            for j in range(d - 1):
                hidx = (xb[:, j].astype(np.int32) * 256 + xb[:, j + 1]) % H
                logits += B[hidx, :]
            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            dlogits = probs / bsz
            # gradients
            gU = np.zeros_like(U)
            for j in range(d):
                one_hot = np.eye(256, dtype=np.float32)[xb[:, j]]
                gU[j] = one_hot.T @ dlogits
            gU += L2 * U
            all_hidx = np.concatenate([(xb[:, j].astype(np.int32) * 256 + xb[:, j + 1]) % H for j in range(d - 1)])
            all_dlogits = np.repeat(dlogits, d - 1, axis=0)
            one_hot_b = np.eye(H, dtype=np.float32)[all_hidx]
            gB = one_hot_b.T @ all_dlogits
            gB += L2 * B
            gbias = dlogits.sum(axis=0)
            grads = [gU, gB, gbias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    # evaluate
    n_val = X_val.shape[0]
    logits = bias[None, :].repeat(n_val, axis=0)
    for j in range(d):
        logits += U[j, X_val[:, j], :]
    for j in range(d - 1):
        hidx = (X_val[:, j].astype(np.int32) * 256 + X_val[:, j + 1]) % H
        logits += B[hidx, :]
    pred = logits.argmax(axis=1)
    acc = (pred == y_val).mean()
    return acc, (U, B, bias)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    best_H, best_acc, best_params = None, -1.0, None
    for H in H_CANDIDATES:
        acc, params = train_eval(card, X_tr, y_tr, X_val, y_val, H)
        print(f"H={H} val_acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc, best_H, best_params = acc, H, params
    print(f"Best H={best_H} acc={best_acc:.4f}", file=sys.stderr)
    # retrain on all data
    U, B, bias = best_params
    rng = np.random.default_rng(SEED)
    # re-initialize with same seed to be deterministic, then train on all
    U = (rng.uniform(-1, 1, (d, 256, C)) * math.sqrt(6.0 / (256 * d))).astype(np.float32)
    B = (rng.uniform(-1, 1, (best_H, C)) * math.sqrt(6.0 / best_H)).astype(np.float32)
    bias = np.zeros(C, np.float32)
    params = [U, B, bias]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bsz = xb.shape[0]
            logits = bias[None, :].repeat(bsz, axis=0)
            for j in range(d):
                logits += U[j, xb[:, j], :]
            for j in range(d - 1):
                hidx = (xb[:, j].astype(np.int32) * 256 + xb[:, j + 1]) % best_H
                logits += B[hidx, :]
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            dlogits = probs / bsz
            gU = np.zeros_like(U)
            for j in range(d):
                one_hot = np.eye(256, dtype=np.float32)[xb[:, j]]
                gU[j] = one_hot.T @ dlogits
            gU += L2 * U
            all_hidx = np.concatenate([(xb[:, j].astype(np.int32) * 256 + xb[:, j + 1]) % best_H for j in range(d - 1)])
            all_dlogits = np.repeat(dlogits, d - 1, axis=0)
            one_hot_b = np.eye(best_H, dtype=np.float32)[all_hidx]
            gB = one_hot_b.T @ all_dlogits
            gB += L2 * B
            gbias = dlogits.sum(axis=0)
            grads = [gU, gB, gbias]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    # write weights
    U, B, bias = params
    out = struct.pack("<4i", 0x48424C31, d, best_H, C)
    out += U.astype("<f4").tobytes()
    out += B.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Hashed bigram linear model: ./infer weights.bin in.rec N
 * Input record: d bytes. Output: one byte class index.
 * Weights format: int32 magic 0x48424C31, d, H, C; float32 U[d][256][C], B[H][C], bias[C]. */
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
    if (nw < 16 || hd[0] != 0x48424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * C + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *U = (const float *)(hd + 4);
    const float *B = U + (long)d * 256 * C;
    const float *bias = B + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(logits, bias, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *u = U + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) logits[c] += u[c];
        }
        for (int j = 0; j < d - 1; j++) {
            int hidx = (xi[j] * 256 + xi[j + 1]) % H;
            const float *b = B + (long)hidx * C;
            for (int c = 0; c < C; c++) logits[c] += b[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```