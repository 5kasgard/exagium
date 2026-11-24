```hypothesis
Factorization machine (FM) with K=4 embeddings per field captures pairwise interactions between the 14 mixed-type columns. It should beat label-linear (0.8524) on Adult because the dataset is known to benefit from feature crosses, while inference stays O(d·K) ≈ 56 FLOPs/item – cheaper than label-mlp16 (2325 Ir/item). Failure: dev accuracy ≤ 0.855 or Ir/item > 2000.
```
```python
"""Factorization-machine family for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: per-column tables of 256 linear weights + K-dim embeddings (precomputed for every byte value).
Ordinal columns are z-scored on training data; categorical columns use training categories + 255=unseen.
Training: Adam on logistic loss, L2 on weights+embeddings. Hyper-parameters chosen on last 20% of train.in."""
import json, math, struct, sys, os

import numpy as np

SEED = 1
K_CHOICES = [4, 8]
LR_CHOICES = [0.01, 0.03]
L2_CHOICES = [1e-4, 1e-3]
EPOCH_CHOICES = [10, 20]
BATCH = 256
SAMPLES = 600000
MAX_EPOCHS = 50

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_encoders(X, types):
    """Return list of (lin_table[256], emb_table[256,K], is_ordinal) per column."""
    d = X.shape[1]
    enc = []
    for j in range(d):
        col = X[:, j]
        if types[j] == "ordinal":
            mu = col.mean()
            sd = col.std()
            if sd == 0: sd = 1.0
            lin = np.zeros(256, np.float32)
            emb = np.zeros((256, 1), np.float32)  # K=1 placeholder, will expand later
            for v in range(256):
                z = (v - mu) / sd
                lin[v] = 0.0          # linear weight learned later
                emb[v, 0] = z         # embedding = scaled value (single dim)
            enc.append((lin, emb, True, mu, sd))
        else:
            vals, inv = np.unique(col, return_inverse=True)
            n_cat = len(vals)
            mapping = np.full(256, n_cat, dtype=np.int32)  # 255 -> unseen index n_cat
            mapping[vals] = np.arange(n_cat)
            lin = np.zeros(256, np.float32)
            emb = np.zeros((256, 1), np.float32)  # placeholder
            enc.append((lin, emb, False, mapping, n_cat + 1))
    return enc

def expand_embeddings(enc, K):
    """Expand embedding dimension from 1 to K for ordinal, allocate K for categorical."""
    new_enc = []
    for lin, emb, is_ord, *rest in enc:
        if is_ord:
            mu, sd = rest
            embK = np.zeros((256, K), np.float32)
            embK[:, 0] = emb[:, 0]
            new_enc.append((lin, embK, True, mu, sd))
        else:
            mapping, n_cat = rest
            embK = np.zeros((256, K), np.float32)
            new_enc.append((lin, embK, False, mapping, n_cat))
    return new_enc

def forward(X, enc, w0, K):
    """Compute scores for batch X (n,d) -> (n,2) logits."""
    n = X.shape[0]
    linear = np.full(n, w0, np.float32)
    sum_v = np.zeros((n, K), np.float32)
    sum_v2 = np.zeros((n, K), np.float32)
    for j, (lin, emb, is_ord, *_) in enumerate(enc):
        b = X[:, j]
        linear += lin[b]
        v = emb[b]                # (n,K)
        sum_v += v
        sum_v2 += v * v
    fm = 0.5 * np.sum(sum_v * sum_v - sum_v2, axis=1)
    logit = linear + fm
    # binary -> two logits [ -logit, logit ] for softmax
    return np.stack([-logit, logit], axis=1)

def loss_grad(logits, y):
    """logits (n,2), y (n,) in {0,1}. Return loss, grad_logits (n,2)."""
    n = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
    grad = probs.copy()
    grad[np.arange(n), y] -= 1.0
    grad /= n
    return loss, grad

def backward(X, enc, grad_logits, K, w0, L2):
    """Compute gradients for w0, lin_tables, emb_tables."""
    n = X.shape[0]
    gw0 = grad_logits[:, 1].sum() - grad_logits[:, 0].sum()  # d/d logit = [ -1, 1 ] for class 1
    glin = [np.zeros(256, np.float32) for _ in enc]
    gemb = [np.zeros((256, K), np.float32) for _ in enc]
    # FM gradient: d/d v = (sum_v - v) * (grad_fm) where grad_fm = d score / d fm = 1
    # score = linear + fm, so d score/d v = sum_v - v
    # grad_fm per example = grad_logits[:,1] - grad_logits[:,0]
    g_fm = grad_logits[:, 1] - grad_logits[:, 0]   # (n,)
    sum_v = np.zeros((n, K), np.float32)
    for j, (_, emb, _, *_) in enumerate(enc):
        sum_v += emb[X[:, j]]
    for j, (lin, emb, is_ord, *_) in enumerate(enc):
        b = X[:, j]
        v = emb[b]                     # (n,K)
        gv = (sum_v - v) * g_fm[:, None]   # (n,K)
        # linear part
        gl = g_fm                       # d score / d linear = 1
        np.add.at(glin[j], b, gl)
        # embedding part
        for k in range(K):
            np.add.at(gemb[j][:, k], b, gv[:, k])
        # L2
        glin[j] += L2 * lin
        gemb[j] += L2 * emb
    return gw0, glin, gemb

def train_model(X, y, enc, K, lr, L2, epochs, rng):
    n, d = X.shape
    w0 = 0.0
    # initialize linear weights small
    for lin, _, _, *_, in enc:
        lin[:] = rng.uniform(-0.01, 0.01, 256).astype(np.float32)
    # initialize embeddings small
    for _, emb, is_ord, *rest in enc:
        if is_ord:
            mu, sd = rest
            for v in range(256):
                z = (v - mu) / sd
                emb[v, 0] = z
                for k in range(1, K):
                    emb[v, k] = rng.uniform(-0.01, 0.01)
        else:
            mapping, n_cat = rest
            for c in range(n_cat):
                emb[mapping == c] = rng.uniform(-0.01, 0.01, K).astype(np.float32)
    # Adam state
    mw0, vw0 = 0.0, 0.0
    mlin = [np.zeros(256, np.float32) for _ in enc]
    vlin = [np.zeros(256, np.float32) for _ in enc]
    memb = [np.zeros((256, K), np.float32) for _ in enc]
    vemb = [np.zeros((256, K), np.float32) for _ in enc]
    t = 0
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            logits = forward(xb, enc, w0, K)
            _, grad = loss_grad(logits, yb)
            gw0, glin, gemb = backward(xb, enc, grad, K, w0, L2)
            t += 1
            # Adam update
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            # w0
            mw0 = beta1 * mw0 + (1 - beta1) * gw0
            vw0 = beta2 * vw0 + (1 - beta2) * gw0 * gw0
            w0 -= lr * (mw0 / (1 - beta1**t)) / (np.sqrt(vw0 / (1 - beta2**t)) + eps)
            for j in range(d):
                mlin[j] = beta1 * mlin[j] + (1 - beta1) * glin[j]
                vlin[j] = beta2 * vlin[j] + (1 - beta2) * glin[j] * glin[j]
                lin = enc[j][0]
                lin -= lr * (mlin[j] / (1 - beta1**t)) / (np.sqrt(vlin[j] / (1 - beta2**t)) + eps)
                memb[j] = beta1 * memb[j] + (1 - beta1) * gemb[j]
                vemb[j] = beta2 * vemb[j] + (1 - beta2) * gemb[j] * gemb[j]
                emb = enc[j][1]
                emb -= lr * (memb[j] / (1 - beta1**t)) / (np.sqrt(vemb[j] / (1 - beta2**t)) + eps)
    return w0, enc

def evaluate(X, y, enc, w0, K):
    logits = forward(X, enc, w0, K)
    pred = logits[:, 1] > logits[:, 0]
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    n = X.shape[0]
    # split last 20% for tuning
    split = n - n // 5
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]
    rng = np.random.default_rng(SEED)
    # build base encoders (K=1) on training split
    base_enc = build_encoders(X_train, types)
    best_acc = -1.0
    best_cfg = None
    best_w0 = None
    best_enc = None
    for K in K_CHOICES:
        for lr in LR_CHOICES:
            for L2 in L2_CHOICES:
                for epochs in EPOCH_CHOICES:
                    enc = expand_embeddings(base_enc, K)
                    w0, enc = train_model(X_train, y_train, enc, K, lr, L2, epochs, rng)
                    acc = evaluate(X_tune, y_tune, enc, w0, K)
                    if acc > best_acc:
                        best_acc = acc
                        best_cfg = (K, lr, L2, epochs)
                        best_w0 = w0
                        best_enc = enc
    print(f"Best cfg: K={best_cfg[0]}, lr={best_cfg[1]}, L2={best_cfg[2]}, epochs={best_cfg[3]}, tune_acc={best_acc:.4f}", file=sys.stderr)
    # retrain on all data with best config
    base_enc_full = build_encoders(X, types)
    enc_full = expand_embeddings(base_enc_full, best_cfg[0])
    rng = np.random.default_rng(SEED)
    w0_final, enc_final = train_model(X, y, enc_full, *best_cfg, rng)
    # pack weights: magic, d, K, C=2, w0, then per column: lin[256], emb[256*K]
    magic = 0x464D4C31  # 'FML1'
    C = 2
    K = best_cfg[0]
    out = struct.pack("<4i", magic, d, K, C)
    out += struct.pack("<f", w0_final)
    for lin, emb, *_ in enc_final:
        out += lin.astype("<f4").tobytes()
        out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* Factorization-machine inference: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte (0 or 1). */
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D4C31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K < 1 || K > 16) { fprintf(stderr, "bad header\n"); return 3; }
    float *ptr = (float *)(hd + 4);
    float w0 = *ptr++;
    size_t expected = 1 + (size_t)d * (256 + 256 * K);
    if (nw != 16 + 4L * expected) { fprintf(stderr, "bad weights size %ld vs %zu\n", nw, 16 + 4L * expected); return 3; }
    float (*lin)[256] = (float (*)[256])ptr;
    ptr += d * 256;
    float (*emb)[256][16] = (float (*)[256][16])ptr;  // K <= 16
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_v[16], sum_v2[16];
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float linear = w0;
        for (int u = 0; u < K; u++) { sum_v[u] = 0.0f; sum_v2[u] = 0.0f; }
        for (int j = 0; j < d; j++) {
            unsigned char b = xi[j];
            linear += lin[j][b];
            for (int u = 0; u < K; u++) {
                float v = emb[j][b][u];
                sum_v[u] += v;
                sum_v2[u] += v * v;
            }
        }
        float fm = 0.0f;
        for (int u = 0; u < K; u++) fm += sum_v[u] * sum_v[u] - sum_v2[u];
        fm *= 0.5f;
        float score = linear + fm;
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```