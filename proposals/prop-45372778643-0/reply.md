```hypothesis
Family: label-fm (Factorization Machine with per-field embeddings).
Why: Adult has 14 mixed-type features; pairwise interactions (e.g., education × occupation) are known to boost tabular accuracy. FM captures all pairwise interactions in O(d·k) time using the sum-square trick, so inference stays fast. With k=8–16 we expect >0.855 dev accuracy while keeping CPU cost near label-linear.
Failure sign: If dev accuracy ≤ 0.852 (label-linear) or instructions/item > 1500, the model is not a worthwhile trade-off.
```

```python
"""label-fm: Factorization Machine for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Trains on last 20% tuning split to pick embedding dim k, then on all train.
Exports: magic 0x464D0001, d, k, C; float32 w[d*256], b[1], V[d*256*k]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, EPOCHS = 0.01, 1e-4, 256, 30
K_CANDIDATES = [4, 8, 16, 32]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fm_logits(X, w, b, V, d, k):
    """X: (n, d) uint8. Returns (n,) logits."""
    n = X.shape[0]
    lin = np.zeros(n, np.float32)
    sum_vec = np.zeros((n, k), np.float32)
    sum_sq = np.zeros(n, np.float32)
    for j in range(d):
        idx = X[:, j].astype(np.int32) + j * 256
        lin += w[idx]
        v = V[idx]          # (n, k)
        sum_vec += v
        sum_sq += np.sum(v * v, axis=1)
    inter = 0.5 * (np.sum(sum_vec * sum_vec, axis=1) - sum_sq)
    return b + lin + inter


def train_fm(X, y, d, k, rng, epochs=EPOCHS, batch=BATCH):
    n = X.shape[0]
    w = np.zeros(d * 256, np.float32)
    b = np.float32(0.0)
    V = rng.normal(0, 0.01, (d * 256, k)).astype(np.float32)
    # Adam state
    mw, mb, mV = np.zeros_like(w), np.float32(0.0), np.zeros_like(V)
    vw, vb, vV = np.zeros_like(w), np.float32(0.0), np.zeros_like(V)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            Xb, yb = X[idx], y[idx]
            # forward
            lin = np.zeros(len(yb), np.float32)
            sum_vec = np.zeros((len(yb), k), np.float32)
            sum_sq = np.zeros(len(yb), np.float32)
            for j in range(d):
                idj = Xb[:, j].astype(np.int32) + j * 256
                lin += w[idj]
                v = V[idj]
                sum_vec += v
                sum_sq += np.sum(v * v, axis=1)
            inter = 0.5 * (np.sum(sum_vec * sum_vec, axis=1) - sum_sq)
            logits = b + lin + inter
            p = sigmoid(logits)
            # gradient of BCE loss
            grad = (p - yb).astype(np.float32)  # (batch,)
            # linear grads
            gw = np.zeros_like(w)
            gb = np.float32(0.0)
            gV = np.zeros_like(V)
            for j in range(d):
                idj = Xb[:, j].astype(np.int32) + j * 256
                # linear weight grad
                np.add.at(gw, idj, grad)
                # V grads: dL/dV = grad * (sum_vec - v_j)
                vj = V[idj]          # (batch, k)
                # sum_vec for this batch
                sv = sum_vec.copy()
                # subtract vj for each sample
                sv -= vj
                # outer product: grad[:, None] * sv
                gv = grad[:, None] * sv   # (batch, k)
                np.add.at(gV, idj, gv)
            gb = grad.sum()
            # L2
            gw += L2 * w
            gV += L2 * V
            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            mw = beta1 * mw + (1 - beta1) * gw
            vw = beta2 * vw + (1 - beta2) * (gw * gw)
            mV = beta1 * mV + (1 - beta1) * gV
            vV = beta2 * vV + (1 - beta2) * (gV * gV)
            mb = beta1 * mb + (1 - beta1) * gb
            vb = beta2 * vb + (1 - beta2) * (gb * gb)
            mw_hat = mw / (1 - beta1 ** t)
            vw_hat = vw / (1 - beta2 ** t)
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            mb_hat = mb / (1 - beta1 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            w -= LR * mw_hat / (np.sqrt(vw_hat) + eps)
            V -= LR * mV_hat / (np.sqrt(vV_hat) + eps)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + eps)
    return w, b, V


def eval_acc(X, y, w, b, V, d, k):
    logits = fm_logits(X, w, b, V, d, k)
    pred = (logits > 0).astype(np.int64)
    return (pred == y).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    # tuning split: last 20%
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_k, best_acc = K_CANDIDATES[0], -1.0
    for k in K_CANDIDATES:
        rng_k = np.random.default_rng(SEED + k)
        w, b, V = train_fm(X_tr, y_tr, d, k, rng_k)
        acc = eval_acc(X_va, y_va, w, b, V, d, k)
        if acc > best_acc:
            best_acc, best_k = acc, k

    # final train on all data
    rng_final = np.random.default_rng(SEED + 1000 + best_k)
    w, b, V = train_fm(X, y, d, best_k, rng_final)

    # write weights
    magic = 0x464D0001
    out = struct.pack("<4i", magic, d, best_k, C)
    out += w.astype("<f4").tobytes()
    out += np.array([b], dtype="<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Each input record: d bytes. Output: 1 byte class (0/1). */
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D0001) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || k <= 0 || k > 256 || d != 14) { fprintf(stderr, "bad header\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 + 1 + (long)d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *w = (const float *)(hd + 4);
    const float *b = w + (long)d * 256;
    const float *V = b + 1;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_vec = malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float logit = b[0];
        memset(sum_vec, 0, sizeof(float) * k);
        float sum_sq = 0.0f;
        for (int j = 0; j < d; j++) {
            int idx = j * 256 + xi[j];
            logit += w[idx];
            const float *v = V + (long)idx * k;
            for (int u = 0; u < k; u++) sum_vec[u] += v[u];
            float norm = 0.0f;
            for (int u = 0; u < k; u++) norm += v[u] * v[u];
            sum_sq += norm;
        }
        float dot_sum = 0.0f;
        for (int u = 0; u < k; u++) dot_sum += sum_vec[u] * sum_vec[u];
        logit += 0.5f * (dot_sum - sum_sq);
        unsigned char pred = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    free(sum_vec);
    return 0;
}
```