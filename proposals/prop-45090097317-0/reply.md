```hypothesis
Propose "label-fm": a Factorization Machine with low-rank pairwise interactions. Adult income prediction benefits from feature crosses (e.g., education × occupation); FMs capture these with O(d·k) parameters and inference cost. Using latent dim k=8, we expect >0.855 accuracy (beating label-linear's 0.8524) at similar CPU cost. Failure: dev accuracy ≤0.8524 or instructions/item >1000.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
TUNING_FRAC = 0.2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def fm_score(X, w0, W, V, k):
    # X: (n, d) uint8
    n, d = X.shape
    linear = w0 + np.sum(W[np.arange(d), X], axis=1)
    inter = 0.0
    for f in range(k):
        vf = V[:, :, f]
        s = np.sum(vf[np.arange(d), X], axis=1)
        s2 = np.sum(vf[np.arange(d), X] ** 2, axis=1)
        inter += 0.5 * (s * s - s2)
    return linear + inter

def train_fm(X, y, w0, W, V, k, lr, l2, batch, epochs, rng):
    n, d = X.shape
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mV, vV = np.zeros_like(V), np.zeros_like(V)
    mw0, vw0 = 0.0, 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            # forward
            linear = w0 + np.sum(W[np.arange(d), xb], axis=1)
            inter = np.zeros(len(xb))
            for f in range(k):
                vf = V[:, :, f]
                s = np.sum(vf[np.arange(d), xb], axis=1)
                s2 = np.sum(vf[np.arange(d), xb] ** 2, axis=1)
                inter += 0.5 * (s * s - s2)
            logits = linear + inter
            p = sigmoid(logits)
            grad = (p - yb).astype(np.float32)  # dL/dlogit
            # gradients
            gw0 = grad.mean()
            gW = np.zeros_like(W)
            gV = np.zeros_like(V)
            # linear part gradient
            for j in range(d):
                vals = xb[:, j]
                np.add.at(gW[j], vals, grad)
            gW /= len(xb)
            # interaction part gradient
            for f in range(k):
                vf = V[:, :, f]
                s = np.sum(vf[np.arange(d), xb], axis=1)  # shape (batch,)
                for j in range(d):
                    vals = xb[:, j]
                    # d(inter)/dV[j,val,f] = (s - vf[j,val]) * grad
                    gv = (s - vf[j, vals]) * grad
                    np.add.at(gV[j, :, f], vals, gv)
            gV /= len(xb)
            # L2 regularization
            gW += l2 * W
            gV += l2 * V
            gw0 += l2 * w0
            # Adam update
            t += 1
            mw0 = beta1 * mw0 + (1 - beta1) * gw0
            vw0 = beta2 * vw0 + (1 - beta2) * (gw0 * gw0)
            w0 -= lr * (mw0 / (1 - beta1 ** t)) / (np.sqrt(vw0 / (1 - beta2 ** t)) + eps)
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * (gW * gW)
            W -= lr * (mW / (1 - beta1 ** t)) / (np.sqrt(vW / (1 - beta2 ** t)) + eps)
            mV = beta1 * mV + (1 - beta1) * gV
            vV = beta2 * vV + (1 - beta2) * (gV * gV)
            V -= lr * (mV / (1 - beta1 ** t)) / (np.sqrt(vV / (1 - beta2 ** t)) + eps)
    return w0, W, V

def eval_acc(X, y, w0, W, V, k):
    scores = fm_score(X, w0, W, V, k)
    preds = (scores > 0).astype(np.int64)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = len(X_all)
    split = int(n * (1 - TUNING_FRAC))
    X_tr, X_val = X_all[:split], X_all[split:]
    y_tr, y_val = y_all[:split], y_all[split:]

    # hyperparameter search on tuning split
    best_acc = -1
    best_params = None
    best_model = None
    rng = np.random.default_rng(SEED)
    for k in [4, 8, 16]:
        for lr in [0.01, 0.03, 0.1]:
            for l2 in [1e-4, 1e-3]:
                for epochs in [20, 40]:
                    batch = 256
                    w0 = 0.0
                    W = np.zeros((d, 256), np.float32)
                    V = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)
                    w0, W, V = train_fm(X_tr, y_tr, w0, W, V, k, lr, l2, batch, epochs, rng)
                    acc = eval_acc(X_val, y_val, w0, W, V, k)
                    if acc > best_acc:
                        best_acc = acc
                        best_params = (k, lr, l2, epochs)
                        best_model = (w0.copy(), W.copy(), V.copy())
    print(f"Best tuning acc: {best_acc:.4f} params: {best_params}", file=sys.stderr)

    # retrain on full data with best params
    k, lr, l2, epochs = best_params
    w0, W, V = best_model
    # continue training on full data for a few more epochs
    w0, W, V = train_fm(X_all, y_all, w0, W, V, k, lr, l2, 256, 10, rng)

    # write weights.bin
    # magic: 0x464D0001, d, k, C, then w0 (float32), W (d*256 float32), V (d*256*k float32)
    out = struct.pack("<4if", 0x464D0001, d, k, C, w0)
    out += W.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D0001) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (d != 14 || C != 2 || k <= 0 || k > 32) { fprintf(stderr, "bad weights dims\n"); return 3; }
    float w0 = *(float *)(hd + 4);
    const float *W = (const float *)(hd + 5);
    const float *V = W + (long)d * 256;
    long expected = 20 + (long)d * 256 * 4 + (long)d * 256 * k * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * k);
    float *sum_v2 = malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float score = w0;
        for (int f = 0; f < k; f++) { sum_v[f] = 0.0f; sum_v2[f] = 0.0f; }
        for (int j = 0; j < d; j++) {
            int v = xi[j];
            score += W[(long)j * 256 + v];
            for (int f = 0; f < k; f++) {
                float val = V[((long)j * 256 + v) * k + f];
                sum_v[f] += val;
                sum_v2[f] += val * val;
            }
        }
        for (int f = 0; f < k; f++) {
            score += 0.5f * (sum_v[f] * sum_v[f] - sum_v2[f]);
        }
        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```