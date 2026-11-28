```hypothesis
Family: label-fm (Factorization Machine). FM models pairwise feature interactions via low-rank embeddings, which are crucial for tabular data with mixed categorical/ordinal features (e.g., education × occupation). Inference cost is O(d·k) (d=14, k=8 → ~224 multiply-adds), comparable to label-linear. If dev accuracy does not exceed label-linear (0.8524) or match label-mlp16 (0.8591) at similar cost, then pairwise interactions are not the limiting factor.
```
```python
import json, math, struct, sys
import numpy as np

K_CHOICES = [4, 8, 16]
LR_CHOICES = [0.01, 0.03, 0.1]
L2_CHOICES = [1e-4, 1e-3]
BATCH = 256
MAX_EPOCHS = 50
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_fm(X, y, d, k, lr, l2, rng, epochs):
    n = X.shape[0]
    w0 = 0.0
    w = np.zeros((d, 256), dtype=np.float32)
    V = rng.normal(0, 0.01, size=(d, 256, k)).astype(np.float32)
    m_w0, v_w0 = 0.0, 0.0
    m_w = np.zeros_like(w); v_w = np.zeros_like(w)
    m_V = np.zeros_like(V); v_V = np.zeros_like(V)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            b = len(idx)
            linear = w0 + np.sum(w[np.arange(d), xb.T], axis=0)
            inter = np.zeros(b, dtype=np.float32)
            for f in range(k):
                Vf = V[:, :, f]
                sum_v = np.sum(Vf[np.arange(d), xb.T], axis=0)
                sum_v2 = np.sum(Vf[np.arange(d), xb.T]**2, axis=0)
                inter += 0.5 * (sum_v**2 - sum_v2)
            p = linear + inter
            prob = 1.0 / (1.0 + np.exp(-p))
            grad = prob - yb
            g_w0 = grad.mean() + l2 * w0
            g_w = np.zeros_like(w)
            np.add.at(g_w, (np.arange(d)[:,None], xb.T), grad)
            g_w = g_w / b + l2 * w
            g_V = np.zeros_like(V)
            for f in range(k):
                Vf = V[:, :, f]
                sum_v = np.sum(Vf[np.arange(d), xb.T], axis=0)
                for i in range(d):
                    v_i = Vf[i, xb[:, i]]
                    g_V[i, xb[:, i], f] += np.sum(grad * (sum_v - v_i))
            g_V = g_V / b + l2 * V
            t += 1
            m_w0 = beta1 * m_w0 + (1-beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1-beta2) * g_w0 * g_w0
            w0 -= lr * (m_w0 / (1 - beta1**t)) / (np.sqrt(v_w0 / (1 - beta2**t)) + eps)
            m_w = beta1 * m_w + (1-beta1) * g_w
            v_w = beta2 * v_w + (1-beta2) * g_w * g_w
            w -= lr * (m_w / (1 - beta1**t)) / (np.sqrt(v_w / (1 - beta2**t)) + eps)
            m_V = beta1 * m_V + (1-beta1) * g_V
            v_V = beta2 * v_V + (1-beta2) * g_V * g_V
            V -= lr * (m_V / (1 - beta1**t)) / (np.sqrt(v_V / (1 - beta2**t)) + eps)
    return w0, w, V

def eval_fm(X, y, d, k, w0, w, V):
    n = X.shape[0]
    linear = w0 + np.sum(w[np.arange(d), X.T], axis=0)
    inter = np.zeros(n, dtype=np.float32)
    for f in range(k):
        Vf = V[:, :, f]
        sum_v = np.sum(Vf[np.arange(d), X.T], axis=0)
        sum_v2 = np.sum(Vf[np.arange(d), X.T]**2, axis=0)
        inter += 0.5 * (sum_v**2 - sum_v2)
    p = linear + inter
    pred = (p > 0).astype(np.int64)
    return np.mean(pred == y)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    assert C == 2
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    val_size = n // 5
    train_size = n - val_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_cfg = None
    for k in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                w0, w, V = train_fm(X_train, y_train, d, k, lr, l2, rng,
                                    min(MAX_EPOCHS, math.ceil(200000 / train_size)))
                acc = eval_fm(X_val, y_val, d, k, w0, w, V)
                if acc > best_acc:
                    best_acc, best_cfg = acc, (k, lr, l2)
    k, lr, l2 = best_cfg
    rng = np.random.default_rng(SEED)
    w0, w, V = train_fm(X, y, d, k, lr, l2, rng,
                        min(MAX_EPOCHS, math.ceil(200000 / n)))
    magic = 0x464D4C31
    out = struct.pack("<4i", magic, d, k, C)
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || d != 14) { fprintf(stderr, "unexpected dims\n"); return 3; }
    long expected_floats = 1L + (long)d * 256 * (1 + k);
    if (nw != 16 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *params = (const float *)(hd + 4);
    float w0 = params[0];
    const float *w = params + 1;
    const float *V = w + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    unsigned char out_byte;
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float score = w0;
        for (int j = 0; j < d; j++) score += w[j * 256 + xi[j]];
        for (int f = 0; f < k; f++) {
            float sum_v = 0.0f, sum_v2 = 0.0f;
            for (int j = 0; j < d; j++) {
                float v = V[((long)j * 256 + xi[j]) * k + f];
                sum_v += v; sum_v2 += v * v;
            }
            score += 0.5f * (sum_v * sum_v - sum_v2);
        }
        out_byte = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    return 0;
}
```