```hypothesis
Family: label-fm (Factorization Machine with per-value embeddings). The adult dataset has 14 mixed ordinal/categorical columns with 256 possible values each. A low-rank FM (D=4) captures pairwise interactions cheaply: 14*256*4 ≈ 14k embedding params + 14*256 linear params. Training uses Adam on log-loss with L2, tuned on last 20% of train. Inference is O(d*D) ≈ 56 FLOPs/item. Should beat label-linear (0.8524) by modeling interactions; if dev accuracy ≤ 0.8524 or latency ≥ label-linear, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
D_CHOICES = [2, 4, 8]
LR_CHOICES = [0.01, 0.03]
L2_CHOICES = [1e-4, 1e-3]
EPOCHS = 20
BATCH = 256
TUNE_FRAC = 0.2

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def predict(X, bias, linear, emb, d, D):
    n = X.shape[0]
    scores = np.full(n, bias, dtype=np.float32)
    for i in range(d):
        vals = X[:, i]
        scores += linear[i, vals]
    # FM term
    sum_emb = np.zeros((n, D), dtype=np.float32)
    sum_emb_sq = np.zeros((n, D), dtype=np.float32)
    for i in range(d):
        vals = X[:, i]
        e = emb[i, vals, :]          # (n, D)
        sum_emb += e
        sum_emb_sq += e * e
    fm = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
    scores += fm
    return sigmoid(scores)

def train_one(X, y, d, D, lr, l2, epochs, batch, rng):
    n = X.shape[0]
    bias = 0.0
    linear = np.zeros((d, 256), dtype=np.float32)
    emb = rng.normal(0, 0.01, size=(d, 256, D)).astype(np.float32)
    # Adam state
    m_bias, v_bias = 0.0, 0.0
    m_lin, v_lin = np.zeros_like(linear), np.zeros_like(linear)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            xb = X[idx]
            yb = y[idx]
            bsz = len(yb)
            # forward
            scores = np.full(bsz, bias, dtype=np.float32)
            for i in range(d):
                vals = xb[:, i]
                scores += linear[i, vals]
            sum_emb = np.zeros((bsz, D), dtype=np.float32)
            sum_emb_sq = np.zeros((bsz, D), dtype=np.float32)
            for i in range(d):
                vals = xb[:, i]
                e = emb[i, vals, :]
                sum_emb += e
                sum_emb_sq += e * e
            fm = 0.5 * np.sum(sum_emb * sum_emb - sum_emb_sq, axis=1)
            scores += fm
            p = sigmoid(scores)
            grad = (p - yb).astype(np.float32)  # (bsz,)
            # gradients
            g_bias = grad.mean()
            g_lin = np.zeros_like(linear)
            g_emb = np.zeros_like(emb)
            for i in range(d):
                vals = xb[:, i]
                np.add.at(g_lin[i], vals, grad)
                e = emb[i, vals, :]  # (bsz, D)
                # dFM/de = (sum_emb - e) * grad[:, None]
                tmp = (sum_emb - e) * grad[:, None]
                np.add.at(g_emb[i], vals, tmp)
            # L2
            g_lin += l2 * linear
            g_emb += l2 * emb
            # Adam updates
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            # bias
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            m_hat = m_bias / (1 - beta1 ** t)
            v_hat = v_bias / (1 - beta2 ** t)
            bias -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # linear
            m_lin = beta1 * m_lin + (1 - beta1) * g_lin
            v_lin = beta2 * v_lin + (1 - beta2) * g_lin * g_lin
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            linear -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # emb
            m_emb = beta1 * m_emb + (1 - beta1) * g_emb
            v_emb = beta2 * v_emb + (1 - beta2) * g_emb * g_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return bias, linear, emb

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.float32)
    d = card["input"]["bytes"]
    n = X.shape[0]
    split = int(n * (1 - TUNE_FRAC))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_model = None
    for D in D_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                bias, linear, emb = train_one(X_tr, y_tr, d, D, lr, l2, EPOCHS, BATCH, rng)
                p = predict(X_te, bias, linear, emb, d, D)
                acc = np.mean((p > 0.5) == y_te)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (D, lr, l2)
                    best_model = (bias, linear, emb)
    # retrain on full data with best params
    D, lr, l2 = best_params
    rng = np.random.default_rng(SEED)
    bias, linear, emb = train_one(X, y, d, D, lr, l2, EPOCHS, BATCH, rng)

    # write weights: magic, d, D, C; then bias, linear[d][256], emb[d][256][D]
    C = card["classes"]
    out = struct.pack("<4i", 0x464D4C31, d, D, C)
    out += struct.pack("<f", bias)
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
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
    int d = hd[1], D = hd[2], C = hd[3];
    if (C != 2 || D < 1 || D > 8 || d != 14) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L + 4L * d * 256 + 4L * d * 256 * D;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *linear = p + 1;
    const float *emb = linear + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float sum_emb[8], sum_emb_sq[8];
    unsigned char best;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float score = bias;
        for (int u = 0; u < D; u++) { sum_emb[u] = 0.0f; sum_emb_sq[u] = 0.0f; }
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            score += linear[(long)j * 256 + v];
            const float *e = emb + ((long)j * 256 + v) * D;
            for (int u = 0; u < D; u++) {
                float val = e[u];
                sum_emb[u] += val;
                sum_emb_sq[u] += val * val;
            }
        }
        for (int u = 0; u < D; u++) score += 0.5f * (sum_emb[u] * sum_emb[u] - sum_emb_sq[u]);
        float p1 = 1.0f / (1.0f + expf(-score));
        best = (p1 > 0.5f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```