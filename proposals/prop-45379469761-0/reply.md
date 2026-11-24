```hypothesis
Propose label-fm: a Factorization Machine with latent dimension k=8 on the 14 raw byte columns. Each column value gets a linear weight and a k-d embedding; pairwise interactions are computed via the efficient (sum v)^2 - sum v^2 trick. FMs excel on sparse categorical data by modeling all feature interactions at O(d*k) cost, which should capture non-linearities missed by label-linear while staying fast. If it does not beat label-linear on dev accuracy, the interactions may be uninformative or the dataset too small for the extra parameters.
```

```python
import json, math, struct, sys

import numpy as np

# hyper-parameters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
K_CANDIDATES = [4, 8, 16]   # choose on tuning split

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def fm_score(X, bias, linear_w, emb, k):
    """X: (n, d) uint8. Returns scores (n,)."""
    n, d = X.shape
    # linear part
    lin = bias + np.sum(linear_w[np.arange(d), X.T].T, axis=1)
    # interaction part
    sum_v = np.zeros((n, k), dtype=np.float32)
    sum_v_sq = np.zeros((n, k), dtype=np.float32)
    for j in range(d):
        e = emb[j, X[:, j]]          # (n, k)
        sum_v += e
        sum_v_sq += e * e
    inter = 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=1)
    return lin + inter

def train_fm(X_tr, y_tr, X_val, y_val, k, rng):
    n, d = X_tr.shape
    # init
    bias = np.float32(0.0)
    linear_w = np.zeros((d, 256), dtype=np.float32)
    emb = (rng.normal(0, 0.01, (d, 256, k)) * np.sqrt(2.0 / k)).astype(np.float32)
    params = [bias, linear_w, emb]
    # Adam state
    m_bias, v_bias = np.float32(0.0), np.float32(0.0)
    m_lin, v_lin = np.zeros_like(linear_w), np.zeros_like(linear_w)
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    t = 0
    best_val = -1.0
    best_params = None
    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X_tr[idx]
            yb = y_tr[idx]
            # forward
            scores = fm_score(xb, bias, linear_w, emb, k)
            p = sigmoid(scores)
            # gradient of binary cross-entropy
            grad = (p - yb).astype(np.float32) / len(yb)   # (batch,)
            # linear gradients
            g_bias = np.sum(grad)
            g_lin = np.zeros_like(linear_w)
            g_emb = np.zeros_like(emb)
            for j in range(d):
                vals = xb[:, j]
                np.add.at(g_lin[j], vals, grad)
            # interaction gradients
            # compute sum_v and sum_v_sq for this batch
            batch_size = len(idx)
            sum_v = np.zeros((batch_size, k), dtype=np.float32)
            sum_v_sq = np.zeros((batch_size, k), dtype=np.float32)
            for j in range(d):
                e = emb[j, xb[:, j]]
                sum_v += e
                sum_v_sq += e * e
            # gradient w.r.t each embedding vector
            # dL/dv = grad * (sum_v - v)
            for j in range(d):
                vals = xb[:, j]
                v = emb[j, vals]
                # (batch, k) * (batch, 1) -> (batch, k)
                g = grad[:, None] * (sum_v - v)
                np.add.at(g_emb[j], vals, g)
            # Adam update
            t += 1
            # bias
            m_bias = 0.9 * m_bias + 0.1 * g_bias
            v_bias = 0.999 * v_bias + 0.001 * g_bias * g_bias
            bias -= LR * (m_bias / (1 - 0.9**t)) / (np.sqrt(v_bias / (1 - 0.999**t)) + 1e-8)
            # linear
            m_lin = 0.9 * m_lin + 0.1 * g_lin
            v_lin = 0.999 * v_lin + 0.001 * g_lin * g_lin
            linear_w -= LR * (m_lin / (1 - 0.9**t)) / (np.sqrt(v_lin / (1 - 0.999**t)) + 1e-8)
            # emb
            m_emb = 0.9 * m_emb + 0.1 * g_emb
            v_emb = 0.999 * v_emb + 0.001 * g_emb * g_emb
            emb -= LR * (m_emb / (1 - 0.9**t)) / (np.sqrt(v_emb / (1 - 0.999**t)) + 1e-8)
        # validation
        val_scores = fm_score(X_val, bias, linear_w, emb, k)
        val_pred = (val_scores > 0).astype(np.int64)
        val_acc = np.mean(val_pred == y_val)
        if val_acc > best_val:
            best_val = val_acc
            best_params = (bias.copy(), linear_w.copy(), emb.copy())
    return best_params, best_val

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    assert d == 14
    rng = np.random.default_rng(SEED)
    # tuning split: last 20%
    n = len(X)
    split = int(0.8 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    best_k = 8
    best_val_acc = -1.0
    best_model = None
    for k in K_CANDIDATES:
        model, val_acc = train_fm(X_tr, y_tr, X_val, y_val, k, rng)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_k = k
            best_model = model
    # retrain on full data with best_k
    print(f"chosen k={best_k} val_acc={best_val_acc:.4f}", file=sys.stderr)
    final_model, _ = train_fm(X, y, X_val, y_val, best_k, rng)  # use same function but on full data
    bias, linear_w, emb = final_model
    # write weights
    # magic 0x464D4C31, d, k, C
    out = struct.pack("<4i", 0x464D4C31, d, best_k, 2)
    out += np.array([bias], dtype="<f4").tobytes()
    out += linear_w.astype("<f4").tobytes()
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
    if (d != 14 || C != 2 || k <= 0 || k > 32) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * k);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *linear_w = p + 1;
    const float *emb = linear_w + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float linear = bias;
        float sum_v[32] = {0};
        float sum_v_sq[32] = {0};
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            linear += linear_w[(long)j * 256 + v];
            const float *e = emb + ((long)j * 256 + v) * k;
            for (int u = 0; u < k; u++) {
                float val = e[u];
                sum_v[u] += val;
                sum_v_sq[u] += val * val;
            }
        }
        float inter = 0.0f;
        for (int u = 0; u < k; u++) inter += 0.5f * (sum_v[u] * sum_v[u] - sum_v_sq[u]);
        float score = linear + inter;
        unsigned char best = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```