```hypothesis
Family: label-fm (Factorization Machine). FM models all pairwise feature interactions via low-rank embeddings, which is highly effective for tabular data with mixed categorical/numerical features. The Adult dataset has 14 features (6 ordinal quantized to 255 bins, 8 categorical) where interactions (e.g., education×occupation) are predictive. A linear model misses these; FM captures them with O(d*K) parameters (K=8) and inference cost similar to linear. Expected dev accuracy >0.858 (beating label-linear's 0.8524 and approaching label-mlp16's 0.8591) at ~1000 Ir/item. If dev accuracy ≤0.8524, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
K_CHOICES = [4, 8, 16]
LR_CHOICES = [0.01, 0.03]
L2 = 1e-4
BATCH = 256
MAX_EPOCHS = 30

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_logits(X, w0, w, v, K):
    n, d = X.shape
    linear = w0 + np.sum(w[np.arange(d), X.T], axis=0)
    sum_v = np.zeros((n, K), dtype=np.float32)
    sum_v2 = np.zeros((n, K), dtype=np.float32)
    for f in range(d):
        vals = X[:, f]
        sum_v += v[f, vals, :]
        sum_v2 += v[f, vals, :] ** 2
    interaction = 0.5 * (np.sum(sum_v * sum_v, axis=1) - np.sum(sum_v2, axis=1))
    return linear + interaction

def train_fm(X, y, K, lr, epochs, rng):
    n, d = X.shape
    w0 = 0.0
    w = np.zeros((d, 256), dtype=np.float32)
    v = rng.normal(0, 0.01, (d, 256, K)).astype(np.float32)
    m_w0, v_w0 = 0.0, 0.0
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_v = np.zeros_like(v)
    v_v = np.zeros_like(v)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            linear = w0 + np.sum(w[np.arange(d), Xb.T], axis=0)
            sum_v = np.zeros((len(yb), K), dtype=np.float32)
            sum_v2 = np.zeros((len(yb), K), dtype=np.float32)
            for f in range(d):
                vals = Xb[:, f]
                sum_v += v[f, vals, :]
                sum_v2 += v[f, vals, :] ** 2
            interaction = 0.5 * (np.sum(sum_v * sum_v, axis=1) - np.sum(sum_v2, axis=1))
            logits = linear + interaction
            probs = sigmoid(logits)
            dlogit = probs - yb
            g_w0 = np.mean(dlogit)
            g_w = np.zeros_like(w)
            for f in range(d):
                vals = Xb[:, f]
                for val in np.unique(vals):
                    mask = (vals == val)
                    g_w[f, val] = np.mean(dlogit[mask])
            g_v = np.zeros_like(v)
            for f in range(d):
                vals = Xb[:, f]
                for val in np.unique(vals):
                    mask = (vals == val)
                    if not np.any(mask):
                        continue
                    sv = sum_v[mask]
                    g_v[f, val] = np.mean(dlogit[mask, np.newaxis] * (sv - v[f, val]), axis=0)
            t += 1
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            w0 -= lr * (m_w0 / (1 - beta1**t)) / (np.sqrt(v_w0 / (1 - beta2**t)) + eps)
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            w -= lr * (m_w / (1 - beta1**t)) / (np.sqrt(v_w / (1 - beta2**t)) + eps)
            m_v = beta1 * m_v + (1 - beta1) * g_v
            v_v = beta2 * v_v + (1 - beta2) * g_v * g_v
            v -= lr * (m_v / (1 - beta1**t)) / (np.sqrt(v_v / (1 - beta2**t)) + eps)
            w -= lr * L2 * w
            v -= lr * L2 * v
    return w0, w, v

def eval_accuracy(X, y, w0, w, v, K):
    logits = compute_logits(X, w0, w, v, K)
    preds = (logits > 0).astype(np.int64)
    return np.mean(preds == y)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    rng = np.random.default_rng(SEED)
    best_acc = -1
    best_params = None
    best_model = None
    for K in K_CHOICES:
        for lr in LR_CHOICES:
            w0, w, v = train_fm(X_train, y_train, K, lr, MAX_EPOCHS, rng)
            acc = eval_accuracy(X_val, y_val, w0, w, v, K)
            if acc > best_acc:
                best_acc = acc
                best_params = (K, lr)
                best_model = (w0, w, v)
    print(f"Tuning: best K={best_params[0]}, lr={best_params[1]}, val_acc={best_acc:.4f}", file=sys.stderr)
    K, lr = best_params
    rng = np.random.default_rng(SEED)
    w0, w, v = train_fm(X, y, K, lr, MAX_EPOCHS, rng)
    MAGIC = 0x464D3130
    C = 2
    out = struct.pack("<4i", MAGIC, d, K, C)
    out += np.array([w0], dtype=np.float32).tobytes()
    out += w.astype(np.float32).tobytes()
    out += v.astype(np.float32).tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (0 or 1). */
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
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C != 2 || d != 14 || K < 1 || K > 64) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16 + 4L * (1 + (long)d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    float *p = (float *)(hd + 4);
    float w0 = p[0];
    float *w = p + 1;
    float *v = w + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = calloc(K, sizeof(float));
    float *sum_v2 = calloc(K, sizeof(float));
    unsigned char out_byte;
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        float logit = w0;
        for (int kk = 0; kk < K; kk++) {
            sum_v[kk] = 0.0f;
            sum_v2[kk] = 0.0f;
        }
        for (int f = 0; f < d; f++) {
            unsigned char val = xi[f];
            int idx = f * 256 + val;
            logit += w[idx];
            float *v_ptr = v + (long)idx * K;
            for (int kk = 0; kk < K; kk++) {
                float vf = v_ptr[kk];
                sum_v[kk] += vf;
                sum_v2[kk] += vf * vf;
            }
        }
        float interaction = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            interaction += sum_v[kk] * sum_v[kk] - sum_v2[kk];
        }
        logit += 0.5f * interaction;
        out_byte = (logit > 0.0f) ? 1 : 0;
        put_rec(stdout, &out_byte, 1);
    }
    free(sum_v);
    free(sum_v2);
    return 0;
}
```