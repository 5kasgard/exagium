```hypothesis
Factorization Machine (FM) family for tabular classification. FM captures pairwise feature interactions via low-rank embeddings, which is effective for categorical/ordinal tabular data like Adult. With 14 features and rank 4, it adds only ~14k parameters over linear, keeping inference fast (O(d*k)). Expected to beat label-linear (0.8524) on dev accuracy while staying cheaper than label-mlp16 (2325 Ir/item). If dev accuracy ≤ 0.855 or Ir/item > 2000, the model does not meet its goal.
```
```python
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split (last 20% of train)
K_CHOICES = [2, 4, 8]
LR_CHOICES = [0.01, 0.03]
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_fm(X, y, k, lr, epochs, batch, rng):
    n, d = X.shape
    # Parameters: bias, w[d,256], v[d,256,k]
    w = rng.normal(0, 0.01, (d, 256)).astype(np.float32)
    v = rng.normal(0, 0.01, (d, 256, k)).astype(np.float32)
    bias = np.float32(0.0)
    # Adam state
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)
    m_b, v_b = np.float32(0.0), np.float32(0.0)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            x_batch = X[idx]          # (batch, d)
            y_batch = y[idx]          # (batch,)
            # Forward
            lin = bias + np.sum(w[np.arange(d), x_batch.T], axis=0)  # (batch,)
            v_batch = v[np.arange(d), x_batch.T].transpose(1,0,2)   # (batch, d, k)
            sum_v = np.sum(v_batch, axis=1)                         # (batch, k)
            sum_v2 = np.sum(v_batch**2, axis=1)                     # (batch, k)
            inter = 0.5 * np.sum(sum_v**2 - sum_v2, axis=1)         # (batch,)
            logits = lin + inter
            # Logistic loss gradient
            probs = 1.0 / (1.0 + np.exp(-logits))
            grad = probs - y_batch                                 # (batch,)
            # Gradients
            grad_bias = np.mean(grad)
            grad_w = np.zeros_like(w)
            for j in range(d):
                vals = x_batch[:, j]
                np.add.at(grad_w[j], vals, grad)
            grad_w /= batch
            grad_v = np.zeros_like(v)
            for f in range(k):
                sum_v_f = sum_v[:, f]
                for j in range(d):
                    vals = x_batch[:, j]
                    v_j = v_batch[:, j, f]
                    g = grad * (sum_v_f - v_j)
                    np.add.at(grad_v[j, :, f], vals, g)
            grad_v /= batch
            # L2 regularization
            grad_w += L2 * w
            grad_v += L2 * v
            # Adam update
            t += 1
            # bias
            m_b = 0.9 * m_b + 0.1 * grad_bias
            v_b = 0.999 * v_b + 0.001 * grad_bias * grad_bias
            bias -= lr * (m_b / (1 - 0.9**t)) / (np.sqrt(v_b / (1 - 0.999**t)) + 1e-8)
            # w
            m_w = 0.9 * m_w + 0.1 * grad_w
            v_w = 0.999 * v_w + 0.001 * grad_w * grad_w
            w -= lr * (m_w / (1 - 0.9**t)) / (np.sqrt(v_w / (1 - 0.999**t)) + 1e-8)
            # v
            m_v = 0.9 * m_v + 0.1 * grad_v
            v_v = 0.999 * v_v + 0.001 * grad_v * grad_v
            v -= lr * (m_v / (1 - 0.9**t)) / (np.sqrt(v_v / (1 - 0.999**t)) + 1e-8)
    return bias, w, v

def evaluate(X, y, bias, w, v):
    n, d = X.shape
    lin = bias + np.sum(w[np.arange(d), X.T], axis=0)
    v_batch = v[np.arange(d), X.T].transpose(1,0,2)
    sum_v = np.sum(v_batch, axis=1)
    sum_v2 = np.sum(v_batch**2, axis=1)
    inter = 0.5 * np.sum(sum_v**2 - sum_v2, axis=1)
    logits = lin + inter
    preds = (logits > 0).astype(np.int64)
    return np.mean(preds == y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    # Split: last 20% for tuning
    n = X.shape[0]
    tune_size = n // 5
    train_X, tune_X = X[:-tune_size], X[-tune_size:]
    train_y, tune_y = y[:-tune_size], y[-tune_size:]
    
    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_k, best_lr = K_CHOICES[0], LR_CHOICES[0]
    for k in K_CHOICES:
        for lr in LR_CHOICES:
            bias, w, v = train_fm(train_X, train_y, k, lr, MAX_EPOCHS, BATCH, rng)
            acc = evaluate(tune_X, tune_y, bias, w, v)
            if acc > best_acc:
                best_acc = acc
                best_k, best_lr = k, lr
    print(f"Best config: k={best_k}, lr={best_lr}, tune acc={best_acc:.4f}", file=sys.stderr)
    
    # Retrain on full data with best config (deterministic seed)
    rng = np.random.default_rng(SEED)
    bias, w, v = train_fm(X, y, best_k, best_lr, MAX_EPOCHS, BATCH, rng)
    
    # Save weights: magic, d, k, C, bias, w[d,256], v[d,256,k]
    magic = 0x464D4C31  # "FML1"
    out = struct.pack("<4i", magic, d, best_k, 2)
    out += struct.pack("<f", bias)
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
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
    if (C != 2 || d != 14 || k < 1 || k > 8) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * k;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *p = (const float *)(hd + 4);
    float bias = p[0];
    const float *w = p + 1;
    const float *v = w + (long)d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * k);
    float *sum_v2 = malloc(sizeof(float) * k);
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float score = bias;
        for (int f = 0; f < k; f++) { sum_v[f] = 0.0f; sum_v2[f] = 0.0f; }
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            score += w[(long)j * 256 + val];
            const float *vj = v + ((long)j * 256 + val) * k;
            for (int f = 0; f < k; f++) {
                float vf = vj[f];
                sum_v[f] += vf;
                sum_v2[f] += vf * vf;
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