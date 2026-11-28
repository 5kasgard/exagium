```hypothesis
Family: label-embed (embedding MLP). Learns per-position byte embeddings (16x256xE), sums them, then a ReLU hidden layer (H) to 65-class output. Low-rank factorization of one-hot linear model reduces parameters/compute; hidden layer adds non-linearity. Should beat label-linear (0.3375) by better generalization on sparse categorical data. Failure if dev accuracy ≤ 0.34 or Ir/item > 10k.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
L2 = 1e-4
BATCH = 64
TUNE_EPOCHS = 10
FINAL_EPOCHS = 30

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def evaluate(X, y, d, C, E, H, params):
    emb, W1, b1, W2, b2 = params
    n = X.shape[0]
    emb_sum = np.zeros((n, E), dtype=np.float32)
    for j in range(d):
        emb_sum += emb[j, X[:, j]]
    h = np.maximum(emb_sum @ W1 + b1, 0)
    logits = h @ W2 + b2
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def train_model(X, y, d, C, E, H, lr, epochs, rng, init_params=None):
    if init_params is None:
        emb = (rng.normal(0, 0.01, size=(d, 256, E))).astype(np.float32)
        W1 = (rng.normal(0, math.sqrt(2.0 / E), size=(E, H))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.normal(0, math.sqrt(2.0 / H), size=(H, C))).astype(np.float32)
        b2 = np.zeros(C, np.float32)
    else:
        emb, W1, b1, W2, b2 = [p.copy() for p in init_params]
    params = [emb, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]
            yb = y[idx]
            emb_sum = np.zeros((len(xb), E), dtype=np.float32)
            for j in range(d):
                emb_sum += emb[j, xb[:, j]]
            h = np.maximum(emb_sum @ W1 + b1, 0)
            logits = h @ W2 + b2
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs.copy()
            grad_logits[np.arange(len(yb)), yb] -= 1.0
            grad_logits /= len(yb)
            grad_W2 = h.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)
            grad_h = grad_logits @ W2.T
            grad_h_pre = grad_h * (h > 0)
            grad_W1 = emb_sum.T @ grad_h_pre
            grad_b1 = grad_h_pre.sum(axis=0)
            grad_emb_sum = grad_h_pre @ W1.T
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_emb_sum)
            grad_emb += L2 * emb
            grad_W1 += L2 * W1
            grad_W2 += L2 * W2
            t += 1
            grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return params

def write_weights(path, d, C, E, H, params):
    emb, W1, b1, W2, b2 = params
    magic = 0x44424D45  # 'EMBD' little-endian
    with open(path, "wb") as f:
        f.write(struct.pack("<5i", magic, d, E, H, C))
        f.write(emb.astype("<f4").tobytes())
        f.write(W1.astype("<f4").tobytes())
        f.write(b1.astype("<f4").tobytes())
        f.write(W2.astype("<f4").tobytes())
        f.write(b2.astype("<f4").tobytes())

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]  # 16

    n = X.shape[0]
    tune_size = n // 5
    train_X, train_y = X[:-tune_size], y[:-tune_size]
    tune_X, tune_y = X[-tune_size:], y[-tune_size:]

    best_acc = -1.0
    best_params = None
    best_cfg = None

    for E in [8, 16]:
        for H in [32, 64]:
            for lr in [0.01, 0.003]:
                rng = np.random.default_rng(SEED)
                params = train_model(train_X, train_y, d, C, E, H, lr, TUNE_EPOCHS, rng)
                acc = evaluate(tune_X, tune_y, d, C, E, H, params)
                if acc > best_acc:
                    best_acc = acc
                    best_params = params
                    best_cfg = (E, H, lr)

    print(f"Best config: E={best_cfg[0]}, H={best_cfg[1]}, lr={best_cfg[2]}, tune_acc={best_acc:.4f}")

    E, H, lr = best_cfg
    rng = np.random.default_rng(SEED)
    final_params = train_model(X, y, d, C, E, H, lr, FINAL_EPOCHS, rng, init_params=best_params)
    write_weights(sys.argv[4], d, C, E, H, final_params)

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
    if (nw < 20 || hd[0] != 0x44424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20 + (long)d * 256 * E * 4 + (long)E * H * 4 + H * 4 + (long)H * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    float *emb = (float *)(hd + 5);
    float *W1 = emb + (long)d * 256 * E;
    float *b1 = W1 + (long)E * H;
    float *W2 = b1 + H;
    float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * E);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int e = 0; e < E; e++) emb_sum[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *emb_j = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) emb_sum[e] += emb_j[e];
        }
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            for (int e = 0; e < E; e++) sum += emb_sum[e] * W1[(long)e * H + u];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int u = 0; u < H; u++) sum += h[u] * W2[(long)u * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```