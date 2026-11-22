```hypothesis
We propose label-embsum: a low-rank factorized linear model that learns a D-dimensional embedding for each (position, byte) pair, sums the 16 embeddings, and applies a linear layer to predict the next character. This captures pairwise interactions through the shared embedding space while using fewer parameters than the full linear model (16*256*D + D*65 vs 16*256*65). It should beat label-linear on accuracy because the embedding sum allows the model to combine evidence across positions non-linearly in the embedding space. If it does not beat label-linear, it would indicate that the task requires explicit higher-order interactions or non-linearities beyond a rank-constrained linear model.
```

```python
import json
import struct
import sys
import numpy as np

SEED = 1
D_CHOICES = [16, 32, 64]
LR_CHOICES = [0.01, 0.005]
TUNE_EPOCHS = 5
FINAL_EPOCHS = 20
BATCH = 64
L2 = 1e-4

def records(path):
    data = open(path, "rb").read()
    out = []
    i = 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X_train, y_train, X_val, y_val, D, LR, epochs, seed):
    rng = np.random.default_rng(seed)
    n, d = X_train.shape
    C = int(y_train.max()) + 1
    scale = np.sqrt(3.0 / D)
    E = rng.uniform(-scale, scale, (d, 256, D)).astype(np.float32)
    W = rng.uniform(-scale, scale, (D, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    mE, vE = np.zeros_like(E), np.zeros_like(E)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    indices = np.arange(n)
    for epoch in range(epochs):
        rng.shuffle(indices)
        for start in range(0, n, BATCH):
            idx = indices[start:start + BATCH]
            Xb = X_train[idx]
            yb = y_train[idx]
            B = len(yb)
            h = np.zeros((B, D), dtype=np.float32)
            for j in range(d):
                h += E[j, Xb[:, j], :]
            scores = h @ W + b
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            grad_scores = probs
            grad_scores[np.arange(B), yb] -= 1.0
            grad_scores /= B
            grad_W = h.T @ grad_scores
            grad_b = grad_scores.sum(axis=0)
            grad_h = grad_scores @ W.T
            grad_E = np.zeros_like(E)
            for j in range(d):
                np.add.at(grad_E[j], Xb[:, j], grad_h)
            grad_W += L2 * W
            grad_E += L2 * E
            t += 1
            mE = beta1 * mE + (1 - beta1) * grad_E
            vE = beta2 * vE + (1 - beta2) * (grad_E * grad_E)
            mE_hat = mE / (1 - beta1 ** t)
            vE_hat = vE / (1 - beta2 ** t)
            E -= LR * mE_hat / (np.sqrt(vE_hat) + eps)
            mW = beta1 * mW + (1 - beta1) * grad_W
            vW = beta2 * vW + (1 - beta2) * (grad_W * grad_W)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= LR * mW_hat / (np.sqrt(vW_hat) + eps)
            mb = beta1 * mb + (1 - beta1) * grad_b
            vb = beta2 * vb + (1 - beta2) * (grad_b * grad_b)
            mb_hat = mb / (1 - beta1 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            b -= LR * mb_hat / (np.sqrt(vb_hat) + eps)
    if X_val is not None:
        n_val = X_val.shape[0]
        h_val = np.zeros((n_val, D), dtype=np.float32)
        for j in range(d):
            h_val += E[j, X_val[:, j], :]
        scores_val = h_val @ W + b
        preds = scores_val.argmax(axis=1)
        val_acc = (preds == y_val).mean()
    else:
        val_acc = 0.0
    return E, W, b, val_acc

def main():
    card = json.load(open(sys.argv[1]))
    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_recs), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in y_recs], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]
    assert d == 16
    split = int(n * 0.8)
    X_train_all, y_train_all = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]
    best_acc = -1.0
    best_D = D_CHOICES[0]
    best_LR = LR_CHOICES[0]
    for D in D_CHOICES:
        for LR in LR_CHOICES:
            config_seed = SEED * 1000 + D * 10 + int(LR * 10000)
            E, W, b, acc = train_model(X_train_all, y_train_all, X_tune, y_tune, D, LR, TUNE_EPOCHS, config_seed)
            if acc > best_acc:
                best_acc = acc
                best_D = D
                best_LR = LR
    final_seed = SEED * 10000 + best_D * 10 + int(best_LR * 10000)
    E, W, b, _ = train_model(X, y, None, None, best_D, best_LR, FINAL_EPOCHS, final_seed)
    magic = 0x454D4253
    out = struct.pack("<4i", magic, d, best_D, C)
    out += E.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x454D4253) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)d * 256 * D;
    const float *b = W + (long)D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < D; j++) h[j] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = E + ((long)j * 256 + xi[j]) * D;
            for (int kk = 0; kk < D; kk++) h[kk] += e[kk];
        }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int kk = 0; kk < D; kk++) sum += h[kk] * W[(long)kk * C + c];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```