```hypothesis
Family: label-emb (embedding-based linear model). Learns a dense embedding (dim D) for each of the 16 context positions and 256 byte values, sums the 16 embeddings, and applies a linear classifier to 65 classes. This factorizes the full one-hot weight matrix (16*256*65) into (16*256*D + D*65), capturing character similarities and reducing overfitting. D is chosen on the tuning split (last 20% of train). Expected to exceed label-linear's 0.3375 dev accuracy. Failure if accuracy <= 0.3375 or inference cost > label-linear.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameters (fixed)
SEED = 1
LR = 0.01
L2 = 1e-4
BATCH = 64
EPOCHS = 20
TUNE_EPOCHS = 10
D_CANDIDATES = [16, 32, 64]

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def train_model(X, y, indices, D, epochs, rng_seed):
    n = len(indices)
    rng = np.random.default_rng(rng_seed)
    C = 65
    d = X.shape[1]
    E = rng.normal(0, 0.01, size=(d, 256, D)).astype(np.float32)
    W = rng.normal(0, 0.01, size=(D, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    mE, vE = np.zeros_like(E), np.zeros_like(E)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    t = 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[indices[idx]]
            yb = y[indices[idx]]
            B = xb.shape[0]
            emb_sum = np.zeros((B, D), dtype=np.float32)
            for pos in range(d):
                emb_sum += E[pos, xb[:, pos], :]
            logits = emb_sum @ W + b
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(B), yb] -= 1.0
            grad_logits /= B
            grad_W = emb_sum.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(axis=0)
            grad_emb_sum = grad_logits @ W.T
            grad_E = np.zeros_like(E)
            for pos in range(d):
                np.add.at(grad_E[pos], xb[:, pos], grad_emb_sum)
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
    return E, W, b

def evaluate(X, y, indices, E, W, b):
    d = X.shape[1]
    correct = 0
    total = len(indices)
    for idx in indices:
        xb = X[idx:idx+1]
        yb = y[idx]
        emb_sum = np.zeros(E.shape[2], dtype=np.float32)
        for pos in range(d):
            emb_sum += E[pos, xb[0, pos], :]
        logits = emb_sum @ W + b
        pred = np.argmax(logits)
        if pred == yb:
            correct += 1
    return correct / total

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X_bytes = b"".join(records(sys.argv[2]))
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, d)
    y_bytes = b"".join(records(sys.argv[3]))
    y = np.frombuffer(y_bytes, dtype=np.uint8)
    n = X.shape[0]
    tune_size = n // 5
    train_indices = np.arange(n - tune_size)
    tune_indices = np.arange(n - tune_size, n)
    best_D = D_CANDIDATES[0]
    best_acc = -1.0
    for D in D_CANDIDATES:
        E, W, b = train_model(X, y, train_indices, D, TUNE_EPOCHS, SEED)
        acc = evaluate(X, y, tune_indices, E, W, b)
        if acc > best_acc:
            best_acc = acc
            best_D = D
    all_indices = np.arange(n)
    E, W, b = train_model(X, y, all_indices, best_D, EPOCHS, SEED)
    magic = 0x4C424D45
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
    if (nw < 16 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    long expected = 16L + 4L * ((long)d * 256 * D + (long)D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *E = (const float *)(hd + 4);
    const float *W = E + (long)d * 256 * D;
    const float *b = W + (long)D * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * D);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < D; j++) emb_sum[j] = 0.0f;
        for (int p = 0; p < d; p++) {
            const float *e = E + ((long)p * 256 + xi[p]) * D;
            for (int j = 0; j < D; j++) emb_sum[j] += e[j];
        }
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int j = 0; j < D; j++) sum += emb_sum[j] * W[(long)j * C + c];
            scores[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```