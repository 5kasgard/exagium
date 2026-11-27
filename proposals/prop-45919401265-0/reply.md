```hypothesis
Family: label-emb (embedding-based logistic regression). Each of the 14 byte features gets a learned 8-dim embedding; concatenated vector fed to a linear layer. This captures non-linear feature interactions cheaply and should beat label-linear (0.8524) on dev while staying faster than label-mlp16. If dev accuracy ≤ 0.8524 or latency ≥ label-mlp16, the hypothesis fails.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
BATCH = 64
MAX_EPOCHS_TUNE = 10
MAX_EPOCHS_FULL = 20
LR_GRID = [1e-3, 3e-3]
D_GRID = [4, 8, 16]
L2 = 1e-4

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_val_split(X, y, val_frac=0.2):
    n = X.shape[0]
    split = int(n * (1 - val_frac))
    return X[:split], y[:split], X[split:], y[split:]

def init_params(d, D, C, rng):
    emb = (rng.uniform(-0.1, 0.1, (d, 256, D))).astype(np.float32)
    W = (rng.uniform(-0.1, 0.1, (d * D, C)) * math.sqrt(2.0 / (d * D))).astype(np.float32)
    b = np.zeros(C, np.float32)
    return emb, W, b

def forward(X, emb, W, b):
    n, d = X.shape
    D = emb.shape[2]
    # lookup embeddings
    emb_vecs = emb[np.arange(d), X]  # (n, d, D)
    flat = emb_vecs.reshape(n, d * D)  # (n, d*D)
    logits = flat @ W + b
    return logits, flat

def loss_grad(logits, y, flat, W, emb, L2):
    n = logits.shape[0]
    # softmax + cross entropy gradient
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n
    # grad for linear layer
    gW = flat.T @ probs + L2 * W
    gb = probs.sum(axis=0)
    # grad for embeddings
    gflat = probs @ W.T  # (n, d*D)
    gemb = np.zeros_like(emb)
    gemb_reshaped = gflat.reshape(n, emb.shape[1], emb.shape[2])
    # accumulate gradients per feature per value
    for j in range(emb.shape[1]):
        idx = X[:, j]
        np.add.at(gemb[:, j, :], idx, gemb_reshaped[:, j, :])
    gemb += L2 * emb
    return gemb, gW, gb

def evaluate(X, y, emb, W, b):
    logits, _ = forward(X, emb, W, b)
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def train_loop(X, y, emb, W, b, epochs, lr, rng):
    n = X.shape[0]
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            logits, flat = forward(xb, emb, W, b)
            gemb, gW, gb = loss_grad(logits, yb, flat, W, emb, L2)
            t += 1
            # Adam updates
            for param, grad, m, v in [(emb, gemb, m_emb, v_emb),
                                      (W, gW, m_W, v_W),
                                      (b, gb, m_b, v_b)]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return emb, W, b

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    rng = np.random.default_rng(SEED)

    # split for tuning
    X_tr, y_tr, X_val, y_val = train_val_split(X, y)

    best_acc, best_params = -1, None
    for D in D_GRID:
        for lr in LR_GRID:
            emb, W, b = init_params(d, D, C, rng)
            emb, W, b = train_loop(X_tr, y_tr, emb, W, b, MAX_EPOCHS_TUNE, lr, rng)
            acc = evaluate(X_val, y_val, emb, W, b)
            if acc > best_acc:
                best_acc, best_params = acc, (D, lr, emb.copy(), W.copy(), b.copy())

    # retrain on full data with best hyperparameters
    D, lr, emb, W, b = best_params
    emb, W, b = train_loop(X, y, emb, W, b, MAX_EPOCHS_FULL, lr, rng)

    # write weights.bin
    # magic 0x4C424D45 ('EMBL'), d, D, C
    out = struct.pack("<4i", 0x4C424D45, d, D, C)
    out += emb.astype("<f4").tobytes()
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
    if (d != 14 || C != 2 || D <= 0 || D > 32) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)d * D * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * D);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            memcpy(concat + j * D, e, D * sizeof(float));
        }
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int idx = 0; idx < d * D; idx++) {
            float v = concat[idx];
            for (int c = 0; c < C; c++) scores[c] += v * W[(long)idx * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```