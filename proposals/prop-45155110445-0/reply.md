```hypothesis
Factorization Machine (FM) with shared embeddings captures pairwise feature interactions
crucial for tabular data like Adult, while staying lightweight. Treating all 14 columns as
categorical (256 values each) with a small embedding dim K=8 gives ~36k params. On the
tuning split (last 20% of train.in) we grid-search K∈{4,8,16} and lr∈{1e-3,3e-3}, then
retrain on all data. Expected dev accuracy >0.86 (beating label-linear's 0.8524) at similar
inference cost (~O(d·K) per item). Failure: if best dev accuracy ≤0.8524 or inference
instructions/item exceed label-linear by >2×.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split
K_CHOICES = [4, 8, 16]
LR_CHOICES = [1e-3, 3e-3]
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def forward(X_batch, w0, W_lin, V, d, K, C):
    B = X_batch.shape[0]
    # linear part
    linear = np.broadcast_to(w0, (B, C)).copy()
    for j in range(d):
        linear += W_lin[:, j, X_batch[:, j]].T
    # interaction part
    sum_v = np.zeros((B, K), dtype=np.float32)
    sum_v_sq = np.zeros((B, K), dtype=np.float32)
    for j in range(d):
        v = V[j, X_batch[:, j], :]
        sum_v += v
        sum_v_sq += v * v
    interaction = 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=1)
    logits = linear + interaction[:, None]
    return logits

def loss_and_grad(X_batch, y_batch, w0, W_lin, V, d, K, C, L2):
    B = X_batch.shape[0]
    # forward
    linear = np.broadcast_to(w0, (B, C)).copy()
    for j in range(d):
        linear += W_lin[:, j, X_batch[:, j]].T
    sum_v = np.zeros((B, K), dtype=np.float32)
    sum_v_sq = np.zeros((B, K), dtype=np.float32)
    for j in range(d):
        v = V[j, X_batch[:, j], :]
        sum_v += v
        sum_v_sq += v * v
    interaction = 0.5 * np.sum(sum_v * sum_v - sum_v_sq, axis=1)
    logits = linear + interaction[:, None]

    # softmax + cross-entropy
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits, dtype=np.float32)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(B), y_batch] + 1e-12).mean()

    # gradient wrt logits
    dlogits = probs
    dlogits[np.arange(B), y_batch] -= 1.0
    dlogits /= B

    # gradients
    dw0 = dlogits.sum(axis=0)
    dW_lin = np.zeros_like(W_lin)
    for j in range(d):
        for v in range(256):
            mask = (X_batch[:, j] == v)
            if mask.any():
                dW_lin[:, j, v] = dlogits[mask].sum(axis=0)

    dV = np.zeros_like(V)
    dinteraction = dlogits.sum(axis=1)  # (B,)
    for j in range(d):
        for v in range(256):
            mask = (X_batch[:, j] == v)
            if mask.any():
                # sum_v for masked samples
                sv = sum_v[mask]  # (m, K)
                v_vec = V[j, v]   # (K,)
                # grad = (sv - v_vec) * dinteraction[:,None]
                dV[j, v] = ((sv - v_vec) * dinteraction[mask, None]).sum(axis=0)

    # L2 regularization
    dw0 += L2 * w0
    dW_lin += L2 * W_lin
    dV += L2 * V

    return loss, dw0, dW_lin, dV

def train_loop(X, y, d, K, C, lr, epochs, rng):
    n = X.shape[0]
    # init
    w0 = np.zeros(C, np.float32)
    W_lin = np.zeros((C, d, 256), np.float32)
    V = (rng.uniform(-0.01, 0.01, (d, 256, K)) * math.sqrt(6.0 / K)).astype(np.float32)
    params = [w0, W_lin, V]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            loss, dw0, dW_lin, dV = loss_and_grad(Xb, yb, w0, W_lin, V, d, K, C, L2)
            grads = [dw0, dW_lin, dV]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return w0, W_lin, V

def evaluate(X, y, w0, W_lin, V, d, K, C):
    logits = forward(X, w0, W_lin, V, d, K, C)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # tuning split: last 20% in file order
    val_size = n // 5
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))

    best_acc = -1.0
    best_params = None
    best_K = best_lr = None

    for K in K_CHOICES:
        for lr in LR_CHOICES:
            w0, W_lin, V = train_loop(X_train, y_train, d, K, C, lr, epochs, rng)
            acc = evaluate(X_val, y_val, w0, W_lin, V, d, K, C)
            if acc > best_acc:
                best_acc = acc
                best_params = (w0.copy(), W_lin.copy(), V.copy())
                best_K, best_lr = K, lr

    print(f"Tuning: best K={best_K}, lr={best_lr}, val_acc={best_acc:.4f}", file=sys.stderr)

    # retrain on all data
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    w0, W_lin, V = train_loop(X, y, d, best_K, C, best_lr, epochs_full, rng)

    # write weights.bin
    # magic: 0x464D3130 ("FM10")
    out = struct.pack("<4i", 0x464D3130, d, best_K, C)
    out += w0.astype("<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad C\n"); return 3; }
    long expected = 16L + 4L * (C + (long)C * d * 256 + (long)d * 256 * K);
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    const float *w0 = p; p += C;
    const float *W_lin = p; p += (long)C * d * 256;
    const float *V = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * K);
    float *sum_v_sq = malloc(sizeof(float) * K);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int c = 0; c < C; c++) logits[c] = w0[c];

        for (int f = 0; f < K; f++) { sum_v[f] = 0.0f; sum_v_sq[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *Wj = W_lin + (long)j * 256;
            for (int c = 0; c < C; c++) logits[c] += Wj[(long)c * d * 256 + v];
            const float *Vj = V + ((long)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float val = Vj[f];
                sum_v[f] += val;
                sum_v_sq[f] += val * val;
            }
        }

        float interaction = 0.0f;
        for (int f = 0; f < K; f++) interaction += sum_v[f] * sum_v[f] - sum_v_sq[f];
        interaction *= 0.5f;

        for (int c = 0; c < C; c++) logits[c] += interaction;

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```